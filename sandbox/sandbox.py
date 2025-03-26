from llm_sandbox.docker import SandboxDockerSession
from llm_sandbox.podman import SandboxPodmanSession
import os

class StreamingSandboxSession:
    def __init__(self, image=None, dockerfile=None, keep_template=False, 
                 stream=True, verbose=True, runtime_configs=None, 
                 container_type=None, **kwargs):
        """
        Create a streaming sandbox session with either Docker or Podman.
        
        Args:
            image: Container image to use
            dockerfile: Path to Dockerfile (alternative to image)
            keep_template: Whether to keep the image after session is closed
            stream: Whether to stream output
            verbose: Whether to print verbose output
            runtime_configs: Additional configurations for the container
            container_type: Force a specific container type ('docker' or 'podman')
            **kwargs: Additional arguments to pass to the container session
        """
        self.verbose = verbose
        self.session = None
        
        # Determine which container technology to use
        if container_type:
            # If explicitly specified, use that
            self._initialize_session(container_type, image, dockerfile, keep_template, 
                                     verbose, runtime_configs, **kwargs)
        else:
            # Try Docker first, then Podman
            if check_docker_running():
                self._initialize_session('docker', image, dockerfile, keep_template, 
                                         verbose, runtime_configs, **kwargs)
            elif check_podman_running():
                self._initialize_session('podman', image, dockerfile, keep_template, 
                                         verbose, runtime_configs, **kwargs)
            else:
                raise RuntimeError("Neither Docker nor Podman are available. Please install and start one of them.")
    
    def _initialize_session(self, container_type, image, dockerfile, keep_template, 
                        verbose, runtime_configs, **kwargs):
        """Initialize the appropriate container session."""
        if self.verbose:
            print(f"Using {container_type} as container runtime")
            
        if container_type == 'docker':
            self.session = SandboxDockerSession(
                image=image,
                dockerfile=dockerfile,
                keep_template=keep_template,
                verbose=verbose,
                runtime_configs=runtime_configs,
                **kwargs
            )
        elif container_type == 'podman':
            # For Podman, create client with correct URI
            from podman import PodmanClient
            import os
            
            # Use custom URI based on your storage location
            uri = f"unix:/tmp/lf37cyti/podman-storage/podman/podman.sock"
            custom_client = None
            try:
                custom_client = PodmanClient(uri)
                custom_client.info()  # Test if it works
            except Exception:
                custom_client = None
                
            # Set podman image name
            podman_image = image
            if image and not dockerfile and '/' not in image:
                podman_image = f"docker.io/library/{image}"
                if verbose:
                    print(f"Using fully qualified image name for Podman: {podman_image}")
                
                # Get image ID for direct lookup
                import subprocess
                try:
                    result = subprocess.run(
                        ["podman", "images", "--quiet", podman_image],
                        capture_output=True, text=True, check=True
                    )
                    image_id = result.stdout.strip()
                    if image_id and verbose:
                        print(f"Found image with ID: {image_id}, using ID directly")
                except Exception as e:
                    if verbose:
                        print(f"Note: Couldn't get image ID: {e}")
                    
            self.session = SandboxPodmanSession(
                image=podman_image,
                dockerfile=dockerfile,
                keep_template=keep_template,
                verbose=verbose,
                runtime_configs=runtime_configs,
                client=custom_client,
                **kwargs
            )

    # Delegate methods to the underlying session
    def open(self):
        return self.session.open()
    
    def close(self):
        return self.session.close()
    
    def execute_command(self, command, workdir=None):
        return self.session.execute_command(command, workdir)
    
    def copy_to_runtime(self, src, dest):
        return self.session.copy_to_runtime(src, dest)
    
    def copy_from_runtime(self, src, dest):
        return self.session.copy_from_runtime(src, dest)
    
    def execute_command_streaming(self, command, workdir=None):
        """Stream command output from container."""
        if not self.session:
            raise RuntimeError("Session is not open")
        
        kwargs = {"stream": True, "tty": True}
        if workdir:
            kwargs["workdir"] = workdir
            
        if isinstance(self.session, SandboxDockerSession):
            _, output_stream = self.session.container.exec_run(command, **kwargs)
        else:  # PodmanSession
            _, output_stream = self.session.container.exec_run(command, **kwargs)
        
        for chunk in output_stream:
            yield chunk.decode("utf-8")

def check_docker_running():
    """Check if Docker is running and available."""
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except (ImportError, docker.errors.DockerException) as e:
        if isinstance(e, ImportError):
            print("Docker Python package not installed. Unable to use Docker.")
        else:
            print("Could not connect to Docker daemon. Is Docker Desktop running?")
            print("\nDetailed error:", str(e))
        return False

def check_podman_running():
    """Check if Podman is running and available."""
    try:
        from podman import PodmanClient
        import os
        
        # Use the same URI pattern as podman CLI with your custom storage
        uri = f"unix:/tmp/lf37cyti/podman-storage/podman/podman.sock"
        client = PodmanClient(uri)
        try:
            client.info()
            return True
        except Exception:
            # Try default socket path if custom fails
            client = PodmanClient()
            client.info()
            return True
    except (ImportError, Exception) as e:
        if isinstance(e, ImportError):
            print("Podman Python package not installed. Unable to use Podman.")
        else:
            print("Could not connect to Podman. Is Podman installed and running?")
            print("\nDetailed error:", str(e))
        return False
    
def setup_sandbox_environment(session, reinstall=False):
    """Set up the sandbox environment with required files and dependencies."""
    print("Setting up sandbox environment...")
    
    session.execute_command("mkdir -p /sandbox/workspace/systems")
    session.execute_command("mkdir -p /sandbox/workspace/agentic_system")
    session.execute_command("mkdir -p /sandbox/workspace/automated_systems")
    
    required_files = [
        ("agentic_system/virtual_agentic_system.py", "/sandbox/workspace/agentic_system/virtual_agentic_system.py"),
        ("agentic_system/large_language_model.py", "/sandbox/workspace/agentic_system/large_language_model.py"),
        ("agentic_system/materialize.py", "/sandbox/workspace/agentic_system/materialize.py"),
        ("agentic_system/udiff.py", "/sandbox/workspace/agentic_system/udiff.py"),
        ("agentic_system/target_system_template.py", "/sandbox/workspace/agentic_system/target_system_template.py"),
        ("systems/system_prompts.py", "/sandbox/workspace/systems/system_prompts.py"),
        ("systems/MetaSystem.py", "/sandbox/workspace/systems/MetaSystem.py"),
        ("sandbox/run_meta.py", "/sandbox/workspace/run_meta.py"),
        ("sandbox/run_target.py", "/sandbox/workspace/run_target.py"),
        (".env", "/sandbox/workspace/.env")
    ]
    
    for src_path, dest_path in required_files:
        if os.path.exists(src_path):
            session.copy_to_runtime(src_path, dest_path)
        else:
            print(f"Warning: Required file {src_path} not found")
    
    if reinstall:
        print("Installing dependencies in sandbox...")
        dependencies = [
            "langgraph==0.3.5", 
            "langchain_openai==0.3.8",
            "langchain_google_genai==2.0.11",
            "python-dotenv==1.0.1",
            "dill==0.3.9"
        ]
        session.execute_command(f"pip install {' '.join(dependencies)}")
    
    print("Sandbox environment set up successfully!")
    return True