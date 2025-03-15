from llm_sandbox.docker import SandboxDockerSession
import os

class StreamingSandboxSession(SandboxDockerSession):
    def execute_command_streaming(self, command, workdir=None):
        if not self.container:
            raise RuntimeError("Session is not open")

        kwargs = {"stream": True, "tty": True}
        if workdir:
            kwargs["workdir"] = workdir
            
        _, output_stream = self.container.exec_run(command, **kwargs)
        
        for chunk in output_stream:
            yield chunk.decode("utf-8")

def check_docker_running():
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except docker.errors.DockerException as e:
        print("Could not connect to Docker daemon. Is Docker Desktop running?")
        print("\nDetailed error:", str(e))
        return False
    
def setup_sandbox_environment(session, reinstall=False):
    print("Setting up sandbox environment...")
    
    session.execute_command("mkdir -p /sandbox/workspace/systems")
    session.execute_command("mkdir -p /sandbox/workspace/agentic_system")
    session.execute_command("mkdir -p /sandbox/workspace/automated_systems")
    
    required_files = [
        ("agentic_system/virtual_agentic_system.py", "/sandbox/workspace/agentic_system/virtual_agentic_system.py"),
        ("agentic_system/large_language_model.py", "/sandbox/workspace/agentic_system/large_language_model.py"),
        ("agentic_system/materialize.py", "/sandbox/workspace/agentic_system/materialize.py"),
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