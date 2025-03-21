import os
import argparse
from meta_system import create_meta_system
from sandbox.sandbox import StreamingSandboxSession, setup_sandbox_environment, check_docker_running

def run_meta_system_in_sandbox(session, problem_statement, target_name, optimize_system=None):
    quoted_problem = problem_statement.replace('"', '\\"')
    command = f"python3 /sandbox/workspace/run_meta.py \"{quoted_problem}\" \"{target_name}\" " + (
        f"\"{optimize_system}\"" if optimize_system else "")
    
    for chunk in session.execute_command_streaming(command):
        print(chunk, end="", flush=True)
    
    print("\nMeta system execution completed!")
    
    # Copy any generated systems back to the host
    if "automated_systems" in str(session.execute_command("ls -la /sandbox/workspace")):
        print("Copying generated systems back to host...")
        os.makedirs("automated_systems", exist_ok=True)
        target_file_name = target_name.replace("/", "_").replace("\\", "_").replace(":", "_") + ".py"
        
        if target_file_name in str(session.execute_command("ls -la /sandbox/workspace/automated_systems")):
            session.copy_from_runtime(
                f"/sandbox/workspace/automated_systems/{target_file_name}", 
                f"automated_systems/{target_file_name}"
            )
            print(f"Copied {target_file_name} back to host")
    
    return True

def main():
    prompt = """
    Design a system to solve 'Project Euler' tasks.
    Project Euler challenges participants to solve complex mathematical and computational problems
    using programming skills and mathematical insights.
    
    The system should be really small and easy for the start, just one agent and one tool.
    The tool should be really generic, just an exec tool; so that the agent can solve any problem.
    The agent should have a clear and comprehensive system prompt so that it know how to use the tool.
    """

    target_name = "SimpleEulerSolver"

    parser = argparse.ArgumentParser(description="Run agentic systems in a sandboxed environment")
    parser.add_argument("--no-keep-template", dest="keep_template", action="store_false", help="Don't keep the Docker image after the session is closed")
    parser.add_argument("--reinstall", action="store_true", help="Reinstall dependencies.")
    parser.add_argument("--materialize", action="store_true", help="Materialize meta system")
    parser.add_argument("--problem", default=prompt, help="Problem statement to solve")
    parser.add_argument("--name", default=target_name, help="Target system name")
    parser.add_argument("--optimize-system", default=None, help="Specify target system name to optimize or change")
    
    args = parser.parse_args()
    print(args)
    
    if args.materialize:
        create_meta_system()
    elif not os.path.exists("systems/MetaSystem.py"):
        print("Error: MetaSystem.py not found and --skip-materialize is set")
        return
    
    if not check_docker_running():
        return
    
    session = StreamingSandboxSession(
        # dockerfile="Dockerfile",
        image = "python:3.11-slim",
        keep_template=args.keep_template,
        stream=True,
        verbose=True
    )
    
    try:
        session.open()
        if setup_sandbox_environment(session, args.reinstall):
            run_meta_system_in_sandbox(session, args.problem, args.name, args.optimize_system)
            print("Finished successfully!")
        else:
            print("Failed to set up sandbox environment")

    finally:
        print("Session closed.")
        session.close()

if __name__ == "__main__":
    main()