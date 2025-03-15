import json
import argparse
from sandbox.sandbox import StreamingSandboxSession, setup_sandbox_environment, check_docker_running

def run_target_system(session, system_name, state=None):
    cmd_parts = [f"python3 /sandbox/workspace/run_target.py --system_name=\"{system_name}\""]
    
    if state:
        state = json.dumps(state)
        quoted_state = state.replace('"', '\\"')
        cmd_parts.append(f'--state="{quoted_state}"')
    
    command = " ".join(cmd_parts)
    print(f"Executing: {command}")
    
    for chunk in session.execute_command_streaming(command):
        print(chunk, end="", flush=True)
    
    print("\nTarget system execution completed")

def main():
    parser = argparse.ArgumentParser(description="Run target agentic systems in a sandboxed environment")
    parser.add_argument("--system_name", help="Name of the target system to run")
    
    args = parser.parse_args()
    
    if not check_docker_running():
        return
    
    session = StreamingSandboxSession(
        image="python:3.11-slim",
        keep_template=True,
        stream=True,
        verbose=True
    )
    
    state = {"messages": ["Hello"]}
    
    try:
        session.open()
        if setup_sandbox_environment(session):
            run_target_system(
                session, 
                args.system_name, 
                state
            )
    finally:
        print("Closing session...")
        session.close()

if __name__ == "__main__":
    main()