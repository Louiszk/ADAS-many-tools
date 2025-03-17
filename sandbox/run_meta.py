import os
import sys
import json
import time
import shutil
from langchain_core.messages import HumanMessage

sys.path.append('/sandbox/workspace')
from systems import MetaSystem

def main():
    problem_statement = "Create me a simple system that can produce eggs."
    if len(sys.argv) >= 2:
        problem_statement = sys.argv[1]
   
    system_name = "Eggs"
    if len(sys.argv) >= 3:
        system_name = sys.argv[2]
   
    optimize_from_file = None
    if len(sys.argv) >= 4:
        optimize_from_file = sys.argv[3]
   
    print(f"Running meta system for '{system_name}'...")
    
    MetaSystem.target_system_name = system_name
    MetaSystem.target_system_file = "/sandbox/workspace/automated_systems/" + system_name.replace("/", "_").replace("\\", "_").replace(":", "_") + ".py"
    
    try:
        # Initialize target system file
        source_path = "/sandbox/workspace/agentic_system/target_system_template.py"
        initial_code = ""
        
        if optimize_from_file:
            source_path = "/sandbox/workspace/automated_systems/" + optimize_from_file.replace("/", "_").replace("\\", "_").replace(":", "_") + ".py"

        if os.path.exists(source_path):
            with open(source_path, 'r') as f:
                initial_code = f.read()
                
            with open(MetaSystem.target_system_file, 'w') as f:
                f.write(initial_code)
                
            print(f"Initialized target system from {source_path}")
        else:
            print("Target system file path does not exist.")
                
        workflow, tools = MetaSystem.build_system()
        inputs = {"messages": [HumanMessage(content=problem_statement)]}
       
        print("Streaming meta system execution...")
        for output in workflow.stream(inputs, config={"recursion_limit": 80}):
            print(list(output.keys()))
           
            for out in output.values():
                if "messages" in out:
                    messages = out["messages"]
                    if messages:
                        last_msg = messages[-1]
                        msg_type = getattr(last_msg, 'type', 'Unknown')
                        content = getattr(last_msg, 'content', '')
                        tool_calls = getattr(last_msg, 'tool_calls', '')
                        print(f"\n[{msg_type}]: {content}\n {tool_calls}")
            time.sleep(2)
       
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error running meta system: {str(e)}")

if __name__ == "__main__":
    main()