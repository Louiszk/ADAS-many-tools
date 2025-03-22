import os
import sys
import json
import dill as pickle
import time
from langchain_core.messages import HumanMessage

sys.path.append('/sandbox/workspace')
from agentic_system.virtual_agentic_system import VirtualAgenticSystem
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
   
    try:
        # Optimize another agentic system
        if optimize_from_file:
            path = "/sandbox/workspace/automated_systems/" + optimize_from_file.replace("/", "_").replace("\\", "_").replace(":", "_")
            try:
                with open(path + '.pkl', 'rb') as f:
                    MetaSystem.target_system = pickle.load(f)

                MetaSystem.target_system.system_name = system_name

                print("Code initialized")
            except Exception as e:
                print(f"Error initializing: {e}")
                MetaSystem.target_system = VirtualAgenticSystem(system_name)
        else:
            MetaSystem.target_system = VirtualAgenticSystem(system_name)
       
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
                        print(f"\n[{msg_type}]: {content}\n")
                
                if "design_completed" in out and out["design_completed"]:
                    print("Design completed.")
            
            time.sleep(2)
       
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error running meta system: {str(e)}")

if __name__ == "__main__":
    main()