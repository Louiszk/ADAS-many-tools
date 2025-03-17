# MetaSystem System Configuration
# Total nodes: 3
# Total tools: 4

from langgraph.graph import StateGraph
from langchain_core.tools import tool
import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from typing import Dict, List, Any, Callable, Optional, Union, TypeVar, Generic, Tuple, Set, TypedDict
from agentic_system.large_language_model import LargeLanguageModel, execute_tool_calls
import json
import traceback
import re
import sys
import subprocess
from systems import system_prompts

# Target system file path
target_system_file = "/sandbox/workspace/automated_systems/target_system.py"
target_system_name = "DefaultSystem"

def build_system():
    # Define state attributes for the system
    class AgentState(TypedDict):
        messages: List[Any]

    # Initialize graph with state
    graph = StateGraph(AgentState)

    # Tool definitions
    # ===== Tool Definitions =====
    tools = {}
    # Tool: PipInstall
    # Description: Securely installs a Python package using pip
    def pipinstall_function(package_name: str) -> str:
        """Securely install a Python package using pip.
    
        Args:
            package_name: Name of the package to install. Only accepts valid package names 
                        (letters, numbers, dots, underscores, dashes).
    
        Returns:
            Installation result message
        """
    
        # Validate package name to prevent command injection
        valid_pattern = r'^[a-zA-Z0-9._-]+(\s*[=<>!]=\s*[0-9a-zA-Z.]+)?$'
    
        if not re.match(valid_pattern, package_name):
            return f"Error: Invalid package name format. Package name '{package_name}' contains invalid characters."
    
        try:
            process = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                shell=False
            )
    
            if process.returncode == 0:
                return f"Successfully installed {package_name}"
            else:
                return f"Error installing {package_name}:\n{process.stdout}"
    
        except Exception as e:
            return f"Installation failed: {str(e)}"
    

    tools["PipInstall"] = tool(runnable=pipinstall_function, name_or_callable="PipInstall")

    # Tool: TestSystem
    # Description: Tests the target system with a given state
    def testsystem_function(state: str) -> str:
        """Test the target system with an input state, streaming all intermediate messages.
    
        Args:
            state: A json string with state attributes e.g. "{'messages': ['Test Input'], 'attr2': [3, 5]}"
    
        Returns:
            Test results along with intermediate outputs and any error messages encountered.
        """
        all_outputs = []
        error_message = ""
        
        try:
            state_dict = json.loads(state)
            
            namespace = {}
            with open(target_system_file, 'r') as f:
                source_code = f.read()
                
            if "set_entry_point" not in source_code or "set_finish_point" not in source_code:
                return "Error testing system: You must set an entry point and finish point before testing"
            
            exec(source_code, namespace, namespace)
            
            if 'build_system' not in namespace:
                return "Error: Could not find build_system function in generated code"
            
            target_workflow, _ = namespace['build_system']()
            
            for output in target_workflow.stream(state_dict):
                all_outputs.append(output)
            
        except Exception as e:
            tb_string = traceback.format_exc()
            error_message = f"\n\n Error while testing the system:\n{tb_string}"
            
        result = all_outputs if all_outputs else {}
        
        test_result = f"Test completed.\n <TestResults>\n{result}\n</TestResults>"
        
        final_output = test_result + error_message
        return final_output
    

    tools["TestSystem"] = tool(runnable=testsystem_function, name_or_callable="TestSystem")

    # Tool: ChangeCode
    # Description: Modifies the target system file using a diff
    def changecode_function(diff: str) -> str:
        """Modify the target system file using a unified diff.

        Args:
            diff: A unified diff string representing the changes to make to the target system file.

        Returns:
            Status message indicating success or failure
        """
        try:
            from agentic_system.udiff import find_diffs, do_replace, hunk_to_before_after, no_match_error, SearchTextNotUnique
            
            with open(target_system_file, 'r') as f:
                content = f.read()
            
            edits = find_diffs(diff)
            
            if not edits:
                return no_match_error
            
            success = False
            failed_hunks = []

            for _, hunk in edits:
                try:
                    # Apply the diff
                    new_content = do_replace(target_system_file, content, hunk)
                    if new_content is not None:
                        content = new_content
                        success = True
                    else:
                        # failed hunks for debugging
                        before_text, _ = hunk_to_before_after(hunk)
                        failed_hunks.append({
                            "before_text": before_text[:150] + ("..." if len(before_text) > 150 else ""),
                            "hunk_lines": len(hunk),
                            "error": "no_match"
                        })
                except SearchTextNotUnique:
                    before_text, _ = hunk_to_before_after(hunk)
                    failed_hunks.append({
                        "before_text": before_text[:150] + ("..." if len(before_text) > 150 else ""),
                        "hunk_lines": len(hunk),
                        "error": "not_unique"
                    })

            if not success:
                error_msg = f"Error: Failed to apply diffs to the system.\n"
                
                for i, failed in enumerate(failed_hunks):
                    if failed.get("error") == "not_unique":
                        error_msg += f"Hunk #{i+1} matched multiple locations:\n```\n{failed['before_text']}\n```\n"
                    else:
                        error_msg += f"Hunk #{i+1} failed to match:\n```\n{failed['before_text']}\n```\n"
                
                return error_msg
            
            with open(target_system_file, 'w') as f:
                f.write(content)
            
            return f"Successfully applied diff to the system."
        except Exception as e:
            return f"Error applying diff: {repr(e)}"
    
    tools["ChangeCode"] = tool(runnable=changecode_function, name_or_callable="ChangeCode")

    # Tool: EndDesign
    # Description: Finalizes the system design process
    def enddesign_function() -> str:
        """Finalize the system design process.
    
        Returns:
            Status message
        """
        try:
            with open(target_system_file, 'r') as f:
                content = f.read()
            
            if "set_entry_point" not in content or "set_finish_point" not in content:
                return "Error finalizing system: You must set an entry point and finish point before finalizing"
            
            # We could test here again

            return "Design process completed successfully."
        except Exception as e:
            return f"Error finalizing system: {repr(e)}"
    

    tools["EndDesign"] = tool(runnable=enddesign_function, name_or_callable="EndDesign")

    # Register tools with LargeLanguageModel class
    LargeLanguageModel.register_available_tools(tools)
    
    # ===== Node Definitions =====
    # Node: MetaThinker
    # Description: Meta Thinker Agent
    def metathinker_function(state: Dict[str, Any]) -> Dict[str, Any]:
        llm = LargeLanguageModel(temperature=0.4, model_name="gemini-2.0-flash", wrapper="google")
        context_length = 4*3 # multiples of three
        messages = state.get("messages", [])
        initial_message, current_messages = messages[0], messages[1:]
        last_messages = current_messages[-context_length:] if len(current_messages) >= context_length else current_messages
        
        # Read the current content of the target system file
        with open(target_system_file, 'r') as f:
            code_content = f.read()
        
        code_message = "Current Code:\n" + code_content
        
        full_messages = [SystemMessage(content=system_prompts.meta_thinker), initial_message] + last_messages + [HumanMessage(content=code_message)]
        response = llm.invoke(full_messages)
        return {"messages": messages + [response]}
    

    graph.add_node("MetaThinker", metathinker_function)

    # Node: MetaExecutor
    # Description: Meta Executor Agent
    def metaexecutor_function(state: Dict[str, Any]) -> Dict[str, Any]:
        llm = LargeLanguageModel(temperature=0.4)
        
        llm.bind_tools(["PipInstall", "TestSystem", "ChangeCode", "EndDesign"], parallel_tool_calls=False)
        
        messages = state.get("messages", [])
        # append only the last message
        full_messages = [SystemMessage(content=system_prompts.meta_executor), messages[-1]]
        
        response = llm.invoke(full_messages)
        tool_messages, tool_results = execute_tool_calls(response)
        
        return {"messages": messages + [response] + tool_messages}
    

    graph.add_node("MetaExecutor", metaexecutor_function)

    # Node: EndDesign
    # Description: Terminal node for workflow completion
    def enddesign_function(state: Dict[str, Any]) -> Dict[str, Any]:
        return state
    

    graph.add_node("EndDesign", enddesign_function)

    # ===== Standard Edges =====
    graph.add_edge("MetaThinker", "MetaExecutor")

    # ===== Conditional Edges =====
    # Conditional Router from: MetaExecutor
    def metaexecutor_router(state: Dict[str, Any]) -> str:
        """Routes to EndDesign if that tool was called, otherwise back to MetaThinker."""
        messages = state.get("messages", [])
        for message in reversed(messages):
            if getattr(message, "type", None) == "tool" and getattr(message, "name", None) == "EndDesign":
                return "EndDesign"
        return "MetaThinker"
    

    graph.add_conditional_edges("MetaExecutor", metaexecutor_router, {'MetaThinker': 'MetaThinker', 'EndDesign': 'EndDesign'})

    # ===== Entry/Exit Configuration =====
    graph.set_entry_point("MetaThinker")
    graph.set_finish_point("EndDesign")

    # ===== Compilation =====
    workflow = graph.compile()
    return workflow, tools