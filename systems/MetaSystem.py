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
from tqdm import tqdm
import traceback
import dill as pickle
import re
import sys
import subprocess
from systems import system_prompts
from agentic_system.materialize import materialize_system
target_system = None


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
        state = json.loads(state)
    
        try:
            if not (target_system.entry_point and target_system.finish_point):
                return "Error testing system: You must set an entry point and finish point before testing"
    
            source_code = materialize_system(target_system, None)
            namespace = {}
            exec(source_code, namespace, namespace)
    
            if 'build_system' not in namespace:
                return "Error: Could not find build_system function in generated code"
    
            target_workflow, _ = namespace['build_system']()
            pbar = tqdm()
    
            for output in target_workflow.stream(state):
                all_outputs.append(output)
                pbar.update(1)
    
            pbar.close()
    
        except Exception as e:
            tb_string = traceback.format_exc()
            error_message = f"\n\n Error while testing the system:\n{tb_string}"
    
        result = all_outputs if all_outputs else {}
    
        test_result = f"Test completed.\n <TestResults>\n{result}\n</TestResults>"
    
        final_output = test_result + error_message
        return final_output
    

    tools["TestSystem"] = tool(runnable=testsystem_function, name_or_callable="TestSystem")

    # Tool: ChangeCode
    # Description: Modifies the target system by applying a list of operations
    def changecode_function(operations: str) -> str:
        """Modify the target system by applying a list of operations specified in a JSON string.
    
        Args:
            operations: A JSON string containing a list of operations to apply.
    
        Returns:
            Results of the operations or error messages.
        """
        try:
            ops = json.loads(operations)
            if not isinstance(ops, list):
                ops = [ops]
            results = []
            for op in ops:
                operation = op.get("operation")
                if operation == "create_node":
                    name = op["name"]
                    function_code = op["function_code"]
                    description = op.get("description", "")
                    func = target_system.get_function(function_code)
                    target_system.create_node(name, func, description, function_code)
                    results.append(f"Created node '{name}'")
    
                elif operation == "create_tool":
                    name = op["name"]
                    description = op["description"]
                    function_code = op["function_code"]
                    func = target_system.get_function(function_code)
                    target_system.create_tool(name, description, func, function_code)
                    results.append(f"Created tool '{name}'")
    
                elif operation == "add_edge":
                    source = op["source"]
                    target = op["target"]
                    target_system.create_edge(source, target)
                    results.append(f"Added edge from '{source}' to '{target}'")
    
                elif operation == "add_conditional_edge":
                    source = op["source"]
                    condition_code = op["condition_code"]
                    path_map = op.get("path_map", None)
                    condition_func = target_system.get_function(condition_code)
                    target_system.create_conditional_edge(source, condition_func, condition_code, path_map)
                    results.append(f"Added conditional edge from '{source}'")
    
                elif operation == "set_entry_point":
                    node = op["node"]
                    target_system.set_entry_point(node)
                    results.append(f"Set entry point to '{node}'")
    
                elif operation == "set_finish_point":
                    node = op["node"]
                    target_system.set_finish_point(node)
                    results.append(f"Set finish point to '{node}'")
    
                elif operation == "delete_node":
                    node_name = op["node_name"]
                    result = target_system.delete_node(node_name)
                    if result:
                        results.append(f"Deleted node '{node_name}'")
                    else:
                        results.append(f"Failed to delete node '{node_name}'")
    
                elif operation == "delete_edge":
                    source = op["source"]
                    target = op["target"]
                    result = target_system.delete_edge(source, target)
                    if result:
                        results.append(f"Deleted edge from '{source}' to '{target}'")
                    else:
                        results.append(f"No such edge from '{source}' to '{target}'")
    
                elif operation == "delete_conditional_edge":
                    source = op["source"]
                    result = target_system.delete_conditional_edge(source)
                    if result:
                        results.append(f"Deleted conditional edge from '{source}'")
                    else:
                        results.append(f"No conditional edge found from '{source}'")
    
                elif operation == "set_state_attributes":
                    attributes = op["attributes"]
                    target_system.set_state_attributes(attributes)
                    results.append(f"Set state attributes: {attributes}")
    
                elif operation == "add_imports":
                    import_statement = op["import_statement"]
                    target_system.add_imports(import_statement)
                    results.append(f"Added import statement: '{import_statement}'")
    
                elif operation == "edit_node":
                    try:
                        name = op["name"]
                        function_code = op.get("function_code")
                        description = op.get("description")
                        func = None
                        if function_code:
                            func = target_system.get_function(function_code)
                        result = target_system.edit_node(name, func=func, description=description, source_code=function_code)
                        if result:
                            results.append(f"Edited node '{name}'")
                        else:
                            results.append(f"Failed to edit node '{name}': Node not found")
                    except Exception as e:
                        results.append(f"Error editing node '{name}': {repr(e)}")
    
                elif operation == "edit_tool":
                    try:
                        name = op["name"]
                        function_code = op.get("function_code")
                        description = op.get("description")
                        func = None
                        if function_code:
                            func = target_system.get_function(function_code)
                        result = target_system.edit_tool(name, new_function=func, new_description=description, source_code=function_code)
                        if result:
                            results.append(f"Edited tool '{name}'")
                        else:
                            results.append(f"Failed to edit tool '{name}': Tool not found")
                    except Exception as e:
                        results.append(f"Error editing tool '{name}': {repr(e)}")
    
                else:
                    results.append(f"Unknown operation: {operation}")
            return "\n".join(results)
        except Exception as e:
            return f"Error modifying system: {repr(e)}"
    

    tools["ChangeCode"] = tool(runnable=changecode_function, name_or_callable="ChangeCode")

    # Tool: EndDesign
    # Description: Finalizes the system design process
    def enddesign_function() -> str:
        """Finalize the system design process.
    
        Returns:
            Status message
        """
        try:
            if not (target_system.entry_point and target_system.finish_point):
                return "Error finalizing system: You must set an entry point and finish point before finalizing"
    
            code_dir = "sandbox/workspace/automated_systems"
            materialize_system(target_system, code_dir)
            print(f"System code materialized to {code_dir}")
    
            pickle_name = target_system.system_name.replace("/", "_").replace("\\", "_").replace(":", "_") + ".pkl"
            pickle_path = os.path.join(code_dir, pickle_name)
            with open(pickle_path, 'wb') as f:
                pickle.dump(target_system, f)
            print(f"System pickled to {pickle_path}")
    
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
        messages = state.get("messages", [])
        initial_message, current_messages = messages[0], messages[1:]
        last_messages = current_messages[-3:] if len(current_messages) >= 3 else current_messages
    
        code_message = "Current Code:\n" + materialize_system(target_system, output_dir=None)
    
        full_messages = [SystemMessage(content=system_prompts.meta_thinker), initial_message] + last_messages + [HumanMessage(content=code_message)]
        response = llm.invoke(full_messages)
        return {"messages": messages + [response]}
    

    graph.add_node("MetaThinker", metathinker_function)

    # Node: MetaExecutor
    # Description: Meta Executor Agent
    def metaexecutor_function(state: Dict[str, Any]) -> Dict[str, Any]:
        llm = LargeLanguageModel(temperature=0.4)
    
        llm.bind_tools([
            "PipInstall", "TestSystem", "ChangeCode", "EndDesign"
        ], parallel_tool_calls=False)
    
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
