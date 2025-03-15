# SimpleEulerSolver System Configuration
# Total nodes: 1
# Total tools: 1

from langgraph.graph import StateGraph
from langchain_core.tools import tool
import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from typing import Dict, List, Any, Callable, Optional, Union, TypeVar, Generic, Tuple, Set, TypedDict
from agentic_system.large_language_model import LargeLanguageModel, execute_tool_calls
import io, contextlib


def build_system():
    # Define state attributes for the system
    class AgentState(TypedDict):
        messages: List[Any]

    # Initialize graph with state
    graph = StateGraph(AgentState)

    # Tool definitions
    # ===== Tool Definitions =====
    tools = {}
    # Tool: execute_python_code
    # Description: Executes Python code and returns the output. Use this tool to run code that solves Project Euler problems. Input should be a valid python script.
    def execute_python_code_function(code: str) -> str:
        """Executes Python code and returns the output. Use this tool to run code that solves Project Euler problems. Input should be a valid python script."""
        try:
            # Capture stdout
            with io.StringIO() as stdout, contextlib.redirect_stdout(stdout):
                # Execute the code
                exec(code, globals())
                # Get the output
                output = stdout.getvalue()
        except Exception as e:
            output = str(e)
        return output

    tools["execute_python_code"] = tool(runnable=execute_python_code_function, name_or_callable="execute_python_code")

    # Register tools with LargeLanguageModel class
    LargeLanguageModel.register_available_tools(tools)
    # ===== Node Definitions =====
    # Node: EulerAgent
    # Description: Agent responsible for solving Project Euler problems.
    def euleragent_function(state):
            llm = LargeLanguageModel(temperature=0.0)
            system_prompt = "You are an expert Python programmer specialized in solving Project Euler problems. You will receive a problem description and your task is to write Python code to solve it, then execute the code and return the raw output of the code execution. Use the 'execute_python_code' tool to run the code. Do not include any comments or explanations in the code, just the code itself. Return the raw output of the code execution."
            llm = llm.bind_tools(["execute_python_code"])
    
            messages = state.get("messages", [])
            full_messages = [SystemMessage(content=system_prompt)] + messages
            response = llm.invoke(full_messages)
            tool_messages, tool_results = execute_tool_calls(response)
    
            raw_output = None
            if 'execute_python_code' in tool_results and tool_results['execute_python_code']:
                raw_output = tool_results['execute_python_code'][0]
    
            new_state = {"messages": messages + [response] + tool_messages, "raw_output": raw_output}
            return new_state

    graph.add_node("EulerAgent", euleragent_function)

    # ===== Entry/Exit Configuration =====
    graph.set_entry_point("EulerAgent")
    graph.set_finish_point("EulerAgent")

    # ===== Compilation =====
    workflow = graph.compile()
    return workflow, tools
