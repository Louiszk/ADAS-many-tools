# Imports
from langgraph.graph import StateGraph
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from typing import Dict, List, Any, Callable, Optional, Union, TypeVar, Generic, Tuple, Set, TypedDict
from agentic_system.large_language_model import LargeLanguageModel, execute_tool_calls


def build_system():
    # Define state attributes for the system
    class AgentState(TypedDict):
        messages: List[Any]

    # Initialize graph with state
    graph = StateGraph(AgentState)

    tools = {}
    # ===== Tool Definitions =====
    # No tools defined yet
    
    def execute_python_code(code: str) -> str:
        '''Executes the given Python code and returns the output.
        
        The input is a string containing the Python code to execute.
        The output is a string containing the stdout and stderr of the execution.
        '''
        import subprocess
        import sys

        try:
            process = subprocess.Popen(
                [sys.executable, '-c', code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()
            if stderr:
                return "Error: " + stderr
            return stdout.strip()
        except Exception as e:
            return "Exception: " + str(e)

    tools["execute_python_code"] = tool(runnable=execute_python_code, name_or_callable="execute_python_code")
    # Register all tools with LargeLanguageModel class
    LargeLanguageModel.register_available_tools(tools)

    # ===== Node Definitions =====
    def agent_node(state):
        llm = LargeLanguageModel(temperature=0.1)
        system_prompt = """You are an expert Python programmer specialized in solving Project Euler problems.
        You will be given a problem description and your task is to write Python code to solve it.
        You have access to an execution tool that allows you to run Python code and get the output.
        Use the execution tool to run your code and verify the solution.
        Print the final answer in the format: "The answer is: [your answer]".
        Use print statements to debug your code.
        Verify your solution before returning it. If you already have the answer from a ToolMessage, extract the answer and return it.

       Once you are confident that the solution is correct, return the final answer.

        Here's how you should operate:
        1.  Understand the problem thoroughly.
        2.  Write Python code to solve the problem.
        3.  Use the `execute_python_code` tool to run the code and get the output.
        4.  If the output is incorrect or contains errors, revise the code and rerun it.
        5.  Use print statements to help debug.
        6.  Repeat steps 3 and 4 until you are confident that the solution is correct.
        6.  Return the final answer.
        """
        llm = llm.bind_tools(["execute_python_code"])

        messages = state.get("messages", [])
        full_messages = [SystemMessage(content=system_prompt)] + messages
        response = llm.invoke(full_messages)

        tool_messages, tool_results = execute_tool_calls(response)

        new_state = {"messages": messages + [response] + tool_messages}
        
        # Extract answer from ToolMessage if it exists
        for message in tool_messages:
            if isinstance(message, ToolMessage):
                answer = message.content
                if answer:
                    return {"messages": messages + [AIMessage(content=f"The answer is: {answer}")]} 

        return new_state 
    graph.add_node("EulerAgent", agent_node)

    def error_handler_node(state):
       # This node handles errors and decides whether to retry or finish
        messages = state.get("messages", [])
        last_message = str(messages[-1])
        if "Error:" in last_message or "Exception:" in last_message:
           # Pass the error message back to the agent
            return "EulerAgent"
        else:
           return "Finish"
    graph.add_node("ErrorHandler", error_handler_node)
    
    def finish_node(state):
        return state

    graph.add_node("Finish", finish_node)
    # ===== Entry/Exit Configuration =====
    graph.set_entry_point("EulerAgent")

    # ===== Edge Configuration =====
    graph.add_conditional_edges("EulerAgent", error_handler_node)

    graph.set_finish_point("Finish")

    graph.add_edge("ErrorHandler", "EulerAgent")


    # ===== Compilation =====
    workflow = graph.compile()
    return workflow, tools