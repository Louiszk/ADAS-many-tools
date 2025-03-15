# SimpleEulerSolver System Configuration
# Total nodes: 1
# Total tools: 1

from langgraph.graph import StateGraph
from langchain_core.tools import tool
import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from typing import Dict, List, Any, Callable, Optional, Union, TypeVar, Generic, Tuple, Set, TypedDict
from agentic_system.large_language_model import LargeLanguageModel, execute_tool_calls
import subprocess


def build_system():
    # Define state attributes for the system
    class AgentState(TypedDict):
        messages: List[Any]
        problem_description: str
        code: str
        result: str

    # Initialize graph with state
    graph = StateGraph(AgentState)

    # Tool definitions
    # ===== Tool Definitions =====
    tools = {}
    # Tool: ExecuteCode
    # Description: Executes the given Python code and returns the output.
    def executecode_function(code: str) -> str:
        """Executes the given Python code and returns the output.
    
        Args:
            code (str): The Python code to execute.
    
        Returns:
            str: The output of the code execution, or an error message if execution fails.
        """
        try:
            process = subprocess.Popen(['python3', '-c', code],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       text=True)
            stdout, stderr = process.communicate(timeout=10)  # Add a timeout to prevent infinite loops
    
            if stderr:
                return f"Error: {stderr}"
            else:
                return stdout.strip()
        except subprocess.TimeoutExpired:
            process.kill()
            return "Error: Code execution timed out."
        except Exception as e:
            return f"Error: {str(e)}"

    tools["ExecuteCode"] = tool(runnable=executecode_function, name_or_callable="ExecuteCode")

    # Register tools with LargeLanguageModel class
    LargeLanguageModel.register_available_tools(tools)
    # ===== Node Definitions =====
    # Node: SolveProblem
    # Description: An agent that takes a problem description and generates Python code to solve it, executing the code to return the result.
    def solveproblem_function(state):
        """Solves the given Project Euler problem using Python code.
    
        Args:
            state (dict): The current state of the system, including the problem description.
    
        Returns:
            dict: The updated state of the system, including the generated code and the result.
        """
        llm = LargeLanguageModel(temperature=0.0)  # Set temperature to 0 for more deterministic results
        system_prompt = """You are an expert Python programmer tasked with solving Project Euler problems.
        You will be given a problem description, and your goal is to write Python code that solves the problem.
        You have access to a tool called `ExecuteCode` that allows you to execute Python code and get the output.
        Use this tool to execute the code and verify that it solves the problem correctly.
        Once you are satisfied with the solution, return the final answer.
    
        Here's how you should proceed:
        1.  Read the problem description carefully and make sure you understand it completely.
        2.  Write Python code that solves the problem.
        3.  Use the `ExecuteCode` tool to execute the code and get the output.
        4.  If the output is incorrect or an error occurs, analyze the code and fix any bugs.
        5.  Repeat steps 3 and 4 until the code produces the correct output.
        6.  Return the final answer.
    
        Example:
        Problem: Find the sum of the digits in the number 100!
        Code:
        ```python
        import math
        number = math.factorial(100)
        s = 0
        for digit in str(number):
            s += int(digit)
        print(s)
        ```
        """
    
        problem_description = state["problem_description"]
        messages = state.get("messages", [])
        full_messages = [SystemMessage(content=system_prompt), HumanMessage(content=f"Problem: {problem_description}")]
    
        llm.bind_tools(["ExecuteCode"])
        response = llm.invoke(full_messages)
        tool_messages, tool_results = execute_tool_calls(response)
    
        new_state = {"messages": messages + [response] + tool_messages, "result": str(tool_results)}
    
        return new_state

    graph.add_node("SolveProblem", solveproblem_function)

    # ===== Entry/Exit Configuration =====
    graph.set_entry_point("SolveProblem")
    graph.set_finish_point("SolveProblem")

    # ===== Compilation =====
    workflow = graph.compile()
    return workflow, tools
