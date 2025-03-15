# SimpleEulerSolver System Configuration
# Total nodes: 1
# Total tools: 1

from langgraph.graph import StateGraph
from langchain_core.tools import tool
import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from typing import Dict, List, Any, Callable, Optional, Union, TypeVar, Generic, Tuple, Set, TypedDict
from agentic_system.large_language_model import LargeLanguageModel, execute_tool_calls


def build_system():
    # Define state attributes for the system
    class AgentState(TypedDict):
        messages: List[Any]
        input: str

    # Initialize graph with state
    graph = StateGraph(AgentState)

    # Tool definitions
    # ===== Tool Definitions =====
    tools = {}
    # Tool: ExecutePythonCode
    # Description: Executes given Python code and returns the output or an error message if execution fails.
    def executepythoncode_function(code: str) -> str:
        '''Executes the given Python code and returns the output.
    
        Args:
            code (str): The Python code to execute.
    
        Returns:
            str: The output of the executed code, or an error message if execution fails.
        '''
        try:
            import io, sys
            from contextlib import redirect_stdout
    
            # Redirect stdout to capture the output
            f = io.StringIO()
            with redirect_stdout(f):
                # Execute the code
                exec(code)
            output = f.getvalue()
            return output
        except Exception as e:
            return str(e)

    tools["ExecutePythonCode"] = tool(runnable=executepythoncode_function, name_or_callable="ExecutePythonCode")

    # Register tools with LargeLanguageModel class
    LargeLanguageModel.register_available_tools(tools)
    # ===== Node Definitions =====
    # Node: AgentNode
    # Description: An agent that solves Project Euler problems by generating and executing Python code.
    def agentnode_function(state):
        llm = LargeLanguageModel(temperature=0.0)
        system_prompt = """You are an expert Python programmer and mathematician. Your goal is to solve Project Euler problems. You will be given a problem description as input. You must generate Python code that solves the problem and then execute the code using the ExecutePythonCode tool. You should first briefly explain your approach, and then present only the final answer, without further explanation.
    
        Here's how you should operate:
        1. Understand the Problem: Carefully read and understand the problem statement.
        2. Plan your Solution: Break down the problem into smaller, manageable steps.
        3. Explain your Approach: Briefly describe your plan to solve the problem *before* generating any code.
        4. Generate Python Code: Write Python code to implement your solution. Ensure the code is correct, efficient, and well-formatted.
        5. Execute the Code: Use the ExecutePythonCode tool to run your code.
        6. Analyze the Results: Check the output of the code execution. If there are errors, debug your code and re-execute it.
        7. Present the Answer: Once you have the correct answer, present it clearly and concisely. Present ONLY the final numerical answer.
    
        Example:
        Problem: Find the sum of the digits in the number 100!
        Solution:
        First, I will calculate 100! using the math.factorial function. Then, I will convert the result to a string, iterate through the digits, convert them back to integers, and sum them up.
        ```python
        import math
        number = math.factorial(100)
        sum_digits = sum(int(digit) for digit in str(number))
        print(sum_digits)
        ```
        """
    
        llm = llm.bind_tools(["ExecutePythonCode"])
    
        messages = state.get("messages", [])
        input_problem = state.get("input", "")
        messages.append(HumanMessage(content=input_problem))
        full_messages = [SystemMessage(content=system_prompt)] + messages
        response = llm.invoke(full_messages)
        tool_messages, tool_results = execute_tool_calls(response)
    
        new_state = {"messages": messages + [response] + tool_messages}
    
        return new_state
    

    graph.add_node("AgentNode", agentnode_function)

    # ===== Entry/Exit Configuration =====
    graph.set_entry_point("AgentNode")
    graph.set_finish_point("AgentNode")

    # ===== Compilation =====
    workflow = graph.compile()
    return workflow, tools
