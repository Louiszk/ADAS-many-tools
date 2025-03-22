# Imports
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Any, TypedDict
from agentic_system.large_language_model import LargeLanguageModel
import re
import sys
from io import StringIO


def build_system():
    # Define state attributes for the system
    class AgentState(TypedDict):
        messages: List[Any]
        problem: str
        solution: str

    # Initialize graph with state
    graph = StateGraph(AgentState)

    # ===== Node Definitions =====
    def agent_node(state):
        llm = LargeLanguageModel(temperature=0.1)
        system_prompt = """
            You will solve math problems with Python code.
            
            Provide exactly one markdown codeblock with a Python solution that prints only the final answer as float.
            The solution must be efficient and correct.
        """

        full_messages = [SystemMessage(content=system_prompt), HumanMessage(content=state["problem"])]
        response = llm.invoke(full_messages)
        
        response_text = response.content
        
        # Extract code from response
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", response_text, re.DOTALL)
        if code_blocks:
            code = code_blocks[0].strip()
        else:
            # If no code blocks, assume the entire response is code
            code = response_text.strip()
        
        # Execute the code to get the answer
        try:
            # Create a safe local environment
            local_env = {}
            
            # Capture stdout to get printed values
            original_stdout = sys.stdout
            captured_output = StringIO()
            sys.stdout = captured_output
            
            try:
                # Execute the code
                exec(code, {"__builtins__": __builtins__}, local_env)
                output = captured_output.getvalue()
            finally:
                sys.stdout = original_stdout
        
        except Exception as e:
            output = f"Error: {str(e)}"
        
        new_state = state.copy()
        new_state["solution"] = output
        
        return new_state

    graph.add_node("GSMHardExecAgent", agent_node)

    # ===== Entry/Exit Configuration =====
    graph.set_entry_point("GSMHardExecAgent")
    graph.set_finish_point("GSMHardExecAgent")

    # ===== Compilation =====
    workflow = graph.compile()
    return workflow, {}