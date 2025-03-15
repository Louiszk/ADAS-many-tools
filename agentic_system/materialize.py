import inspect
import textwrap
import re
import os
import copy

def get_function_source(func, target_name=None):
    """Extract source code from a function."""
    try:
        if hasattr(func, '_source_code'):
            source = copy.deepcopy(func._source_code)
        else:
            source = inspect.getsource(func)

        lines = source.split('\n')
        func_def_line = -1
        
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                func_def_line = i
                break
                
        if func_def_line == -1:
            raise ValueError("Could not find function definition")
            
        lines = lines[func_def_line:]
    
        if target_name:
            match = re.search(r'def\s+([^\s(]+)', lines[0])
            if match:
                original_name = match.group(1)
                lines[0] = re.sub(r'\b' + re.escape(original_name) + r'\b', target_name, lines[0])
        
        source = '\n'.join(lines)
        source = textwrap.dedent(source)

        return source
    except Exception as e:
        print(repr(e))
        error_msg = repr(e).replace('"', '\\"')
        return f'def {target_name or "unknown_function"}(state):\n    return "Function could not be extracted: {error_msg}"'

def materialize_system(system, output_dir="systems"):
    """Generate Python code representation of the system."""
    nodes_count = len(system.nodes)
    tool_count = len(system.tools)
    
    code_lines = [
        f"# {system.system_name} System Configuration",
        f"# Total nodes: {nodes_count}",
        f"# Total tools: {tool_count}",
        "",
        "from langgraph.graph import StateGraph",
        "from langchain_core.tools import tool",
        "import os"
    ]
    
    if system.imports:
        for imp in system.imports:
            if imp not in code_lines:
                code_lines.append(imp)
    
    code_lines.extend([
        "",
        "",
        "def build_system():",
        "    # Define state attributes for the system",
        "    class AgentState(TypedDict):",
    ])
    
    for attr_name, attr_type in system.state_attributes.items():
        code_lines.append(f"        {attr_name}: {attr_type}")
    
    code_lines.extend([
        "",
        "    # Initialize graph with state",
        "    graph = StateGraph(AgentState)",
        "",
        "    # Tool definitions",
    ])
    
    # Tool definitions
    if system.tools:
        code_lines.append("    # ===== Tool Definitions =====")
        code_lines.append("    tools = {}")
    
        # prevent cross-contamination
        tool_implementations = {}
        
        for tool_name, description in system.tools.items():
            function_name = f"{tool_name.lower()}_function"
            
            func_source = ""
            if tool_name in system.tool_functions:
                func = system.tool_functions[tool_name]
                func_source = get_function_source(func, function_name)
            else:
                func_source = f"def {function_name}(input_str):\n    return f\"Processed {{input_str}} with {tool_name}\""
            
            tool_implementations[tool_name] = func_source
            
            indented_source = "\n".join(f"    {line}" for line in tool_implementations[tool_name].split('\n'))
            
            code_lines.extend([
                f"    # Tool: {tool_name}",
                f"    # Description: {description}",
                indented_source,
                "",
                f"    tools[\"{tool_name}\"] = tool(runnable={function_name}, name_or_callable=\"{tool_name}\")",
                ""
            ])

        code_lines.append("    # Register tools with LargeLanguageModel class")
        code_lines.append("    LargeLanguageModel.register_available_tools(tools)")
    
    # Node definitions
    if system.nodes:
        code_lines.append("    # ===== Node Definitions =====")
    
        node_implementations = {}
        
        for node_name, description in system.nodes.items():
            function_name = f"{node_name.lower()}_function"
            
            func_source = ""
            if node_name in system.node_functions:
                func = system.node_functions[node_name]
                func_source = get_function_source(func, function_name)
            else:
                func_source = f"def {function_name}(state):\n    return state"

            node_implementations[node_name] = func_source
            
            indented_source = "\n".join(f"    {line}" for line in node_implementations[node_name].split('\n'))
            
            code_lines.extend([
                f"    # Node: {node_name}",
                f"    # Description: {description}",
                indented_source,
                "",
                f"    graph.add_node(\"{node_name}\", {function_name})",
                ""
            ])
    
    # Standard edges
    if system.edges:
        code_lines.append("    # ===== Standard Edges =====")
        
        for source, target in system.edges:
            code_lines.extend([
                f"    graph.add_edge(\"{source}\", \"{target}\")",
                ""
            ])
    
    # Conditional edges
    if system.conditional_edges:
        code_lines.append("    # ===== Conditional Edges =====")
        
        condition_implementations = {}
        
        for source, edge_info in system.conditional_edges.items():
            function_name = f"{source.lower()}_router"
            
            func_source = ""
            if "condition" in edge_info:
                func = edge_info["condition"]
                func_source = get_function_source(func, function_name)
            else:
                func_source = f"def {function_name}(state):\n    return \"default_target\""

            condition_implementations[source] = func_source
            
            path_map_str = ""
            if "path_map" in edge_info:
                path_map = edge_info["path_map"]
                path_map_str = f", {repr(path_map)}"
            
            indented_source = "\n".join(f"    {line}" for line in condition_implementations[source].split('\n'))
            
            code_lines.extend([
                f"    # Conditional Router from: {source}",
                indented_source,
                "",
                f"    graph.add_conditional_edges(\"{source}\", {function_name}{path_map_str})",
                ""
            ])

    # Entry/Exit Configuration
    code_lines.extend([
        "    # ===== Entry/Exit Configuration =====",
        f"    graph.set_entry_point(\"{system.entry_point}\")",
        f"    graph.set_finish_point(\"{system.finish_point}\")",
        "",
        "    # ===== Compilation =====",
        "    workflow = graph.compile()",
        "    return workflow, tools" if system.tools else "    return workflow, {}",
        ""
    ])

    code = "\n".join(code_lines)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, system.system_name.replace("/", "_").replace("\\", "_").replace(":", "_") + ".py")
        with open(filename, "w") as f:
            f.write(code)
        
    return code