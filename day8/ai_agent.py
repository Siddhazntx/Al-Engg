import os
import ast
import json
import math
import operator
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv
from groq import Groq

# Load environment variables (checks current folder .env first, then parent/working dir)
_current_dir = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_current_dir / ".env")
load_dotenv()

# Read API keys from environment
GROQ_API_KEY = os.getenv("groq_api_key") or os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("tavily_api_key") or os.getenv("TAVILY_API_KEY")

# Groq Model requested by the user
MODEL_NAME = "openai/gpt-oss-120b"

# Initialize Groq client if key is available
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# ============================================================================
# TOOL 1: Web Searching using Tavily API
# ============================================================================
def web_search(query: str) -> str:
    """
    Search the web using the Tavily API and return the searched answer and key results.
    
    Parameters:
        query (str): The search query to look up on the web.
        
    Returns:
        str: Summarized search answer and extracted content snippets.
    """
    api_key = TAVILY_API_KEY or os.getenv("tavily_api_key") or os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: Tavily API key is missing. Please set 'tavily_api_key' in your .env file."

    # Try using tavily-python client first; fallback to direct REST call via httpx
    try:
        try:
            from tavily import TavilyClient
            tavily = TavilyClient(api_key=api_key)
            response = tavily.search(query=query, search_depth="basic", include_answer=True)
        except ImportError:
            import httpx
            resp = httpx.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "basic",
                    "include_answer": True,
                },
                timeout=30.0,
            )
            if resp.status_code != 200:
                return f"Tavily API request failed with status code {resp.status_code}: {resp.text}"
            response = resp.json()

        # Format search results
        results_output = []
        if response.get("answer"):
            results_output.append(f"Direct Answer: {response['answer']}\n")

        search_results = response.get("results", [])
        if search_results:
            results_output.append("Top Web Results:")
            for idx, item in enumerate(search_results[:5], start=1):
                title = item.get("title", "No Title")
                content = item.get("content", "")
                url = item.get("url", "")
                results_output.append(f"{idx}. {title}\n   Snippet: {content}\n   Source: {url}")

        if not results_output:
            return "No relevant search results found."

        return "\n".join(results_output)

    except Exception as e:
        return f"Error executing web search: {str(e)}"


# ============================================================================
# TOOL 2: Basic Mathematics Operations
# ============================================================================
# Supported operators for safe AST evaluation
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Supported math functions and constants
SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "pi": math.pi,
    "e": math.e,
}


def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression string (e.g., '2*2', '10 + 5 / 2').
    
    Parameters:
        expression (str): The mathematical expression to evaluate.
        
    Returns:
        str: The calculated numerical result or an error message.
    """
    try:
        clean_expr = expression.strip()

        # Parse mathematical expression into an AST tree
        node = ast.parse(clean_expr, mode="eval")

        def _evaluate(sub_node: Any) -> Any:
            if isinstance(sub_node, ast.Expression):
                return _evaluate(sub_node.body)
            elif isinstance(sub_node, ast.Constant):  # Python 3.8+ numbers/constants
                if isinstance(sub_node.value, (int, float)):
                    return sub_node.value
                raise ValueError(f"Unsupported constant type: {type(sub_node.value).__name__}")
            elif isinstance(sub_node, ast.BinOp):
                left = _evaluate(sub_node.left)
                right = _evaluate(sub_node.right)
                op_type = type(sub_node.op)
                if op_type in SAFE_OPERATORS:
                    return SAFE_OPERATORS[op_type](left, right)
                raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
            elif isinstance(sub_node, ast.UnaryOp):
                operand = _evaluate(sub_node.operand)
                op_type = type(sub_node.op)
                if op_type in SAFE_OPERATORS:
                    return SAFE_OPERATORS[op_type](operand)
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            elif isinstance(sub_node, ast.Call):
                if isinstance(sub_node.func, ast.Name) and sub_node.func.id in SAFE_FUNCTIONS:
                    func = SAFE_FUNCTIONS[sub_node.func.id]
                    args = [_evaluate(arg) for arg in sub_node.args]
                    return func(*args)
                raise ValueError("Unsupported function call in expression.")
            elif isinstance(sub_node, ast.Name):
                if sub_node.id in SAFE_FUNCTIONS:
                    return SAFE_FUNCTIONS[sub_node.id]
                raise ValueError(f"Unknown variable name: {sub_node.id}")
            else:
                raise TypeError(f"Unsupported expression syntax: {type(sub_node).__name__}")

        result = _evaluate(node)
        return str(result)

    except ZeroDivisionError:
        return "Error: Division by zero is not allowed."
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


# ============================================================================
# TOOL SPECIFICATIONS (Standard OpenAI/Groq Tool Format)
# ============================================================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web using Tavily for current information, real-time facts, news, "
                "or questions requiring up-to-date knowledge beyond the model's training cutoff."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the web.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Perform basic and safe mathematical calculations from an expression string, "
                "for example '2*2', '15 + 27 / 3', '2**10', or 'sqrt(144)'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression string to evaluate (e.g. '2*2').",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

# Map of tool names to their corresponding Python functions
AVAILABLE_TOOLS: Dict[str, Any] = {
    "web_search": web_search,
    "calculate": calculate,
}


# ============================================================================
# AI AGENT FUNCTION
# ============================================================================
def run_agent(user_query: str, verbose: bool = True, max_iterations: int = 5) -> str:
    """
    Run the AI agent with Groq LLM and automatic tool calling.
    
    Parameters:
        user_query (str): The user input query/instruction.
        verbose (bool): Whether to print tool-calling reasoning and execution traces.
        max_iterations (int): Maximum turns of tool calling loop to prevent infinite loops.
        
    Returns:
        str: Final generated response from the agent.
    """
    global client
    if client is None:
        key = os.getenv("groq_api_key") or os.getenv("GROQ_API_KEY")
        if not key:
            return "Error: 'groq_api_key' not found. Please add it to your .env file."
        client = Groq(api_key=key)

    # Conversation state
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant equipped with specialized tools:\n"
                "1. 'web_search': Use this whenever you need to find up-to-date facts, news, or external information.\n"
                "2. 'calculate': Use this whenever you need to perform mathematical operations, calculations, or arithmetic.\n"
                "When a tool is needed, call it with the appropriate parameters. "
                "If no tool is necessary, provide a direct answer."
            ),
        },
        {"role": "user", "content": user_query},
    ]

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"[User Query]: {user_query}")
        print(f"[Using Model]: {MODEL_NAME}")
        print(f"{'=' * 60}")

    # Agent Loop
    for iteration in range(max_iterations):
        # Call Groq LLM with tools and tool_choice='auto'
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        response_message = response.choices[0].message

        # Check if the model decided to call any tools
        tool_calls = response_message.tool_calls
        if not tool_calls:
            # Model has finalized its answer
            if verbose:
                print("\n[Final Answer]:")
            return response_message.content or "No response generated."

        # Format assistant message with tool calls for history
        assistant_msg = {
            "role": "assistant",
            "content": response_message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        }
        messages.append(assistant_msg)

        # If tools were invoked, execute each tool call
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            raw_arguments = tool_call.function.arguments

            try:
                parsed_args = json.loads(raw_arguments)
            except Exception:
                parsed_args = {"query": raw_arguments}

            if verbose:
                print(f"\n[Tool Selected]: {function_name}")
                print(f"[Tool Arguments]: {parsed_args}")

            # Execute tool function
            if function_name in AVAILABLE_TOOLS:
                tool_func = AVAILABLE_TOOLS[function_name]
                try:
                    tool_output = tool_func(**parsed_args)
                except Exception as ex:
                    tool_output = f"Error calling function {function_name}: {str(ex)}"
            else:
                tool_output = f"Error: Tool '{function_name}' is not recognized."

            if verbose:
                print(f"[Tool Output]:\n{tool_output}\n")

            # Append tool result back to the conversation history
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": str(tool_output),
                }
            )

    return "Agent reached maximum tool iterations without finishing."


# ============================================================================
# MAIN ENTRYPOINT / TESTING
# ============================================================================
if __name__ == "__main__":
    # print("\n--- Testing AI Agent with Groq ---")

    # # Example 1: Basic calculation
    # math_test = "Calculate 25 * 4 + (150 / 3) and tell me the answer."
    # answer_math = run_agent(math_test)
    # print(answer_math)

    # # Example 2: Web searching
    # web_test = "What is the latest news about Groq AI chips?"
    # answer_web = run_agent(web_test)
    # print(answer_web)

    # Interactive prompt mode
    print("\n" + "=" * 60)
    user_input = input("Enter your custom query for the AI agent (or press Enter to exit): ").strip()
    if user_input:
        final_result = run_agent(user_input)
        print(final_result)
