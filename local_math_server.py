from fastmcp import FastMCP

# Create MCP Server
mcp = FastMCP("Math Server")


# -----------------------------
# Addition Tool
# -----------------------------

@mcp.tool()
def add(a: float, b: float) -> float:
    """
    Add two numbers.
    """

    return a + b


# -----------------------------
# Subtraction Tool
# -----------------------------

@mcp.tool()
def subtract(a: float, b: float) -> float:
    """
    Subtract two numbers.
    """

    return a - b


# -----------------------------
# Multiplication Tool
# -----------------------------

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers.
    """

    return a * b


# -----------------------------
# Division Tool
# -----------------------------

@mcp.tool()
def divide(a: float, b: float) -> float:
    """
    Divide two numbers.
    """

    if b == 0:
        return "Division by zero is not allowed."

    return a / b


# -----------------------------
# Run Server
# -----------------------------

if __name__ == "__main__":
    mcp.run()