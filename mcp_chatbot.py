import asyncio
from typing import Annotated

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.messages import BaseMessage

from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langgraph.graph.message import add_messages

from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from langchain_mcp_adapters.client import MultiServerMCPClient

from typing_extensions import TypedDict

load_dotenv()

####################################################
# STATE
####################################################

class ChatState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


####################################################
# BUILD GRAPH
####################################################

async def build_graph():

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    ####################################################
    # MCP CLIENT
    ####################################################

    client = MultiServerMCPClient(

        {

            "math": {

                "command": "python",

                "args": [
                    "../local_math_server/server.py"
                ],

                "transport": "stdio"

            },

            "weather": {

                "url": "http://localhost:8000/sse",

                "transport": "sse"

            }

        }

    )

    ####################################################
    # LOAD TOOLS
    ####################################################

    tools = await client.get_tools()

    llm_with_tools = llm.bind_tools(tools)

    ####################################################
    # CHAT NODE
    ####################################################

    async def chatbot(state: ChatState):

        response = await llm_with_tools.ainvoke(
            state["messages"]
        )

        return {

            "messages": [response]

        }

    ####################################################
    # GRAPH
    ####################################################

    graph = StateGraph(ChatState)

    graph.add_node(
        "chatbot",
        chatbot
    )

    graph.add_node(
        "tools",
        ToolNode(tools)
    )

    graph.add_edge(
        START,
        "chatbot"
    )

    graph.add_conditional_edges(
        "chatbot",
        tools_condition
    )

    graph.add_edge(
        "tools",
        "chatbot"
    )

    graph.add_edge(
        "chatbot",
        END
    )

    return graph.compile()


####################################################
# MAIN
####################################################

async def main():

    chatbot = await build_graph()

    print("=" * 60)
    print("LangGraph MCP Client Started")
    print("=" * 60)

    while True:

        question = input("\nYou : ")

        if question.lower() in ["exit", "quit"]:

            break

        response = await chatbot.ainvoke(

            {

                "messages": question

            }

        )

        print()

        print("Assistant :")

        print(response["messages"][-1].content)


if __name__ == "__main__":

    asyncio.run(main())