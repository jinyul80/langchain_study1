from typing import Union
from dotenv import load_dotenv
from langchain.tools import tool, Tool
from langchain.tools.render import render_text_description
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad.log import format_log_to_str


load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a given text.

    Args:
        text (str): given text

    Returns:
        int: length of the text
    """

    print(f"get_text_length enter with {text=}")
    text = text.strip("\n").strip('"')

    return len(text)


def find_tool_by_name(tools: list[Tool], tool_name: str) -> Tool:
    """Find a tool by name in a list of tools."""

    for tool in tools:
        if tool.name == tool_name:
            return tool

    raise ValueError(f"Tool with name '{tool_name}' not found")


if __name__ == "__main__":

    print("Hello ReAct LangChain!")

    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    llm = ChatGroq(temperature=0, model="qwen-2.5-32b", stop=["\nObservation"])
    intermediate_steps = []

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    agent_step = ""

    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length in characters of the text DOG? ",
                "agent_scratchpad": intermediate_steps,
            }
        )
        print("*" * 20)
        print(agent_step)

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input

            observation = tool_to_use.func(str(tool_input))
            print(f"{observation=}")

            intermediate_steps.append((agent_step, str(observation)))

    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)
