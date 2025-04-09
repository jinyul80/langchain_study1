import os
import sys
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

current_directory = os.getcwd()
sys.path.append(current_directory)

from tools.tools import get_profile_url_from_brave_search

load_dotenv()


def lookup(name: str) -> str:

    llm = ChatGroq(temperature=0, model="qwen-2.5-32b")

    template = """given the full name {name_of_person} I want you to get it me a link to their Linkedin profile page.
                Your answer should contain only a URL"""

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=get_profile_url_from_brave_search,
            description="useful for when you need get the Linkedin Page URL",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )

    linked_profile_url = result["output"]

    return linked_profile_url


if __name__ == "__main__":

    linkedin_url = lookup(name="Eden Marco Udemy")
    print(f"{linkedin_url=}")
