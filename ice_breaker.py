import os
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers.string import StrOutputParser

from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent


def ice_break_with(name: str) -> str:
    linkedin_username = linkedin_lookup_agent(name)

    # Linkedin 접속 정보를 가져오지 못하므로 mock=True로 설정하여 임시적으로 정보를 가져옵니다.
    linkedin_data = scrape_linkedin_profile(linkedin_username, mock=True)

    summary_template = """
        given the Linkedin information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
        
        모든 대답은 한글로 대답해 주세요.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatGroq(temperature=0, model="qwen-2.5-32b")

    chain = summary_prompt_template | llm | StrOutputParser()

    res = chain.invoke(input={"information": linkedin_data})

    print(res)


if __name__ == "__main__":
    print("hello Langchain!")

    ice_break_with(name="eden marco")
