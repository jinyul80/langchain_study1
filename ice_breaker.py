import os
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers.string import StrOutputParser

from third_parties.linkedin import scrape_linkedin_profile


if __name__ == "__main__":
    print("hello Langchain!")

    summary_template = """
        given the Linkedin information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
        
        모든 대답은 한글로 대답해 주세요.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatGroq(temperature=0, model="qwen-2.5-coder-32b")

    chain = summary_prompt_template | llm | StrOutputParser()

    linkedin_data = scrape_linkedin_profile("", mock=True)

    res = chain.invoke(input={"information": linkedin_data})

    print(res)
