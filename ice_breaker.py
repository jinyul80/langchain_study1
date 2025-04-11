from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

from third_parties.linkedin import scrape_linkedin_profile
from third_parties.twitter import scrape_user_tweets
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from agents.twitter_looup_agent import lookup as twitter_lookup_agent
from output_parsers import summary_parser, Summary


def ice_break_with(name: str) -> tuple[Summary, str]:

    linkedin_username = linkedin_lookup_agent(name)
    # Linkedin 접속 정보를 가져오지 못하므로 mock=True로 설정하여 임시적으로 정보를 가져옵니다.
    linkedin_data = scrape_linkedin_profile(linkedin_username, mock=True)

    # Twitter 정보를 가져옵니다.
    twitter_username = twitter_lookup_agent(name)
    tweets = scrape_user_tweets(twitter_username, mock=True)

    summary_template = """
        given the imformation about a person from linkedin {information},
        and their latest twitter posts {tweeter_posts} I want you to create:
        1. a short summary
        2. two interesting facts about them
        
        Use both information from twitter and linkedin
        
        모든 대답은 한글로 대답해 주세요.
        \n{format_instructions}
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information", "tweeter_posts"],
        template=summary_template,
        partial_variables={
            "format_instructions": summary_parser.get_format_instructions()
        },
    )

    llm = ChatGroq(temperature=0, model="qwen-2.5-32b")

    chain = summary_prompt_template | llm | summary_parser

    res: Summary = chain.invoke(
        input={"information": linkedin_data, "tweeter_posts": tweets}
    )

    return res, linkedin_data.get("profile_pic_url")


if __name__ == "__main__":
    print("hello Langchain!")

    ice_break_with(name="eden marco")
