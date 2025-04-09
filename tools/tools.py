from langchain_community.tools import BraveSearch
import json


def get_profile_url_from_brave_search(name: str):

    search = BraveSearch().from_search_kwargs(search_kwargs={"count": 4})
    result = search.run(name)

    return result
