import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")


llm_model = "gpt-4o-mini"

llm = ChatOpenAI(model=llm_model, temperature=0.0)

system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an AI assistant that helps generate article titles."
)

user_prompt = HumanMessagePromptTemplate.from_template(
    """
    You are tasked with creating a name for a article.
    The article is here for you to examine {article}

    The name should be based of the context of the article.
    Be creative, but make sure the names are clear, catchy,
    and relevant to the theme of the article.

    Only output the article name, no other explanation or text
    can be provided.
    """,
    input_variables=["article"]
)

first_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

chain_one = (
    {"article": lambda x:x["article"]}
    | first_prompt
    | llm
    | {"article_title": lambda x:x.content}
)

article_title_msg = chain_one.invoke({"article": "Anime"})
print(article_title_msg)

second_user_prompt = HumanMessagePromptTemplate.from_template(
    """
    You are tasked with crerating a description for the article>
    The article is here for you to examine:
    -----
    {article}
    -----
    Here is the article title '{article_title}'.

    Output the SEO friendly article description within 200 characters. Do not output
    anything other than the description.
    """,
    input_variables=["article", "article_title"]
)

second_prompt = ChatPromptTemplate.from_messages([system_prompt, second_user_prompt])

chain_two = (
    {
        "article": lambda x:x["article"],
        "article_title": lambda x:x["article_title"]
    }
    | second_prompt
    | llm
    | {"summary": lambda x:x.content}
)

article_description_msg = chain_two.invoke(
    {
        "article": "Anime",
        "article_title": article_title_msg["article_title"]
    }
)

print(article_description_msg)