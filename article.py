import os
from threading import ThreadError
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
from skimage import io
import matplotlib.pyplot as plt
import openai

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

second_user_prompt = HumanMessagePromptTemplate.from_template(
    """
    You are tasked with creating a description for the article>
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


third_user_prompt = HumanMessagePromptTemplate.from_template(
    """You are tasked with creating a new paragraph for the
    article. The article is here for you to examine:

    ---

    {article}

    ---

    Choose one paragraph to review and edit. During your edit,
    ensure you provide constructive feedback to the user so they
    can learn where to improve their own writing.""",
    input_variables=["article"]
)     

third_prompt = ChatPromptTemplate.from_messages(
    [
        system_prompt,
        third_user_prompt
    ]
)

class Paragraph(BaseModel):
    original_paragraph: str = Field(description="The original paragraph")
    edited_paragraph: str = Field(description="The improved edited paragraph")
    feedback: str = Field(description=(
        "Constructive feedback on the original paragraph"
    ))

structured_llm = llm.with_structured_output(Paragraph)

chain_three = (
    {"article": lambda x: x["article"]}
    | third_prompt
    | structured_llm
    | {
        "original_paragraph": lambda x: x.original_paragraph,
        "edited_paragraph": lambda x: x.edited_paragraph,
        "feedback": lambda x: x.feedback
    }
)

out = chain_three.invoke({"article": "Anime"})


image_prompt = PromptTemplate(
    input_variables=["article"],
    template=(
        "Generate a prompt with less then 500 characters to generate an image"
        "based on the following article: {article}"
    )
)

def generate_and_display_image(image_prompt):
    prompt = image_prompt.content 

    response = openai.OpenAI().images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024"
    )

    image_url = response.data[0].url
    image_data = io.imread(image_url)

    plt.imshow(image_data)
    plt.axis("off")
    plt.show()


image_gen_runnable = RunnableLambda(generate_and_display_image)

chain_four = (
    {"article": lambda x: x["article"]}
    | image_prompt
    | llm
    | image_gen_runnable
)

chain_four.invoke({"article": "Anime"})