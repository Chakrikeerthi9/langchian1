import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('OPENAI_API')

if API_KEY is None:
    print("API_KEY is not set")
else:
    print("API_KEY is set")