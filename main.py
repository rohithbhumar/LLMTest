import os
from openai_key import openai_key
from langchain_openai import OpenAI
# from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate

os.environ['OPENAI_API_KEY'] = openai_key
llm = OpenAI(temperature=0.6)
name = llm("I want one unique Indian food chain name that serves South Indian Cuisine")
print(name)

