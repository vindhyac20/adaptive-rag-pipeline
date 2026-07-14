import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

response = llm.invoke("In one sentence, what is machine learning?")
print(response.content)