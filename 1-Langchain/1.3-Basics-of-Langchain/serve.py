from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from langserve import add_routes
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

## Create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", '{text}')
])
                                                   

parser = StrOutputParser()

## Create chain
## chain defines a step-by-step flow where each component processes and passes data to the next.
chain = prompt_template | model | parser

## App definition
app = FastAPI(
    title = "Langchain Serve",
    version = "1.0",
    description = "A simple API server using Langchain runnable interfaces"
)

## Adding chain routes
add_routes(
    app,
    chain,
    path="/chain"
)

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,
                host = "localhost",
                port=8000
                )
    
    
## http://localhost:8000
## http://localhost:8000/docs

"""
{
  "input": {
    "language": "french",
    "text": "Hello"
  }, 
  "config": {},
  "kwargs": {
    "additionalProp1": {}
  }
}

"""
Input:
{
  "input": {
    "language": "French",
    "text": "Hello, My name is Aman."
  },
  "config": {},
  "kwargs": {
    "additionalProp1": {}
  }
}

Output:
{
  "output": "Bonjour, je m'appelle Aman. \n\n(Note: In French, it's more common to say \"je m'appelle\" instead of \"My name is\", to make it more natural and polite.)",
  "metadata": {
    "run_id": "4a771ae8-9303-4950-b053-9aa153d929d1",
    "feedback_tokens": []
  }
}


"""



"""