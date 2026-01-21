# https://github.com/ollama/ollama #


import os
from dotenv import load_dotenv
# from langchain_ollama import Ollama
from langchain_ollama.llms import OllamaLLM
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


## langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

## Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant. Please respond to the question asked"),
    ("user", "Question:{question}")
])

## Streamlit Framework
st.title("Langchain Demo with LLAMA2")
input_text = st.text_input("What question you have in mind ?")

## Ollama llama2 model
llm = OllamaLLM(model="gemma:2b")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)
