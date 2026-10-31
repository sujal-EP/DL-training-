from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#creating my prompts
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",'You are helpful assistant.Please repond to the questions asked.'),
        ("user","Question:{question}")
    ]
)
#steamlit framework
st.title('Langchain demo with gemma2:9b')
input_text = st.text_input('Enter your question here')

#ollama model
llm = Ollama(model="gemma2:9b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))

