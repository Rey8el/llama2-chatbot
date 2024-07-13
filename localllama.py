import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Ensure the environment variable is set
api_key = os.getenv("LANGCHAIN_API_KEY")
if api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
else:
    st.error("LANGCHAIN_API_KEY environment variable is not set.")
    st.stop()

# Streamlit app
def main():
    st.title("Chat with LangChain")
    
    # Input from user
    user_input = st.text_input("Enter your question:")

    if st.button("Submit"):
        if user_input:
            # LangChain Prompt Template
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant. Please respond to the user queries"),
                    ("user", f"Question:{user_input}")
                ]
            )

            llm = Ollama(model="llama2")
            output_parser = StrOutputParser()
            chain = prompt | llm | output_parser

            response = chain.invoke({"question": user_input})
            st.write(f"Response: {response}")
        else:
            st.write("Please enter a question.")

if __name__ == "__main__":
    main()
