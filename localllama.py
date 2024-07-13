import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import os

# Load environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Streamlit app
def main():
    st.title("Chat with OLLAMA")
    
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
