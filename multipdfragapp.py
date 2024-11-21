import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
# Retaining original imports; commenting out unused ones for now
# from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
# from langchain_anthropic import ChatAnthropic
# from langchain.agents import AgentExecutor, create_tool_calling_agent
import openai

load_dotenv()

# Azure API Configurations
# I used the same approach as the test file to ensure compatibility with Azure OpenAI
endpoint = os.getenv("AZURE_OPENAI_API_BASE")
key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

openai.api_type = "azure"
openai.api_base = endpoint
openai.api_version = api_version
openai.api_key = key


embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")

def get_conversational_chain(ques):

    try:
        # Using Azure OpenAI setup from the test file
        response = openai.ChatCompletion.create(
            engine=deployment_name,  # Use `engine` for Azure deployments
            messages=[
                {"role": "system", "content": """You are a helpful assistant. Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer"""},
                {"role": "user", "content": ques}
            ]
        )
        result = response["choices"][0]["message"]["content"]
        #print("Response:", result)
        st.write("Reply: ", result)
    except Exception as e:
        st.error(f"Error communicating with Azure GPT-4o: {e}")

def user_input(user_question):
    new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)

    retriever = new_db.as_retriever()
    relevant_chunks = retriever.get_relevant_documents(user_question)

    # Passing retrieved chunks to the Azure GPT-4o model
    context = "\n".join([doc.page_content for doc in relevant_chunks])
    query_with_context = f"Context: {context}\n\nQuestion: {user_question}"
    get_conversational_chain(query_with_context)

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("RAG based Chat with PDF")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        try:
            user_input(user_question)
        except Exception as e:
            st.error(f"Error: {e}")

    with st.sidebar:
        st.title("Menu:")
        pdf_doc = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                try:
                    raw_text = pdf_read(pdf_doc)
                    text_chunks = get_chunks(raw_text)
                    vector_store(text_chunks)
                    st.success("Processing complete! You can now ask questions.")
                except Exception as e:
                    st.error(f"Error during processing: {e}")

if __name__ == "__main__":
    main()
