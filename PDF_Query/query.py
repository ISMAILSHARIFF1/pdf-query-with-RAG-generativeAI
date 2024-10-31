## Created by Ismail Shariff
## Imports
import boto3
from mlflow import langchain
from numpy import save
from ollama import embeddings
from sqlalchemy import true
import streamlit as st
import os
import uuid
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
#PDF Loader
from langchain_community.document_loaders import PyPDFLoader
##BEDROCK
from langchain_aws import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock


## AWS & Bedrock configuration
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")
bedrock_client = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)
folder_path="/temp/"

## download index
def download_index():
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}my_faiss.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{folder_path}my_faiss.pkl")


## load LLM for Querying using RAG technique
def get_llm():
    RAG_llm=Bedrock(model_id="anthropic.claude-v2:1", client=bedrock_client,
                    model_kwargs={'max_tokens_to_sample': 512}
                )
    return RAG_llm


## Generate Response using RAG
def generate_response(llm, vector_store, question):
    ## create prompt / template
    prompt_template = """
    Human: Hello AI, please use the given context to provide a clear and concise answer to the 
    question. In case you are not able to determine the answer, you could say that you don't know, 
    don't try to make up an answer.
    <context>
    {context}
    </context
    Question: {question}
    Assistant:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    response = qa({"query":question})
    return response['result']

## main function
def main():
    st.header("Chat with PDF, Query for meaningful answers.")
    download_index()
    dir_list = os.listdir(folder_path)
    st.write(f"Downloaded files are: {folder_path}")
    st.write(dir_list)
    
    ## INDEX creation
    faiss_index = FAISS.load_local(
        index_name="my_faiss",
        folder_path=folder_path,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True,
    )
    st.write("INDEX IS READY")

    question = st.text_input("Input Question")
    if st.button("Ask a Question"):
        with st.spinner("Processing Question..."):
            llm = get_llm()
            #generate response using RAG technique.
            st.write(generate_response(llm,faiss_index, question))
            st.success("Processing Complete!")
        
if __name__=="__main__":
    main()