## Created by Ismail Shariff
## imports
import boto3
import streamlit as st
import os
import uuid
from mlflow import langchain
from numpy import save
from sqlalchemy import true
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
#PDF Loader
from langchain_community.document_loaders import PyPDFLoader
##BEDROCK
from langchain_aws import BedrockEmbeddings

## AWS & Bedrock configuration
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")
bedrock_client = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)


## split the pages into chunks.
def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(pages)
    return chunks

## creation of vector store
def create_vector_store(unique_id, documents):
    vector_store_faiss=FAISS.from_documents(documents, bedrock_embeddings)
    file_name=f"{unique_id}.bin"
    folder_path="/temp/"
    vector_store_faiss.save_local(index_name=file_name, folder_path=folder_path)

    ## upload the files to s3
    s3_client.upload_file(Filename=folder_path+"/"+file_name+".faiss", Bucket=BUCKET_NAME, Key="my_faiss.faiss")
    s3_client.upload_file(Filename=folder_path+"/"+file_name+".pkl", Bucket=BUCKET_NAME, Key="my_faiss.pkl")
    return True

## main function
def main():
    st.write("Welcome to PDF Indexer")
    upload_file = st.file_uploader("Upload or drop a file here", "pdf")
    if upload_file is not None:
        unique_id = uuid.uuid4()
        st.write(f"Unique Id: {unique_id}")
        saved_file_name = f"{unique_id}.pdf"
        with open(saved_file_name, mode="wb") as w:
            w.write(upload_file.getvalue())

        loader = PyPDFLoader(saved_file_name)
        pages = loader.load_and_split()
        st.write(f"Read {len(pages)} Pages")

        ## split Text from the PDF
        splitted_docs = split_text(pages, 1000, 200)
        st.write(f"Splitted doc into the following {len(splitted_docs)} length.")
        
        ## Create Vector store
        st.write("Initiating vector Store Creation")
        result = create_vector_store(unique_id, splitted_docs)

        if result:
            st.write("Indexing is Complte")
        else:
            st.write("Could create vector store, please check logs")

if __name__=="__main__":
    main()