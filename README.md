# pdf-query-with-RAG-generativeAI
An advanced application designed to answer questions based on PDF documents, which are indexed within a vector database. Leveraging Large Language Models (LLMs) with the Retrieval-Augmented Generation (RAG) technique, it generates precise and contextually rich responses.

There are two Python applications in these folders:

PDF Index: This application, built with Streamlit, offers a UI for uploading PDF documents. It converts document text into vector indexes using an LLM from AWS Bedrock, stores them in an S3 bucket on AWS, and builds a vector database for efficient storage and querying.

PDF Query: Also using Streamlit, this application provides a UI for inputting questions. It generates prompts and utilizes LangChain to orchestrate interactions with the vector database. An LLM from AWS Bedrock then processes these prompts to deliver clear, concise answers.
