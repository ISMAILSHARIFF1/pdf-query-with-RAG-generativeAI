# pdf-query-with-RAG-generativeAI
An advanced application designed to answer questions based on PDF documents, which are indexed within a vector database. Leveraging Large Language Models (LLMs) with the Retrieval-Augmented Generation (RAG) technique, it generates precise and contextually rich responses.

There are two Python applications in the two folders:
1. PDF Index: this application uses streamlit to provide a  UI to upload a PDF document. This application creates a vector database from the text in the document and stores it in S3 in AWS. LLM is used to create the vector indexes.
2. PDF Query: this application uses streamlit to provide a UI to input a question as text. This application creates a Prompt and passes it to langchin with a LLM. lanchain orchestraits the uses of vector DB and intructs the LLM to provide a clear and consise answer.
