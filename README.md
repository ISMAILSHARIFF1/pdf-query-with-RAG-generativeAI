# pdf-query-with-RAG-generativeAI
Create a Vector Database from PDF, provide question and LLM answers using RAG technique

There are two Python applications in the two folders:
1. PDF Index: this application uses streamlit to provide a  UI to upload a PDF document. This application creates a vector database from the text in the document and stores it in S3 in AWS. LLM is used to create the vector indexes.
2. PDF Query: this application uses streamlit to provide a UI to input a question as text. This application creates a Prompt and passes it to langchin with a LLM. lanchain orchestraits the uses of vector DB and intructs the LLM to provide a clear and consise answer.
