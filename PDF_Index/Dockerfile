FROM python:3.11
EXPOSE 8083
WORKDIR /app
COPY requirements.txt ./
COPY . ./
RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit", "run", "indexer.py", "--server.port=8083", "--server.address=0.0.0.0"]