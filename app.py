import sys
import configparser
import os
import configparser


from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chains import ConversationalRetrievalChain
from qdrant_client import QdrantClient,models
from qdrant_client.http.models import PointStruct
from openai import OpenAI

from flask import Flask, request, abort
from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.exceptions import (
    InvalidSignatureError
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage
)

#Config Parser
config = configparser.ConfigParser()
config.read('config.ini')

app = Flask(__name__)

channel_access_token = config['Line']['CHANNEL_ACCESS_TOKEN']
channel_secret = config['Line']['CHANNEL_SECRET']
if channel_secret is None:
    print('Specify LINE_CHANNEL_SECRET as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    print('Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.')
    sys.exit(1)

handler = WebhookHandler(channel_secret)

configuration = Configuration(
    access_token=channel_access_token
)

os.environ["OPENAI_API_KEY"]=config['Rag']['OPENAI_API_KEY']
os.environ["QDRANT_URL"]=config['Rag']['QDRANT_URL']

#create new cluseter in qdrant

connection = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key="PtYX2su0b_Xof19YN54ybCUvIZgdA94HqDe0vPUHBQ8CNu7Moun0VQ",
)


connection.recreate_collection(
    collection_name="first_project",
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
)
print("Create collection reponse:", connection)

info = connection.get_collection(collection_name="first_project")

print("Collection info:", info)
for get_info in info:
  print(get_info)


def load_and_split_documents(filepath="baby.pdf"):
  loader = PyPDFLoader(filepath)
  documents = loader.load()
  text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
  return text_splitter.split_documents(documents)


def ask_question_with_context(qa, question, chat_history):

    query = ""
    result = qa({"question": question, "chat_history": chat_history})
    print("answer:", result["answer"])
    chat_history = [(query, result["answer"])]
    return chat_history


def get_embeddings():
  return OpenAIEmbeddings(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    chunk_size=1
  )

from langchain_openai import ChatOpenAI
def get_chat_model():
  return ChatOpenAI(temperature=0)


def get_document_store(docs, embeddings):
  upsert = Qdrant.from_documents(
    docs,
    embeddings,
    url=os.environ.get("QDRANT_URL"),
    collection_name="first_project",
    # force_recreate=True,
    api_key="PtYX2su0b_Xof19YN54ybCUvIZgdA94HqDe0vPUHBQ8CNu7Moun0VQ"
  )
  print(upsert)
  return upsert


def ask_question_with_context(qa, question, chat_history):

    query = ""
    result = qa({"question": question, "chat_history": chat_history})
    print("answer:", result["answer"])
    chat_history = [(query, result["answer"])]
    return chat_history


def main():

    embeddings = get_embeddings()
    docs = load_and_split_documents()
    
    doc_store = get_document_store(docs, embeddings)
    llm = get_chat_model()

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=doc_store.as_retriever(),
        return_source_documents=True,
        verbose=False
    )

    chat_history = []
    while True:
        query = input('you: ')
        if query == 'q':
            break
        chat_history = ask_question_with_context(qa, query, chat_history)


if __name__ == "__main__":
    app.run()