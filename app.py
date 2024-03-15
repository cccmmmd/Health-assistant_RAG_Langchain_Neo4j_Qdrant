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

from flask import Flask, request, abort, render_template
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

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # parse webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

chat_history = []
@handler.add(MessageEvent, message=TextMessageContent)
def message_text(event):
    global chat_history
    query = event.message.text
    
    response = ask_question_with_context(query, chat_history)   

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=response)]
            )
        )

@app.route('/', methods=['GET','POST'])
def home():
   global chat_history
   response=''
   if request.form and 'question' in request.form:
        site = request.form.get('question')
        response = ask_question_with_context(site, chat_history)  
   return render_template('index.html', response=response)


os.environ["OPENAI_API_KEY"]=config['Rag']['OPENAI_API_KEY']
os.environ["QDRANT_URL"]=config['Rag']['QDRANT_URL']
os.environ["QDRANT_COLLECTION_NAME"]=config['Rag']['QDRANT_COLLECTION_NAME']

#create new cluseter in qdrant

connection = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key="PtYX2su0b_Xof19YN54ybCUvIZgdA94HqDe0vPUHBQ8CNu7Moun0VQ",
)


info = connection.get_collection(collection_name=os.environ.get("QDRANT_COLLECTION_NAME"))

print("Collection info:", info)
for get_info in info:
  print(get_info)


def get_embeddings():
  return OpenAIEmbeddings(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    chunk_size=1
  )

from langchain_openai import ChatOpenAI
def get_chat_model():
  return ChatOpenAI(temperature=0)



def ask_question_with_context(question, c_history):
    global qa, chat_history
    
    query = ""
    result = qa({"question": question, "chat_history": c_history})
    print("answer:", result["answer"])
    chat_history = [(query, result["answer"])]

    return result["answer"]

def main():
    global qa
    embeddings = get_embeddings()
    llm = get_chat_model()

    vector_store = Qdrant(
        client=connection,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embeddings=embeddings,
    )


    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        verbose=False
    )
    print(qa)
    # chat_history = []
    # while True:
    #     query = input('you: ')
    #     if query == 'q':
    #         break
    #     chat_history = ask_question_with_context(qa, query, chat_history)
main()

if __name__ == "__main__":
    app.run()