import os
import configparser


from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain

#Config Parser
config = configparser.ConfigParser()
config.read('config.ini')

os.environ["OPENAI_API_KEY"] = config['Rag']['OPENAI_API_KEY']
os.environ["NEO4J_URI"] = config['Neo4j']['NEO4J_URI']
os.environ["NEO4J_USERNAME"] = config['Neo4j']['NEO4J_USERNAME']
os.environ["NEO4J_PASSWORD"] = config['Neo4j']['NEO4J_PASSWORD']

class Neo4j:

    def __init__(self):
        db = Neo4jVector.from_existing_index(
            OpenAIEmbeddings(),
            url=os.environ.get("NEO4J_URI"),
            username=os.environ.get("NEO4J_USERNAME"),
            password=os.environ.get("NEO4J_PASSWORD"),
            index_name="health_report"
        )
        retriever = db.as_retriever()
        template = """你是護士，只能回答健康檢查的問題，其他問題一律不回答。並以繁體中文回答問題。
        SOURCES:
        {question}
        {summaries}
        """
        GERMAN_QA_PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])
        GERMAN_DOC_PROMPT = PromptTemplate(
            template="Inhalt: {page_content}\nQuelle: {source}",
            input_variables=["page_content", "source"])

        qa_chain = load_qa_with_sources_chain(ChatOpenAI(temperature=0), chain_type="stuff",
        prompt=GERMAN_QA_PROMPT,
        document_prompt=GERMAN_DOC_PROMPT)

        self.chain = RetrievalQAWithSourcesChain(combine_documents_chain=qa_chain, retriever=retriever,
        reduce_k_below_max_tokens=True, max_tokens_limit=2000,
        return_source_documents=True)

    chat_history = []

    def ask_question_with_context(self, question):
        global chat_history
    
        query = ""
        result = self.chain({"question": question}, return_only_outputs=True)

        print("answer:", result["answer"])
        chat_history = [(query, result["answer"])]
        return result["answer"]