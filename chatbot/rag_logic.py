# chatbot/rag_logic.py

import os
from django.conf import settings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Define paths and model names
DATA_PATH = "data/company_info.txt"
DB_CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Define the prompt template
PROMPT_TEMPLATE = """
Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Helpful Answer:
"""

class ChatbotPipeline:
    def __init__(self):
        self.qa_chain = self._initialize_qa_chain()

    def _initialize_qa_chain(self):
        # 1. Initialize the LLM (Google Gemini)
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.2,
            convert_system_message_to_human=True
        )

        # 2. Initialize Embeddings (Hugging Face)
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'} # Use CPU for embeddings
        )

        # 3. Load or create the Vector DB (Chroma)
        if os.path.exists(DB_CHROMA_PATH):
            vectordb = Chroma(
                persist_directory=DB_CHROMA_PATH,
                embedding_function=embeddings
            )
            print("Loaded Chroma DB from disk.")
        else:
            print("Chroma DB not found. Please run the ingestion script.")
            # We return None here. The view should handle this case.
            # In a real app, you might want to raise an exception.
            return None

        # 4. Create the retriever
        retriever = vectordb.as_retriever(search_kwargs={"k": 2})

        # 5. Create the QA chain
        custom_prompt = PromptTemplate(
            template=PROMPT_TEMPLATE, 
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": custom_prompt}
        )
        
        return qa_chain

    def query(self, user_question):
        if not self.qa_chain:
            return {"error": "The chatbot is not initialized. Please run the data ingestion script."}
            
        result = self.qa_chain({"query": user_question})
        return {"answer": result["result"]}

# A single instance to be used by the view
# This prevents re-initializing the pipeline on every request
chatbot_pipeline = ChatbotPipeline()