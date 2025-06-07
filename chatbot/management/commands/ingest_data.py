# chatbot/management/commands/ingest_data.py

from django.core.management.base import BaseCommand
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Define paths and model names from rag_logic.py for consistency
from chatbot.rag_logic import DATA_PATH, DB_CHROMA_PATH, EMBEDDING_MODEL_NAME

class Command(BaseCommand):
    help = 'Ingests data from the source file and creates a Chroma vector database.'

    def handle(self, *args, **kwargs):
        self.stdout.write("Starting data ingestion...")

        # 1. Load Documents
        self.stdout.write(f"Loading documents from {DATA_PATH}...")
        loader = TextLoader(DATA_PATH, encoding="utf-8")
        documents = loader.load()

        # 2. Split Documents into Chunks
        self.stdout.write("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        self.stdout.write(f"Split into {len(texts)} chunks.")

        # 3. Create Embeddings
        self.stdout.write(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )

        # 4. Create and Persist the Vector DB
        self.stdout.write(f"Creating and persisting Chroma DB to {DB_CHROMA_PATH}...")
        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=DB_CHROMA_PATH
        )
        vectordb.persist()
        
        self.stdout.write(self.style.SUCCESS('Successfully ingested data and created the vector database.'))