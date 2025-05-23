import os
import faiss
from pathlib import Path
d = 3072  # dimensions of text-embedding-3-large
faiss_index = faiss.IndexFlatL2(d)
from llama_index.core import load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)
from llama_index.extractors.entity import EntityExtractor
from llama_index.core.ingestion import IngestionPipeline
# from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer
from llama_index.readers.file import PDFReader
from llama_index.core.schema import MetadataMode
from services.llm import LLMInitializer


class LlamaIndexRag:
    def __init__(self, config_path="config.json"):
        # Initialize LLM
        llm_initializer = LLMInitializer(config_path)
        self.llm = llm_initializer.initialize_llm()
        self.embed_model = llm_initializer.initilize_embedding()
        
        # Set global LLM and embedding model
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
    
    def load_pdf_documents(self, pdf_path="input/Attention is all you need.pdf"):
        """Load PDF document from specified path"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Load PDF using PDFReader
        reader = PDFReader()
        pdf_data = reader.load_data(file=Path(pdf_path))
        
        # Convert to Document objects
        documents = [Document(text=doc.text, metadata=doc.metadata) for doc in pdf_data]
        
        return documents
    
    def create_processing_pipeline(self):
        """Create the document processing pipeline with transformations"""
        transformations = [
            MarkdownNodeParser(),
            SummaryExtractor(
                summaries=["prev", "self", "next"], 
                llm=self.llm, 
                num_workers=4
            ),
            QuestionsAnsweredExtractor(
                questions=3, 
                llm=self.llm, 
                metadata_mode=MetadataMode.EMBED, 
                num_workers=4
            ),
            # EntityExtractor(
            # prediction_threshold=0.5,
            # label_entities=False,  # include the entity label in the metadata (can be erroneous)
            # device="cpu",  # set to "cuda" if you have a GPU)
            ]
        
        pipeline = IngestionPipeline(transformations=transformations)
        return pipeline
    
    def process_documents(self, pdf_path="input/Attention is all you need.pdf"):
        """Complete document processing workflow"""
        # Load PDF documents
        documents = self.load_pdf_documents(pdf_path)
        
        # Create and run processing pipeline
        pipeline = self.create_processing_pipeline()
        nodes = pipeline.run(documents=documents)
        
        #---------------------------------------------------------------------------------------------
        # Create index from processed documents
        index = VectorStoreIndex.from_documents(documents)
        
        #---------------------------------------------------------------------------------------------
        # # Create vector store and storage context
        # vector_store = MilvusVectorStore(
        #     # uri="./milvus_demo.db",
        #     uri="./milvus_llamaindex.db",
        #     dim=1536,
        #     overwrite=True
        # )
        ## storage_context = StorageContext.from_defaults(vector_store=vector_store)
        #---------------------------------------------------------------------------------------------
        
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # Create index from processed documents
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        
        # save index to disk
        index.storage_context.persist()
        
        # load index from disk
        vector_store = FaissVectorStore.from_persist_dir("./storage")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir="./storage"
        )
        index = load_index_from_storage(storage_context=storage_context)
        #---------------------------------------------------------------------------------------------
        
        
        # Create query engine with post-processor
        query_engine = index.as_query_engine(
            node_postprocessors=[
                SentenceEmbeddingOptimizer(percentile_cutoff=0.5)
            ], llm=self.llm, embed_model=self.embed_model
        )
        return query_engine, nodes, index


# Usage example
if __name__ == "__main__":
    # Initialize the document processor
    rag = LlamaIndexRag(config_path="config.json")
    
    # Process the PDF and create query engine
    query_engine, processed_nodes, vector_index = rag.process_documents(
        pdf_path="input/Attention is all you need.pdf"
    )
    
    # Example query
    que="What is the main contribution of the Transformer architecture ? in detail"
    response = query_engine.query(que)
    print("\n\n",que)
    print(response)