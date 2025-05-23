import os
from llama_index.core import load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import StorageContext, Settings
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer
from services.llm import LLMInitializer


class LlamaIndexQueryOnly:
    def __init__(self, config_path="config.json", storage_dir="./storage"):
        # Initialize LLM
        llm_initializer = LLMInitializer(config_path)
        self.llm = llm_initializer.initialize_llm()
        self.embed_model = llm_initializer.initilize_embedding()
        
        # Set global LLM and embedding model
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        self.storage_dir = storage_dir
        self.index = None
        self.query_engine = None
        
    def load_existing_index(self):
        """Load existing index from storage"""
        if not os.path.exists(self.storage_dir):
            raise FileNotFoundError(f"Storage directory not found: {self.storage_dir}")
        
        # Load index from disk
        vector_store = FaissVectorStore.from_persist_dir(self.storage_dir)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, 
            persist_dir=self.storage_dir
        )
        self.index = load_index_from_storage(storage_context=storage_context)
        
        return self.index
    
    def create_query_engine(self):
        """Create query engine from loaded index"""
        if self.index is None:
            self.load_existing_index()
        
        # Create query engine with post-processor
        self.query_engine = self.index.as_query_engine(
            node_postprocessors=[
                SentenceEmbeddingOptimizer(percentile_cutoff=0.5)
            ], 
            llm=self.llm, 
            embed_model=self.embed_model
        )
        
        return self.query_engine
    
    def query(self, question):
        """Query the loaded index"""
        if self.query_engine is None:
            self.create_query_engine()
        
        response = self.query_engine.query(question)
        return response
    
    def initialize(self):
        """Initialize everything - load index and create query engine"""
        self.load_existing_index()
        self.create_query_engine()
        return self.query_engine


# Usage example
if __name__ == "__main__":
    # Initialize the query-only processor
    rag_query = LlamaIndexQueryOnly(
        config_path="config.json",
        storage_dir="./storage"
    )
    
    # Initialize (load index and create query engine)
    query_engine = rag_query.initialize()
    
    # Example query
    que="What is the main contribution of the Transformer architecture ? in detail"
    response = query_engine.query(que)
    print("\n\n",que)
    print(response)
