
import os
import faiss
from pathlib import Path
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
    
    def inspect_index(self, num_nodes=5):
        """Inspect the contents of the loaded index"""
        if self.index is None:
            self.load_existing_index()
        
        # Get all nodes from the index
        retriever = self.index.as_retriever(similarity_top_k=len(self.index.docstore.docs))
        all_nodes = list(self.index.docstore.docs.values())
        
        print(f"Total nodes in index: {len(all_nodes)}")
        print("=" * 80)
        
        # Show first few nodes
        for i, node in enumerate(all_nodes[:num_nodes]):
            print(f"\n--- NODE {i+1} ---")
            print(f"Node ID: {node.node_id}")
            print(f"Text length: {len(node.text)} characters")
            print(f"Text preview (first 300 chars):")
            print(f"'{node.text[:300]}...'")
            
            if hasattr(node, 'metadata') and node.metadata:
                print(f"Metadata: {node.metadata}")
            
            if hasattr(node, 'embedding') and node.embedding:
                print(f"Embedding dimension: {len(node.embedding)}")
            else:
                print("No embedding stored in node")
            
            print("-" * 50)
    
    def search_nodes(self, query, top_k=3):
        """Search and display retrieved nodes for a query"""
        if self.index is None:
            self.load_existing_index()
        
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        
        print(f"Query: '{query}'")
        print(f"Retrieved {len(nodes)} nodes:")
        print("=" * 80)
        
        for i, node in enumerate(nodes):
            print(f"\n--- RETRIEVED NODE {i+1} (Score: {node.score:.4f}) ---")
            print(f"Node ID: {node.node.node_id}")
            print(f"Text length: {len(node.node.text)} characters")
            print(f"Text content:")
            print(f"'{node.node.text}'")
            
            if hasattr(node.node, 'metadata') and node.node.metadata:
                print(f"Metadata: {node.node.metadata}")
            
            print("-" * 50)
        
        return nodes
    
    def get_node_by_id(self, node_id):
        """Get a specific node by its ID"""
        if self.index is None:
            self.load_existing_index()
        
        if node_id in self.index.docstore.docs:
            node = self.index.docstore.docs[node_id]
            print(f"--- NODE: {node_id} ---")
            print(f"Text: {node.text}")
            print(f"Metadata: {node.metadata if hasattr(node, 'metadata') else 'None'}")
            print(f"Text length: {len(node.text)} characters")
            return node
        else:
            print(f"Node with ID '{node_id}' not found")
            return None


# Usage example
if __name__ == "__main__":
    # Initialize the query-only processor
    rag_query = LlamaIndexQueryOnly(
        config_path="config.json",
        storage_dir="./storage"
    )
    
    # Initialize (load index and create query engine)
    query_engine = rag_query.initialize()
    
    # ========== INSPECT INDEX CONTENTS ==========
    print("INSPECTING INDEX CONTENTS:")
    rag_query.inspect_index(num_nodes=3)  # Show first 3 nodes
    
    # ========== SEARCH SPECIFIC NODES ==========
    print("\n\nSEARCHING FOR RELEVANT NODES:")
    retrieved_nodes = rag_query.search_nodes("transformer attention mechanism", top_k=2)
    
    # ========== GET SPECIFIC NODE BY ID ==========
    # First get all node IDs, then inspect one
    all_nodes = list(rag_query.index.docstore.docs.values())
    if all_nodes:
        first_node_id = all_nodes[0].node_id
        print(f"\n\nGETTING SPECIFIC NODE BY ID:")
        rag_query.get_node_by_id(first_node_id)
    
    # ========== REGULAR QUERYING ==========
    print("\n\nREGULAR QUERY RESULTS:")
    questions = [
        "What is the main contribution of the Transformer architecture? in detail",
        "What is attention mechanism?",
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 50)
        response = rag_query.query(question)
        print(f"Answer: {response}")
        print("=" * 80)