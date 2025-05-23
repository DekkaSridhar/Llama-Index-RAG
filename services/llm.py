import os
import json
import dotenv
# Load environment variables
dotenv.load_dotenv()
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding



class LLMInitializer:
    def __init__(self,config_path="config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        # Retrieve credentials from environment variables
        self.AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
        self.AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        self.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME= os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        # Initialize Azure OpenAI Chat Model in LangChain
        self.llm = None
        self.embed_model= None
        
    def initialize_llm(self):
        self.llm= AzureOpenAI(
            engine=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
            model=self.config.get('model', 'gpt-4o'),
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
            temperature=self.config.get('temperature', 0.3),
            max_tokens=self.config.get('max_tokens', 1000),
            max_retries=self.config.get('max_retries', 3),
        )
        return self.llm
    
    def initilize_embedding(self):
        self.embed_model = AzureOpenAIEmbedding(
            model=os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME'),
            deployment_name=os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME'),
            api_key=os.getenv('AZURE_OPENAI_EMBEDDING_API_KEY'),
            azure_endpoint=os.getenv('AZURE_OPENAI_EMBEDDING_ENDPOINT'),
            api_version=os.getenv('AZURE_OPENAI_EMBEDDING_API_VERSION')
        )
        return self.embed_model
        
        