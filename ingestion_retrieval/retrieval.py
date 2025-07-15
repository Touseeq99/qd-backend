import os
import re
import logging
import time
import traceback
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from functools import lru_cache
import threading
from langchain.embeddings.base import Embeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.runnables import RunnableLambda
import json
# Conditional import for Cohere to avoid version conflicts
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Cohere not available: {e}")
    COHERE_AVAILABLE = False
from typing import List
from pydantic import SecretStr

logger = logging.getLogger(__name__)

# Global variables for caching
_llm_cache = None
_qdrant_client = None
_vector_store = None
_chain_cache = None
_cache_lock = threading.Lock()

# Cache keys for invalidation
_current_qdrant_url = None
_current_qdrant_api = None
_current_collection_name = None

# Global memory for demo (in production, use per-user/session memory)
memory = ConversationBufferWindowMemory(k=2, return_messages=True)

# -------------------------
# ✅ Input Sanitization
# -------------------------
def sanitize_input(user_input: str) -> str:
    user_input = re.sub(r'\b[\w.-]+?@\w+?\.\w+?\b', '[REDACTED_EMAIL]', user_input)
    user_input = re.sub(r'\b\d{10,13}\b', '[REDACTED_PHONE]', user_input)
    user_input = re.sub(r'[{}<>[\]`$]', '', user_input)
    return user_input.strip()

# -------------------------
# ✅ HR Prompt Template (No Welcome Message)
# -------------------------
with open("ingestion_retrieval/hr_routing_prompt.json", "r", encoding="utf-8") as f:
    hr_routing_prompt = json.load(f)
    prompt = hr_routing_prompt["hr_routing_prompt"]
hr_prompt = ChatPromptTemplate.from_template(prompt)


# -------------------------
# ✅ LLM Loader with Safe Fallback
# -------------------------
def load_llm():
    global _llm_cache
    if _llm_cache is not None:
        return _llm_cache
    logger.info("[LLM CACHE] Loading LLM due to cache miss or invalidation...")
    try:
        # Import config here to avoid circular imports
        from config import get_config
        config = get_config()
        llm_config = config.get_active_llm_config() 
        
        logging.info(f"Loading {llm_config['provider']} LLM...")
        
        if llm_config['provider'] == 'groq':
            _llm_cache = ChatGroq(
                model=llm_config['model'],
                api_key=llm_config['api_key'],
                temperature=llm_config['temperature']
            )
        elif llm_config['provider'] == 'google':
            _llm_cache = GoogleGenerativeAI(
                model=llm_config['model'],
                google_api_key=llm_config['api_key'],
                temperature=llm_config['temperature']
            )
        else:
            # Fallback to Google
            _llm_cache = GoogleGenerativeAI(
                model="gemma-3-12b-it",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.1
            )
        
        return _llm_cache
    except Exception as e:
        logging.warning(f"LLM loading failed: {e}")
        # Final fallback
        _llm_cache = GoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1
        )
        return _llm_cache

# -------------------------
# Cache Invalidation
# -------------------------
def invalidate_cache():
    """Invalidate all caches when configuration changes"""
    global _llm_cache, _qdrant_client, _vector_store, _chain_cache
    global _current_qdrant_url, _current_qdrant_api, _current_collection_name
    
    logger.info("Invalidating all caches due to configuration change")
    _llm_cache = None
    _qdrant_client = None
    _vector_store = None
    _chain_cache = None
    _current_qdrant_url = None
    _current_qdrant_api = None
    _current_collection_name = None
    logger.info("[LLM CACHE] _llm_cache set to None; next request will reload LLM with latest config")

def should_invalidate_cache(qdrant_url: str, qdrant_api: str, collection_name: str) -> bool:
    """Check if cache should be invalidated due to connection parameter changes"""
    global _current_qdrant_url, _current_qdrant_api, _current_collection_name
    
    if (_current_qdrant_url != qdrant_url or 
        _current_qdrant_api != qdrant_api or 
        _current_collection_name != collection_name):
        
        _current_qdrant_url = qdrant_url
        _current_qdrant_api = qdrant_api
        _current_collection_name = collection_name 
        return True
    return False

# -------------------------
# HR Assistant Chain (MultiQuery + MMR + Sanitization)
# -------------------------

@lru_cache(maxsize=1)
def get_cached_hr_assistant_chain(qdrant_url: str, qdrant_api: str, collection_name: str) -> Runnable:
    return get_hr_assistant_chain(qdrant_url, qdrant_api, collection_name)

def get_hr_assistant_chain(qdrant_url: str, qdrant_api: str, collection_name: str) -> Runnable:
    global _qdrant_client, _vector_store, _chain_cache
    
    # Check if we need to invalidate cache due to connection changes
    if should_invalidate_cache(qdrant_url, qdrant_api, collection_name):
        invalidate_cache()
        
    # Initialize hybrid retriever
    logger.info("Initializing hybrid retriever...")
    
    # Get config
    from config import get_config
    config = get_config()
    
    # Dense embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.embedding.google_model,
        google_api_key=config.embedding.google_api_key
    )
    
    # Sparse embeddings
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    # Initialize Qdrant client
    _qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api)
    
    # Check if collection exists
    try:
        collection_info = _qdrant_client.get_collection(collection_name)
        has_named_vectors = collection_info.config.params.vectors is not None
        vector_name = "" if not has_named_vectors else "dense"
        sparse_vector_name = "sparse" if has_named_vectors else None
    except Exception as e:
        # If collection doesn't exist, create it with named vectors
        if "doesn't exist" in str(e):
            logger.info(f"Creating new collection: {collection_name}")
            _qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(size=768, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
            )
            vector_name = "dense"
            sparse_vector_name = "sparse"
        else:
            raise
    
    # Initialize vector store with appropriate vector name
    _vector_store = QdrantVectorStore(
        client=_qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=vector_name,
        sparse_vector_name=sparse_vector_name,
    )
    
    # Create retriever with hybrid search
    retriever = _vector_store.as_retriever(  
        search_type="mmr",
        search_kwargs={
            "k": 4,  # Number of documents to retrieve
            "lambda_mult": 0.5  # Balance between relevance and diversity
        },
    )
    
    # Return cached chain if available
    if _chain_cache:
        return _chain_cache
        
    try:
        with _cache_lock:
            # Check again after acquiring lock
            if _chain_cache:
                return _chain_cache
                
            timings = {}
            start = time.time()
            
            # Import config here to avoid circular imports
            from config import get_config
            config = get_config()
            
            # 1. Initialize Embeddings
            embed_start = time.time()
            embeddings = None
            # Prefer Google embeddings if available
            if hasattr(config.embedding, 'google_api_key') and config.embedding.google_api_key and \
               hasattr(config.embedding, 'google_model') and config.embedding.google_model:
                logger.info(f"Using Google embedding model: {config.embedding.google_model}")
                google_api_key = config.embedding.google_api_key
                if google_api_key is not None:
                    google_api_key = SecretStr(google_api_key)
                embeddings = GoogleGenerativeAIEmbeddings(
                    model=config.embedding.google_model,
                    google_api_key=google_api_key
                )
            # Fallback to Cohere if Google is not available
            elif COHERE_AVAILABLE and \
                hasattr(config.embedding, 'cohere_api_key') and config.embedding.cohere_api_key and \
                hasattr(config.embedding, 'cohere_model') and config.embedding.cohere_model:
                logger.info(f"Using Cohere embedding model: {config.embedding.cohere_model}")
                class CohereEmbeddings(Embeddings):
                    def __init__(self, api_key, model):
                        self.client = cohere.Client(api_key)
                        self.model = model
                    def embed_documents(self, texts: List[str]) -> List[List[float]]:
                        resp = self.client.embed(
                            texts=texts, 
                            model=self.model,
                            input_type='search_document'
                        )
                        return [e for e in resp.embeddings if isinstance(e, list)]
                    def embed_query(self, text: str) -> List[float]:
                        resp = self.client.embed(
                            texts=[text], 
                            model=self.model,
                            input_type='search_query'
                        )
                        for e in resp.embeddings:
                            if isinstance(e, list):
                                return e
                        raise ValueError('No valid embedding returned')
                embeddings = CohereEmbeddings(
                    api_key=config.embedding.cohere_api_key,
                    model=config.embedding.cohere_model
                )
            else:
                raise ValueError("No valid embedding provider configured!")
            timings['init_embeddings'] = time.time() - embed_start

            # 2. Initialize Vector Store with persistent client
            vector_start = time.time()
            if _qdrant_client is None:
                _qdrant_client = QdrantClient(
                    url=qdrant_url, 
                    api_key=qdrant_api, 
                )
            
            if _vector_store is None:
                _vector_store = QdrantVectorStore(
                    client=_qdrant_client,
                    collection_name=collection_name,
                    embedding=embeddings
                )
            
            timings['init_vectorstore'] = time.time() - vector_start

            # 3. Setup optimized retriever with HNSW parameters
            retriever_start = time.time()

            # Create a base retriever with optimized parameters
            base_retriever = _vector_store.as_retriever(
                search_type="mmr",  # Faster than MMR
                search_kwargs={
                    "k": min(config.vector_search.search_k, 4),  # Increased for better coverage
                }
            )
            
            # Use the base retriever as mmr_retriever for compatibility
            mmr_retriever = base_retriever
            timings['setup_retriever'] = time.time() - retriever_start
            
            # 4. Load LLM with caching and warm-up
            llm_start = time.time()
            global _llm_cache
            if _llm_cache is None:
                _llm_cache = load_llm()
            llm = _llm_cache
            
            # Warm the LLM on startup for faster first-token output
            try:
                _ = llm.invoke(" ")  # Warm first-token latency
                logger.info("LLM warm-up completed successfully")
            except Exception as e:
                logger.warning(f"LLM warm-up failed: {e}", exc_info=True)
            
            timings['load_llm'] = time.time() - llm_start
            
            # 5. Create QA Chain with optimized prompt
            chain_start = time.time()
            qa_chain = create_stuff_documents_chain(llm, hr_prompt)
            retrieval_chain = create_retrieval_chain(mmr_retriever, qa_chain)
            # Optimized chain with memory
            def chain_with_memory(inputs):
                # Load last 2 questions as chat history
                chat_history = memory.load_memory_variables({})["history"]
                chain_inputs = dict(inputs)
                chain_inputs["chat_history"] = chat_history
                
                output = retrieval_chain.invoke(chain_inputs)
                # Save the current question and output to memory
                memory.save_context({"input": inputs["input"]}, {"output": output.get("answer", "")})
                return output
            
            # Return as a RunnableLambda to match expected type
            chain_with_memory_runnable = RunnableLambda(chain_with_memory)
            _chain_cache = chain_with_memory_runnable
            return chain_with_memory_runnable

    except Exception as e:
        logger.error(f"Error in get_hr_assistant_chain: {str(e)}", exc_info=True)
        raise