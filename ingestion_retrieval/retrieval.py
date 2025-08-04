import os
import re
import logging
import time
import asyncio
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import Runnable
from langchain_google_genai import GoogleGenerativeAI
from functools import lru_cache
import threading
from langchain.embeddings.base import Embeddings
import json
from typing import List
from pydantic import SecretStr
from config import get_config
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from sentence_transformers import SentenceTransformer
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


# ------------------------- 
# ✅ Input Sanitization
# -------------------------

# Async function using Groq to rewrite a query
async def rewrite_query_with_groq(query: str) -> str:
    start_time = time.time()
    try:
        # Initialize LLM
        groq_llm = ChatGroq(
            model_name="gemma2-9b-it",  # Or "gemma2-9b-it" if that's preferred
            temperature=0.2,
            max_tokens=200,
            max_retries=1,
        )
        rewrite_prompt_template = f"""
You are an assistant that rewrites user queries to make them clearer, more specific, and optimized for accurate retrieval from HR policy documents.

Instructions:

If the original query is already well-structured and includes necessary HR or policy-related context, return it without changes.

If the query is too short, vague, or informal, rewrite it by:

Adding relevant HR context (e.g., employee grade, department, HR term like “leave policy”, “attendance”, “travel reimbursement”).

Making it sound like a formal question that fits HR documentation style.

If Query is in another language, change it into English First
Do not add any formatting, explanation, or notes—return only the rewritten query as plain text.

User Query:
"{query}"

Rewritten Query:

"""


        # Call LLM with proper chat format
        task = asyncio.create_task(groq_llm.ainvoke(input=rewrite_prompt_template))

        try:
            response = await asyncio.wait_for(task,timeout=20.0)

            if response and hasattr(response, 'content'):
                cleaned = response.content.strip()

                # Strip markdown or quotes
                if cleaned.startswith("```"):
                    cleaned = cleaned.split('\n', 1)[-1].rsplit("```", 1)[0].strip()
                cleaned = cleaned.strip('\'"')

                if cleaned:
                    logger.info(f"Rewritten query: '{query}' → '{cleaned}'")
                    return cleaned

            return query

        except asyncio.TimeoutError:
            logger.warning("GROQ API timeout on query rewrite.")
            if not task.done():
                task.cancel()
            return query

        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
            raise

        except Exception as e:
            logger.warning(f"GROQ rewrite failed: {e}")
            if not task.done():
                task.cancel()
            logger.info(f"Query rewrite completed in {time.time() - start_time:.2f} seconds")
            return query

    except Exception as outer:
        logger.error(f"Rewrite logic failed: {outer}")
        return query
async def format_query(query: str) -> str:
    """
    Format the query by first rewriting it for better specificity,
    then prepending the required string for retrieval.
    """
    start_time = time.time()
    try:
        # First rewrite the query for better specificity
        rewritten_query = await rewrite_query_with_groq(query)
        logger.debug(f"Rewritten query: {rewritten_query}")
        # Then format it for the embedding model
        return "Represent this sentence for searching relevant passages: " + rewritten_query
    except Exception as e:
        logger.error(f"Error in format_query: {str(e)}")
        # Fallback to original query if rewriting fails
        logger.info(f"Query formatting completed in {time.time() - start_time:.2f} seconds")
        return "Represent this sentence for searching relevant passages: " + query 

def sanitize_input(user_input: str) -> str:
    start_time = time.time()
    user_input = re.sub(r'\b[\w.-]+?@\w+?\.\w+?\b', '[REDACTED_EMAIL]', user_input)
    user_input = re.sub(r'\b\d{10,13}\b', '[REDACTED_PHONE]', user_input)
    user_input = re.sub(r'[{}<>[\]`$]', '', user_input)
    result = user_input.strip()
    logger.debug(f"Input sanitization completed in {time.time() - start_time:.4f} seconds")
    return result

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
    start_time = time.time()
    if _llm_cache is not None:
        logger.debug(f"LLM loaded from cache in {time.time() - start_time:.2f} seconds")
        return _llm_cache
    logger.info("[LLM CACHE] Loading LLM due to cache miss or invalidation...")
    try:
        # Import config here to avoid circular imports
        from config import get_config
        
        config = get_config()
        llm_config = config.get_active_llm_config() 
        
        logging.info(f"Loading {llm_config['provider']} LLM...")
        
        if llm_config['provider'] == 'openrouter':
            _llm_cache = ChatOpenAI(
                model_name=llm_config['model'],
                openai_api_base=llm_config.get('api_base', "https://openrouter.ai/api/v1"),
                temperature=llm_config.get('temperature', 0.1),
                openai_api_key=llm_config['api_key'],
                top_p=llm_config.get('top_p', 0.9),
            )
        elif llm_config['provider'] == 'google':
            _llm_cache = GoogleGenerativeAI(
                model=llm_config['model'], 
                google_api_key=llm_config['api_key'],
                temperature=llm_config.get('temperature', 0.1)
            )
        elif llm_config['provider'] == 'groq':
            _llm_cache = ChatGroq(
                model=llm_config['model'],
                api_key=llm_config['api_key'],
                temperature=llm_config.get('temperature', 0.1)
            )
        elif llm_config['provider'] == 'openai':
            _llm_cache = ChatOpenAI(
                model_name=llm_config['model'],
                openai_api_key=llm_config['api_key'],
            )
        else:
            # Fallback to OpenRouter
            _llm_cache = ChatOpenAI(
                model_name=llm_config['model'],
                openai_api_base=llm_config.get('api_base', "https://openrouter.ai/api/v1"),
                temperature=llm_config.get('temperature', 0.1),
                openai_api_key=llm_config['api_key'],
            )
        
        return _llm_cache
    except Exception as e:
        logging.warning(f"LLM loading failed: {e}")
        # Final fallback to Google
        try:
            _llm_cache = GoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.1
            )
            return _llm_cache
        except Exception as e2:
            logging.error(f"Fallback LLM loading also failed: {e2}")
            raise RuntimeError("All LLM providers failed to load")

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

def initialize_embeddings_and_vector_store(qdrant_url: str, qdrant_api: str, collection_name: str, config) -> tuple:
    """Initialize and return embeddings and vector store components."""
    global _qdrant_client, _vector_store
    config = get_config()   

    # Initialize Qdrant client if not already done
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api)
    
    # Initialize SentenceTransformer embeddings
    logger.info("Initializing SentenceTransformer with BAAI/bge-small-en-v1.5 model")
    
    class SentenceTransformerEmbeddings(Embeddings):
        def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
            self.model = SentenceTransformer(model_name)
            
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            """Embed a list of documents using SentenceTransformer."""
            if not texts:
                return []
                
            # Convert input to list of strings if it's a single string
            if isinstance(texts, str):
                texts = [texts]
                
            try:
                # Generate embeddings in batches to handle large inputs
                batch_size = 32
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    embeddings = self.model.encode(
                        batch,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                    all_embeddings.extend(embeddings.tolist())
                
                return all_embeddings
                
            except Exception as e:
                logger.error(f"Error embedding documents: {str(e)}")
                raise
                
        def embed_query(self, text: str) -> List[float]:
            """
            Embed a query using SentenceTransformer with the required prefix.
            This is a synchronous method that runs the async version in an event loop.
            """
            import asyncio
            
            # If we're already in an event loop, use it
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, we need to run the coroutine in a new thread
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        return loop.run_until_complete(
                            asyncio.wrap_future(
                                pool.submit(
                                    asyncio.run_coroutine_threadsafe,
                                    self.aembed_query(text),
                                    loop
                                )
                            )
                        )
            except RuntimeError:
                # No event loop, create a new one
                pass
                
            # If no running loop, create a new one
            return asyncio.run(self.aembed_query(text))
            
        async def aembed_query(self, text: str) -> List[float]:
            """Async version of embed_query."""
            try:
                # Format the query with the required prefix
                formatted_query = await format_query(text)
                embedding = self.model.encode(
                    formatted_query,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                return embedding.tolist()
            except Exception as e:
                logger.error(f"Error in async embed query: {str(e)}")
                raise
            from config import get_config
        
    config = get_config()
    llm_config = config.get_active_llm_config() 
    # Initialize the embeddings
    if llm_config['provider'] == 'openai':
        embeddings = OpenAIEmbeddings(model_name="text-embedding-3-large",dimensions=768)
    else:
        embeddings = SentenceTransformerEmbeddings()
    
    # Initialize sparse embeddings
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    # Check if collection exists and create if needed
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
    
    # Initialize vector store
    _vector_store = QdrantVectorStore(
        client=_qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=vector_name,
        sparse_vector_name=sparse_vector_name,
    )
     
    return embeddings, sparse_embeddings, _qdrant_client, _vector_store

def get_hr_assistant_chain(qdrant_url: str, qdrant_api: str, collection_name: str) -> Runnable:
    start_time = time.time()
    global _qdrant_client, _vector_store, _chain_cache
    
    # Check if we need to invalidate cache due to connection changes
    if should_invalidate_cache(qdrant_url, qdrant_api, collection_name):
        invalidate_cache()
        
    # Initialize hybrid retriever
    logger.info("Initializing hybrid retriever...")
    # Get config
    from config import get_config
    config = get_config()

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
            
            # 1. Initialize Embeddings and Vector Store
            embed_start = time.time()
            if _vector_store is None:
                embeddings, sparse_embeddings, _qdrant_client, _vector_store = initialize_embeddings_and_vector_store(
                    qdrant_url, qdrant_api, collection_name, config
                )
            timings['init_embeddings'] = time.time() - embed_start

            # 3. Setup optimized retriever with HNSW parameters
            retriever_start = time.time()

            # Create a base retriever with optimized parameters
            base_retriever = _vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": config.vector_search.search_k, 
                    "lambda_mult": 0.4,   
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
            
            # 5. Create QA Chain with optimized prompt and chat history
            chain_start = time.time()
            
            def format_chat_history(chat_history: list) -> str:
                """Format chat history for inclusion in the prompt"""
                if not chat_history:
                    return "No previous conversation history."
                    
                formatted_history = []
                for msg in chat_history:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    formatted_history.append(f"{role}: {msg['content']}")
                
                return "\n".join(formatted_history)
            
            # Create a new prompt with chat history
            def get_prompt_with_history(chat_history: list = None):
                history_text = format_chat_history(chat_history or [])
                return ChatPromptTemplate.from_template(
                    hr_prompt.messages[0].prompt.template.replace(
                        "{chat_history}", 
                        history_text
                    )
                )
            
            # Create a chain that includes chat history
            def create_chain_with_history(chat_history: list = None):
                prompt = get_prompt_with_history(chat_history)
                qa_chain = create_stuff_documents_chain(llm, prompt)
                return create_retrieval_chain(mmr_retriever, qa_chain)
            
            # Return a function that can create a chain with specific chat history
            return create_chain_with_history

    except Exception as e:    
        logger.error(f"Error in get_hr_assistant_chain: {str(e)}", exc_info=True)
        raise