# Configure NLTK before any imports that might use it
import os
os.environ["NLTK_DATA"] = os.getenv("NLTK_DATA", "/app/nltk_data")

import time
import logging
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import re
import shutil
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()
import traceback
from pathlib import Path
import sys
from functools import wraps

# Configure logging
def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters - removed emoji for Windows compatibility
    console_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(console_format))
    
    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_dir / 'hr_assistant.log', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(file_format))
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logging
logger = setup_logging()

# Configure NLTK to use a writable directory for data
import nltk
import os
from pathlib import Path

# Create a directory for NLTK data in the current working directory
nltk_data_dir = Path("nltk_data")
nltk_data_dir.mkdir(exist_ok=True, parents=True)

# Set NLTK data path to our custom directory
os.environ["NLTK_DATA"] = str(nltk_data_dir.absolute())

# Download required NLTK data
required_nltk_data = [
    'punkt_tab',
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng',  # Specifically required by unstructured
    'wordnet',
    'stopwords',
    'words',
    'maxent_ne_chunker',
    'book'  # Includes many useful corpora and models
]

try:
    for package in required_nltk_data:
        try:
            nltk.download(package, download_dir=str(nltk_data_dir.absolute()))
            logger.info(f"Successfully downloaded NLTK data: {package}")
        except Exception as e:
            logger.warning(f"Could not download NLTK package {package}: {str(e)}")
    
    # Verify NLTK data path is set correctly
    nltk.data.path.insert(0, str(nltk_data_dir.absolute()))
    logger.info(f"NLTK data path set to: {nltk_data_dir.absolute()}")
    
    # Test NLTK data loading
    try:
        nltk.data.find('tokenizers/punkt', paths=[str(nltk_data_dir.absolute())])
        logger.info("NLTK data verification successful")
    except LookupError:
        logger.warning("NLTK data verification failed - some functionality may be limited")
        
except Exception as e:
    logger.error(f"Error in NLTK data setup: {str(e)}")
    # Continue anyway, NLTK will try to download data when needed

# Import configuration and admin interface
from config import get_config, validate_config
from admin_interface import router as admin_router

# Import other modules after setting up logging
from ingestion_retrieval.ingestion import (
    ingest_documents_to_qdrant_async,
    check_or_create_qdrant_collection,
    delete_qdrant_collection,
    get_collections_with_chunk_counts
)
from ingestion_retrieval.retrieval import get_cached_hr_assistant_chain
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Initialize FastAPI with lifespan management
from contextlib import asynccontextmanager
from fastapi import FastAPI
import asyncio
import signal
import sys

class ServerState:
    def __init__(self):
        self.should_exit = False
        self.force_exit = False
        self.uvicorn_server = None

server_state = ServerState()

async def shutdown():
    """Graceful shutdown handler"""
    if server_state.should_exit:
        return
        
    server_state.should_exit = True
    logger.info("Initiating graceful shutdown...")
    
    if server_state.uvicorn_server:
        server_state.uvicorn_server.keep_running = False
        await server_state.uvicorn_server.shutdown()
    
    logger.info("Shutdown complete")

def handle_signal(signum, frame):
    """Handle OS signals for graceful shutdown"""
    if server_state.should_exit:
        logger.warning("Force shutdown requested...")
        server_state.force_exit = True
        return
        
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    asyncio.create_task(shutdown())

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Register signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_signal)
    
    # Startup
    logger.info("Starting up HR Assistant API...")
    
    try:
        # Store server reference for shutdown
        if hasattr(app, 'state') and hasattr(app.state, 'server'):
            server_state.uvicorn_server = app.state.server
        
        # Validate configuration
        validate_config()
        logger.info("Configuration validated successfully")
        
        # Pre-warm the model in the background
        async def prewarm_model():
            try:
                logger.info("Pre-warming the model...")
                qa_chain = get_cached_hr_assistant_chain(
                    config.database.url, 
                    config.database.api_key, 
                    config.database.collection_name
                )
                logger.info("Model pre-warm complete")
            except Exception as e:
                logger.error(f"Model pre-warm failed: {str(e)}", exc_info=True)
        
        # Start pre-warming in the background
        asyncio.create_task(prewarm_model())
        
        logger.info("Application startup complete")
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}", exc_info=True)
        server_state.should_exit = True
        raise
        
    finally:
        # Cleanup resources on shutdown
        if not server_state.force_exit:
            await shutdown()

# Initialize FastAPI with lifespan manager
app = FastAPI(
    title="HR Assistant API",
    version="1.0.0",
    lifespan=lifespan
)

# Get configuration
config = get_config()
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.security.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,  # Cache preflight requests for 24 hours
)

# Include admin router
app.include_router(admin_router)
 
# Timing decorator
def timeit(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time() 
        result = await func(*args, **kwargs)
        end = time.time()
        logger.info(f"⏱️ {func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

# In-memory storage for sessions and message history
sessions: Dict[str, List[Dict]] = {}

def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Get existing session ID or create a new one if none provided"""
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = []
    return session_id

class Message(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The HR-related question to answer"
    )
    session_id: Optional[str] = Field(
        None,
        description="Session ID to maintain conversation history. If not provided, a new session will be created."
    )
class IngestRequest(BaseModel):
    directory_path: str = Field( 
        "",
        min_length=0,
        max_length=500,
        description="Path to the directory containing documents to ingest (empty for default)"
    )
# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting HR Assistant...")
    try:
        # Validate configuration
        validate_config()
        logger.info("Configuration validated successfully")
        
        # Pre-warming the model with optimized chain
        logger.info("Pre-warming the model with optimized RAG pipeline...")
        qa_chain = get_cached_hr_assistant_chain(
            config.database.url, 
            config.database.api_key, 
            config.database.collection_name
        )
        
        # Skip pipeline testing during startup to avoid timeout issues
        logger.info("Skipping pipeline test during startup (will be tested on first query)")
        
        logger.info("HR Assistant started successfully")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}", exc_info=True)
        raise

# ========================
# ✅ Upload Folder
# ========================
@app.post("/upload")
async def upload_folder(
    folder_name: str = Form(None, min_length=1, max_length=100, regex=r'^[a-zA-Z0-9_-]+$'),
    files: List[UploadFile] = File(..., description="List of files to upload (subfolders supported)")
):
    """
    Upload multiple files or an entire folder (with subfolders and their files) to a specified folder. If no folder_name is provided, defaults to 'Data'.
    The folder structure is preserved under the upload directory.
    - **folder_name**: Name of the folder to store files (alphanumeric, underscores, hyphens only)
    - **files**: List of files to upload (subfolders supported)
    """
    if not folder_name:
        folder_name = "Data"
    logger.info(f"Upload request received - folder: {folder_name}, file count: {len(files) if files else 0}")
    try:
        # Validate folder name
        logger.debug(f"Validating folder name: {folder_name}")
        if not folder_name or not re.match(r'^[a-zA-Z0-9_-]+$', folder_name):
            error_msg = f"Invalid folder name: {folder_name}. Use only letters, numbers, underscores, and hyphens."
            logger.warning(error_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
        upload_dir = os.path.join(config.file_upload.upload_dir, folder_name)
        logger.debug(f"Creating upload directory: {upload_dir}")
        os.makedirs(upload_dir, exist_ok=True)
        if not files:
            error_msg = "No files were provided in the upload request"
            logger.warning(error_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
        saved_count = 0
        for file in files:
            logger.debug(f"Processing file: {file.filename} (content-type: {file.content_type})")
            fname = file.filename or ''
            # Validate file type using configuration
            allowed_extensions = set(config.file_upload.allowed_extensions)
            file_ext = os.path.splitext(fname)[1].lower()
            logger.debug(f"File extension: {file_ext}")
            if not file_ext or file_ext not in allowed_extensions:
                error_msg = f"File type {file_ext or 'unknown'} not allowed. Allowed types: {', '.join(sorted(allowed_extensions))}"
                logger.warning(f"{error_msg} - File: {file.filename}")
                continue
            # Validate file size using configuration
            max_size = config.file_upload.max_file_size_mb * 1024 * 1024
            file.file.seek(0, 2)  # Go to end of file
            file_size = file.file.tell()
            file.file.seek(0)  # Reset file pointer
            logger.debug(f"File size: {file_size} bytes")
            if file_size > max_size:
                logger.warning(f"File {file.filename} is too large. Size: {file_size} bytes, Max size: {max_size} bytes")
                continue
            # Save file - preserve folder structure
            safe_path = os.path.normpath(fname).replace('..', '')
            file_path = os.path.join(upload_dir, safe_path)
            logger.debug(f"Saving file to: {file_path}")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_count += 1
        if saved_count == 0:
            return {"message": "No valid files were uploaded. Only allowed file types and sizes are accepted."}
        logger.info(f"Successfully uploaded {saved_count} files to {folder_name}")
        return {"message": f"Successfully uploaded {saved_count} files to {folder_name}"}
    except HTTPException as he:
        logger.warning(f"HTTPException in upload_folder: {str(he.detail)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in upload_folder: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request"
        )


# ========================
# ✅ Ingest Endpoint
# ========================

@app.post("/ingest")
async def ingest_api(req: IngestRequest):
    # If no directory_path is provided or it's empty, default to 'uploaded_folders/Data'
    directory_path = req.directory_path if req.directory_path else os.path.join(config.file_upload.upload_dir, "Data")
    logger.info(f"Ingest request received - path: {directory_path}")
    
    try:
        abs_path = os.path.abspath(directory_path)
        root_path = os.path.abspath(config.file_upload.upload_dir)
        logger.debug(f"Absolute path: {abs_path}, Root path: {root_path}") 
        
        # Validate the path is within the uploaded_folders directory
        if not abs_path.startswith(root_path):
            error_msg = f"Path must be within the {config.file_upload.upload_dir} directory: {abs_path}"
            logger.warning(error_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
            
        # Check if path exists
        if not os.path.exists(abs_path):
            error_msg = f"Path does not exist: {abs_path}"
            logger.warning(error_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
            
        # Handle both file and directory paths
        if os.path.isfile(abs_path):
            logger.info(f"Processing single file: {abs_path}")
            # Create a temporary directory to store the file
            import tempfile
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy the file to the temp directory
                temp_file = os.path.join(temp_dir, os.path.basename(abs_path))
                shutil.copy2(abs_path, temp_file)
                
                # Process the file
                result = await _process_ingestion(temp_dir)
                
                # Remove the original file after successful processing
                os.remove(abs_path)
                
            return result 
             
        elif os.path.isdir(abs_path):
            logger.info(f"Processing directory: {abs_path}")
            # Process the directory directly
            result = await _process_ingestion(abs_path)
            
            # Remove only the files, keep the directory structure
            _remove_files_only(abs_path)
            logger.info(f"Successfully removed files from directory: {abs_path}")
            
            return result
            
        else:
            error_msg = f"Path is neither a file nor a directory: {abs_path}"
            logger.warning(error_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
            
    except HTTPException as he:
        logger.warning(f"HTTPException in ingest_api: {str(he.detail)}")
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error in ingest_api: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your request"
        )
 
def _remove_files_only(directory_path: str):
    """Remove only files from a directory, keeping the directory structure."""
    try:
        for root, dirs, files in os.walk(directory_path, topdown=False):
            # Remove files first
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    logger.debug(f"Removed file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove file {file_path}: {e}")
            
            # Remove empty directories (but keep the root directory)
            if root != directory_path:
                try:
                    if not os.listdir(root):  # Only remove if empty
                        os.rmdir(root)
                        logger.debug(f"Removed empty directory: {root}")
                except Exception as e:
                    logger.warning(f"Failed to remove directory {root}: {e}")
                    
    except Exception as e:
        logger.error(f"Error removing files from {directory_path}: {e}")

async def _process_ingestion(directory_path: str):
    """Helper function to process ingestion from a directory."""
    try:
        logger.info(f"Starting document ingestion from: {directory_path}")
        logger.debug(f"Qdrant URL: {config.database.url}, Collection: {config.database.collection_name}")

        result = await ingest_documents_to_qdrant_async(
            directory_path=directory_path,
            qdrant_url=config.database.url,
            qdrant_api=config.database.api_key or "",
            collection_name=config.database.collection_name,
        )
        
        logger.info(f"Successfully ingested documents. Chunks ingested: {result.get('chunks_ingested', 0)}")
        
        return {
            "status": result.get("status", "unknown"),
            "chunks_ingested": result.get("chunks_ingested", 0),
            "processed_path": directory_path
        }
        
    except Exception as e:
        logger.error(f"Error during document ingestion: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during document ingestion: {str(e)}"
        )


# ========================
# ✅ Collection Create/Delete
# ========================
@app.post("/collection")
async def collection_manage(action: str = Form(...)):
    try:
        logger.info(f"Collection management action: {action}")
        
        if action == "create":
            from qdrant_client.http import models
            
            # Get collection details from config
            collection_name = config.database.collection_name
            logger.info(f"Creating collection: {collection_name}")
            
            # Create collection with proper configuration
            result = check_or_create_qdrant_collection(
                qdrant_url=config.database.url, 
                qdrant_api=config.database.api_key or "", 
                collection_name=collection_name,
                vector_size=768,  # Default size, can be configured
                distance_metric=models.Distance.COSINE,
                sparse_vectors=True
            )
            
            logger.info(f"Collection operation result: {result}")
            return {"status": "success", "message": result}  
            
        elif action == "delete":
            collection_name = config.database.collection_name
            logger.info(f"Deleting collection: {collection_name}")
            
            delete_qdrant_collection(
                qdrant_url=config.database.url, 
                qdrant_api=config.database.api_key or "", 
                collection_name=collection_name
            )
            return {"status": "success", "message": f"Collection '{collection_name}' deleted"}
            
        else:
            raise HTTPException(
                status_code=400, 
                detail="Invalid action: must be 'create' or 'delete'"
            )
            
    except Exception as e:
        error_msg = f"Error in collection management: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=error_msg
        )


@app.get("/collections/stats")
def collection_stats():
    """
    Return all Qdrant collections with vector (chunk) counts.
    """
    try:
        return get_collections_with_chunk_counts(config.database.url, config.database.api_key or "")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/ask")
@timeit
async def ask_hr(request: QueryRequest):
    try:
        # Get or create session
        session_id = get_or_create_session(request.session_id)
        
        logger.info(f"Session {session_id}: Received question: {request.question}")
        
        # Add user message to history
        sessions[session_id].append({
            "role": "user",
            "content": request.question,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Get chat history (all messages except the current one we just added)
        chat_history = sessions[session_id][:-1]
        
        # Time chain initialization
        start_chain = time.time()
        create_chain_with_history = get_cached_hr_assistant_chain(
            config.database.url, 
            config.database.api_key, 
            config.database.collection_name
        )
        logger.info(f"Chain init: {time.time() - start_chain:.2f}s")
        
        # Create a chain with the current chat history
        qa_chain = create_chain_with_history(chat_history)
        
        # Time question processing
        start_process = time.time()
        result = qa_chain.invoke({"input": request.question})
        logger.info(f"Question processing: {time.time() - start_process:.2f}s")

        # Add assistant response to history
        sessions[session_id].append({
            "role": "assistant",
            "content": result.get("answer", "No response."),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep only the most recent message pair (user + assistant)
        if len(sessions[session_id]) > 2:
            sessions[session_id] = sessions[session_id][-2:]

        # Extract document names from context if available
        sources = [] 
        context = result.get("context")
        if context: 
            import os
            for doc in context:
                source_path = None
                if isinstance(doc, dict):
                    source_path = doc.get("metadata", {}).get("source")
                else:
                    source_path = getattr(getattr(doc, "metadata", {}), "get", lambda k, d=None: None)("source")
                if source_path:
                    sources.append(os.path.basename(source_path))
                    
        return {
            "session_id": session_id,
            "answer": result.get("answer", "No response."),
            "sources": sources,
            "processing_time": f"{time.time() - start_process:.2f}s"
        }
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}")
async def get_session_history(session_id: str):
    """Get the message history for a specific session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "messages": sessions[session_id]
    }

# Add health check endpoint
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy", "service": "hr-assistant"}

@app.get("/test/concurrency")
async def test_concurrency():
    """Test endpoint for concurrency testing"""
    import time
    import threading
    # Simulate some work (5ms)
    time.sleep(0.005)
    return {
        "worker_id": os.getpid(),
        "thread_id": threading.get_ident(),
        "timestamp": time.time()
    }

@app.options("/health")
async def health_check_options(): 
    """OPTIONS handler for health check endpoint"""
    return {"status": "ok"}

@app.get("/metrics")
async def get_metrics():
    """Metrics endpoint for monitoring system resources"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        "process": {
            "memory_mb": memory_info.rss / (1024 * 1024),
            "cpu_percent": process.cpu_percent(),
            "threads": process.num_threads(),
            "connections": len(process.net_connections())
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "virtual_memory": psutil.virtual_memory()._asdict(),
            "disk_usage": psutil.disk_usage('/')._asdict()
        },
        "gunicorn": {
            "workers": int(os.getenv('WORKERS', '8')),
            "threads": int(os.getenv('THREADS', '4')),
            "timeout": int(os.getenv('TIMEOUT', '300'))
        }
    }

# Pre-warm the model on startup
