"""
Admin Interface for HR Assistant
Provides a web-based interface for non-technical teams to manage the application.
"""

import os
import json
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Body, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import logging
from dotenv import load_dotenv
import sys
import threading
import shutil

from config import get_config, validate_config

# --- HR Prompt Editing Endpoints ---
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["Admin"])
@router.get("/prompt", tags=["Prompt"])
def get_hr_prompt():
    try:
        with open("ingestion_retrieval/hr_routing_prompt.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        if "hr_routing_prompt" not in data:
            raise KeyError("'hr_routing_prompt' key missing in JSON file.")
        return {"prompt": data["hr_routing_prompt"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prompt file error: {str(e)}")

class PromptUpdateRequest(BaseModel):
    prompt: str

@router.post("/prompt", tags=["Prompt"])
def update_hr_prompt(request: PromptUpdateRequest):
    path = "ingestion_retrieval/hr_routing_prompt.json"
    try:
        print(f"[DEBUG] Received prompt update request.")
        print(f"[DEBUG] Target file path: {path}")
        print(f"[DEBUG] Checking if file exists: {os.path.exists(path)}")
        if not os.path.exists(path):
            print(f"[ERROR] File does not exist: {path}")
            return {"status": "error", "message": f"File does not exist: {path}"}
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                print(f"[DEBUG] Successfully loaded JSON: {data}")
            except Exception as json_err:
                print(f"[ERROR] Failed to load JSON: {json_err}")
                raise
        print(f"[DEBUG] Updating 'hr_routing_prompt' value.")
        data["hr_routing_prompt"] = request.prompt
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[DEBUG] Successfully wrote updated prompt to file.")
        return {"status": "success"}
    except Exception as e:
        print(f"[ERROR] Exception in update_hr_prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reset_env_default", tags=["Admin"])
def reset_env_default():
    try:
        shutil.copyfile("default/.env", ".env")
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset .env: {str(e)}")

@router.post("/reset_prompt_default", tags=["Prompt"])
def reset_prompt_default():
    src = "default/hr_routing_prompt.json"
    dst = "ingestion_retrieval/hr_routing_prompt.json"
    try:
        print(f"[DEBUG] Reset prompt request received.")
        print(f"[DEBUG] Source path: {src}")
        print(f"[DEBUG] Destination path: {dst}")
        print(f"[DEBUG] Checking if source exists: {os.path.exists(src)}")
        print(f"[DEBUG] Checking if destination directory exists: {os.path.exists(os.path.dirname(dst))}")
        if not os.path.exists(src):
            print(f"[ERROR] Source file does not exist: {src}")
            return {"status": "error", "message": f"Source file does not exist: {src}"}
        if not os.path.exists(os.path.dirname(dst)):
            print(f"[ERROR] Destination directory does not exist: {os.path.dirname(dst)}")
            return {"status": "error", "message": f"Destination directory does not exist: {os.path.dirname(dst)}"}
        print(f"[DEBUG] Attempting to copy file...")
        shutil.copyfile(src, dst)
        print(f"[DEBUG] Successfully copied {src} to {dst}")
        # Optionally, check file contents after copy
        with open(dst, "r", encoding="utf-8") as f:
            data = f.read()
            print(f"[DEBUG] Destination file contents after copy: {data[:200]}...")  # Print first 200 chars
        return {"status": "success"}
    except Exception as e:
        print(f"[ERROR] Exception in reset_prompt_default: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset prompt: {str(e)}")

# Pydantic models for admin operations
class SystemStatus(BaseModel):
    status: str
    uptime: str
    memory_usage: Dict[str, Any]
    disk_usage: Dict[str, Any]
    active_connections: int
    api_stats: Dict[str, int] = {"total_calls": 0, "successful_calls": 0, "error_calls": 0}
    collection_stats: Dict[str, Any]

class ConfigurationUpdate(BaseModel):
    section: str
    key: str
    value: str

class EnvFileUpdate(BaseModel):
    key: str
    value: str
    description: str = ""

class BackupRequest(BaseModel):
    include_vectors: bool = True
    include_uploads: bool = True
    description: str = ""

class MaintenanceRequest(BaseModel):
    action: str  # "cleanup_logs", "optimize_collection", "restart_services"
    parameters: Dict[str, Any] = {}

class AdminDashboard:
    """Admin dashboard functionality"""
    
    def __init__(self):
        self.config = get_config()
        self.start_time = datetime.now()
        self.env_file_path = Path(".env")
        self._load_env_variables()
        # Initialize API call counters
        self.api_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "error_calls": 0
        }
    
    def _load_env_variables(self):
        """Load environment variables from .env file and system environment"""
        try:
            # Load from .env file
            env_vars = self.read_env_file()
            # Load from system environment (overrides .env)
            for key, value in os.environ.items():
                env_vars[key] = value
            self.env_vars = env_vars
            return True
        except Exception as e:
            logger.error(f"Error loading environment variables: {e}")
            return False
    
    def read_env_file(self) -> Dict[str, str]:
        """Read the .env file and return as dictionary"""
        try:
            if not self.env_file_path.exists():
                return {}
            
            env_vars = {}
            with open(self.env_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Handle quoted values
                        if value.startswith('"') and value.endswith('"'):
                            # Remove outer quotes and unescape inner quotes
                            value = value[1:-1].replace('\\"', '"')
                        elif value.startswith("'") and value.endswith("'"):
                            # Remove outer single quotes
                            value = value[1:-1]
                        
                        env_vars[key] = value
            return env_vars
        except Exception as e:
            logger.error(f"Error reading .env file: {e}")
            return {}

    def reload_environment(self):
        """Reload environment variables from both .env file and system"""
        success = self._load_env_variables()
        if success:
            # Reload any dependent configurations
            self.config = get_config()
            return True
        return False
    
    def write_env_file(self, env_vars: Dict[str, str]) -> bool:
        """Write the .env file from dictionary"""
        try:
            # Create backup of current .env file
            if self.env_file_path.exists():
                backup_path = self.env_file_path.with_suffix('.env.backup')
                shutil.copy2(self.env_file_path, backup_path)
                logger.info(f"Created backup of .env file: {backup_path}")
            
            # Write new .env file
            with open(self.env_file_path, 'w', encoding='utf-8') as f:
                for key, value in env_vars.items():
                    # Properly format the value - add quotes if it contains spaces or special characters
                    if value and (' ' in value or '"' in value or "'" in value or '#' in value):
                        # Escape quotes and wrap in double quotes
                        escaped_value = value.replace('"', '\\"')
                        formatted_value = f'"{escaped_value}"'
                    else:
                        formatted_value = value
                    f.write(f"{key}={formatted_value}\n")
            
            logger.info("Successfully updated .env file")
            return True
        except Exception as e:
            logger.error(f"Error writing .env file: {e}")
            return False
    
    def update_env_variable(self, key: str, value: str) -> Dict[str, Any]:
        """Update a single environment variable in .env file"""
        try:
            env_vars = self.read_env_file()
            env_vars[key] = value
            
            if self.write_env_file(env_vars):
                return {
                    "status": "success",
                    "message": f"Successfully updated {key}",
                    "key": key,
                    "value": value
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to write .env file"
                }
        except Exception as e:
            logger.error(f"Error updating env variable: {e}")
            return {
                "status": "error",
                "message": f"Error updating {key}: {str(e)}"
            }
    
    def get_env_file_content(self) -> Dict[str, Any]:
        """Get the current .env file content with descriptions"""
        env_vars = self.read_env_file()
        
        # Define descriptions for common environment variables
        descriptions = {
            "QDRANT_URL": "Qdrant database server URL",
            "QDRANT_API_KEY": "Qdrant API key (leave empty if not using authentication)",
            "QDRANT_COLLECTION": "Name of the Qdrant collection for HR documents",
            "GOOGLE_API_KEY": "Google API key for embeddings and LLM (required)",
            "GROQ_API_KEY": "Groq API key for LLM (optional, for faster responses)",
            "OPENROUTER_API_KEY": "OpenRouter API key for LLM (optional)",
            "MODEL_PRIORITY": "LLM priority: 1=Groq, 2=Google, 3=OpenRouter",
            "GROQ_MODEL": "Groq model name (e.g., llama3-70b-8192)",
            "GOOGLE_MODEL": "Google model name (e.g., gemma-3-12b-it)",
            "OPENROUTER_MODEL": "OpenRouter model name (e.g., gpt-4)",
            "TEMPERATURE": "AI model temperature (0.1=factual, 0.8=creative)",
            "EMBEDDING_MODEL": "Google embedding model name",
            "COHERE_API_KEY": "Cohere API key for fallback embeddings",
            "COHERE_EMBEDDING_MODEL": "Cohere embedding model name",
            "PORT": "Server port (default: 8000)",
            "WORKERS": "Number of worker processes (default: 4)",
            "TIMEOUT": "Request timeout in seconds (default: 120)",
            "KEEP_ALIVE": "Keep-alive timeout in seconds (default: 30)",
            "LOG_LEVEL": "Logging level: DEBUG, INFO, WARNING, ERROR",
            "MAX_FILE_SIZE_MB": "Maximum file upload size in MB (default: 200)",
            "ALLOWED_FILE_EXTENSIONS": "Comma-separated list of allowed file extensions",
            "UPLOAD_DIR": "Directory for uploaded files (default: uploaded_folders)",
            "CHUNK_SIZE": "Document chunk size for processing (default: 512)",
            "CHUNK_OVERLAP": "Document chunk overlap (default: 128)",
            "SEARCH_K": "Number of search results to retrieve (default: 4)",
            "FETCH_K": "Number of results to fetch before filtering (default: 15)",
            "SCORE_THRESHOLD": "Minimum similarity score (0.0-1.0, default: 0.5)",
            "ALLOWED_ORIGINS": "Comma-separated list of allowed CORS origins",
            "RATE_LIMIT_PER_MINUTE": "API rate limit per minute per IP (default: 60)",
            "ENABLE_CORS": "Enable CORS (true/false, default: true)",
            "ENABLE_REQUEST_LOGGING": "Enable detailed request logging (true/false)",
            "ENABLE_PERFORMANCE_MONITORING": "Enable performance monitoring (true/false)",
            "LOG_FILE_PATH": "Path to log file (default: logs/hr_assistant.log)",
            "AUTO_BACKUP_ENABLED": "Enable automatic backups (true/false, default: true)",
            "BACKUP_FREQUENCY_HOURS": "Backup frequency in hours (default: 24)",
            "BACKUP_RETENTION_DAYS": "Backup retention in days (default: 30)",
            "BACKUP_DIR": "Backup directory (default: backups)"
        }
        
        return {
            "variables": env_vars,
            "descriptions": descriptions,
            "file_path": str(self.env_file_path.absolute())
        }
    
    def update_api_stats(self, success: bool = True) -> None:
        """Update API call statistics
        
        Args:
            success: Whether the API call was successful
        """
        self.api_stats["total_calls"] += 1
        if success:
            self.api_stats["successful_calls"] += 1
        else:
            self.api_stats["error_calls"] += 1

    def get_system_status(self) -> SystemStatus:
        """Get comprehensive system status"""
        try:
            # Calculate uptime
            uptime = datetime.now() - self.start_time
            
            # Get memory usage (simplified)
            import psutil
            memory = psutil.virtual_memory()
            memory_usage = {
                "total_gb": round(memory.total / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "percent": memory.percent
            }
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_usage = {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent": round((disk.used / disk.total) * 100, 2)
            }
            
            # Get collection stats
            from ingestion_retrieval.ingestion import get_collections_with_chunk_counts
            collection_stats = get_collections_with_chunk_counts(
                self.config.database.url,
                self.config.database.api_key or ""
            )
            
            return SystemStatus(
                status="healthy",
                uptime=str(uptime).split('.')[0],  # Remove microseconds
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                active_connections=0,  # Would need to implement connection tracking
                api_stats=self.api_stats,
                collection_stats=collection_stats
            )
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return SystemStatus(
                status="error",
                uptime="unknown",
                memory_usage={},
                disk_usage={},
                active_connections=0,
                api_stats={"total_calls": 0, "successful_calls": 0, "error_calls": 0},
                collection_stats={}
            )
    
    def _get_last_backup_time(self) -> str:
        """Get the last backup time"""
        backup_dir = Path(self.config.backup.backup_dir)
        if backup_dir.exists():
            backup_files = list(backup_dir.glob("*.tar.gz"))
            if backup_files:
                latest = max(backup_files, key=lambda x: x.stat().st_mtime)
                return datetime.fromtimestamp(latest.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        return "No backups found"
    
    def create_backup(self, request: BackupRequest) -> Dict[str, Any]:
        """Create a system backup"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"hr_assistant_backup_{timestamp}.tar.gz"
            backup_path = Path(self.config.backup.backup_dir) / backup_name
            
            # Create backup directory if it doesn't exist
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup archive
            import tarfile
            with tarfile.open(backup_path, "w:gz") as tar:
                # Add uploaded files if requested
                if request.include_uploads:
                    upload_dir = Path(self.config.file_upload.upload_dir)
                    if upload_dir.exists():
                        tar.add(upload_dir, arcname="uploaded_folders")
                
                # Add logs
                log_dir = Path("logs")
                if log_dir.exists():
                    tar.add(log_dir, arcname="logs")
                
                # Add configuration
                config_files = [".env", "config.py"]
                for config_file in config_files:
                    if Path(config_file).exists():
                        tar.add(config_file)
            
            # Create backup metadata
            metadata = {
                "timestamp": timestamp,
                "description": request.description,
                "include_vectors": request.include_vectors,
                "include_uploads": request.include_uploads,
                "size_mb": round(backup_path.stat().st_size / (1024**2), 2)
            }
            
            metadata_path = backup_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Backup created: {backup_name}")
            return {
                "status": "success",
                "backup_name": backup_name,
                "size_mb": metadata["size_mb"],
                "message": "Backup created successfully"
            }
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            raise HTTPException(status_code=500, detail=f"Backup failed: {str(e)}")
    
    def cleanup_old_backups(self) -> Dict[str, Any]:
        """Clean up old backups based on retention policy"""
        try:
            backup_dir = Path(self.config.backup.backup_dir)
            if not backup_dir.exists():
                return {"status": "success", "message": "No backup directory found"}
            
            cutoff_date = datetime.now() - timedelta(days=self.config.backup.backup_retention_days)
            deleted_count = 0
            
            for backup_file in backup_dir.glob("*.tar.gz"):
                if datetime.fromtimestamp(backup_file.stat().st_mtime) < cutoff_date:
                    backup_file.unlink()
                    # Also delete metadata file
                    metadata_file = backup_file.with_suffix('.json')
                    if metadata_file.exists():
                        metadata_file.unlink()
                    deleted_count += 1
            
            return {
                "status": "success",
                "deleted_count": deleted_count,
                "message": f"Deleted {deleted_count} old backups"
            }
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")
    
    def cleanup_logs(self, days_to_keep: int = 30) -> Dict[str, Any]:
        """Clean up old log files"""
        try:
            log_dir = Path("logs")
            if not log_dir.exists():
                return {"status": "success", "message": "No log directory found"}
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            deleted_count = 0
            
            for log_file in log_dir.glob("*.log*"):
                if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_date:
                    log_file.unlink()
                    deleted_count += 1
            
            return {
                "status": "success",
                "deleted_count": deleted_count,
                "message": f"Deleted {deleted_count} old log files"
            }
            
        except Exception as e:
            logger.error(f"Log cleanup failed: {e}")
            raise HTTPException(status_code=500, detail=f"Log cleanup failed: {str(e)}")
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration"""
        try:
            active_llm = self.config.get_active_llm_config()
            provider = active_llm.get("provider", "unknown")
            model = active_llm.get("model", "")
            
            return {
                "database": {
                    "url": self.config.database.url,
                    "collection": self.config.database.collection_name,
                    "vector_size": self.config.database.vector_size
                },
                "llm": {
                    "active_provider": provider,
                    "model": model,  # Frontend expects this as 'model' not 'active_model'
                    "temperature": getattr(self.config.llm, 'temperature', 0.8)
                },
                "server": {
                    "port": getattr(self.config.server, 'port', 8000),
                    "workers": getattr(self.config.server, 'workers', 4),
                    "log_level": getattr(self.config.server, 'log_level', 'INFO')
                },
                "file_upload": {
                    "max_size_mb": getattr(self.config.file_upload, 'max_file_size_mb', 200),
                    "allowed_extensions": getattr(self.config.file_upload, 'allowed_extensions', ['.pdf', '.docx', '.txt'])
                },
                "backup": {
                    "enabled": getattr(self.config.backup, 'auto_backup_enabled', True),
                    "frequency_hours": getattr(self.config.backup, 'backup_frequency_hours', 24),
                    "retention_days": getattr(self.config.backup, 'backup_retention_days', 7)
                }
            }
        except Exception as e:
            logger.error(f"Error in get_configuration_summary: {str(e)}")
            # Return a safe default configuration if there's an error
            return {
                "database": {
                    "url": "unknown",
                    "collection": "unknown",
                    "vector_size": 0
                },
                "llm": {
                    "active_provider": "error",
                    "model": "unknown",
                    "temperature": 0.8
                },
                "server": {
                    "port": 8000,
                    "workers": 4,
                    "log_level": "ERROR"
                },
                "file_upload": {
                    "max_size_mb": 200,
                    "allowed_extensions": ['.pdf', '.docx', '.txt']
                },
                "backup": {
                    "enabled": False,
                    "frequency_hours": 24,
                    "retention_days": 7
                }
            }

class ModelUpdateRequest(BaseModel):
    provider: str
    groq_model: str
    google_model: str
    openrouter_model: str
    embedding_provider: str
    embedding_model_google: str
    embedding_model_cohere: str

# Global admin dashboard instance
admin_dashboard = AdminDashboard()

# API Endpoints

# --- HR Prompt Editing in Admin Dashboard HTML ---
# Add a section to the admin dashboard HTML for editing the prompt.
# This is a minimal JS/HTML snippet to be inserted in the admin_dashboard_html() function:
# (You may want to style/integrate it as needed)
#
# <section id="hr-prompt-section">
#   <h2>Edit HR Routing Prompt</h2>
#   <textarea id="hr-prompt-textarea" rows="20" style="width:100%"></textarea><br>
#   <button onclick="savePrompt()">Save Prompt</button>
#   <span id="hr-prompt-status"></span>
# </section>
# <script>
# async function loadPrompt() {
#   const res = await fetch('/admin/prompt');
#   const data = await res.json();
#   document.getElementById('hr-prompt-textarea').value = data.prompt;
# }
# async function savePrompt() {
#   const prompt = document.getElementById('hr-prompt-textarea').value;
#   const res = await fetch('/admin/prompt', {
#     method: 'POST',
#     headers: {'Content-Type': 'application/json'},
#     body: JSON.stringify({prompt})
#   });
#   if (res.ok) {
#     document.getElementById('hr-prompt-status').innerText = 'Saved!';
#   } else {
#     document.getElementById('hr-prompt-status').innerText = 'Error saving prompt.';
#   }
# }
# window.onload = loadPrompt;
# </script>
#
# Insert this section in your admin_dashboard_html() HTML response where appropriate.
@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status and health information"""
    return admin_dashboard.get_system_status()

@router.post("/backup")
async def create_backup(request: BackupRequest):
    """Create a system backup"""
    return admin_dashboard.create_backup(request)

@router.post("/backup/cleanup")
async def cleanup_backups():
    """Clean up old backups based on retention policy"""
    return admin_dashboard.cleanup_old_backups()

@router.post("/logs/cleanup")
async def cleanup_logs(days_to_keep: int = 30):
    """Clean up old log files"""
    return admin_dashboard.cleanup_logs(days_to_keep)

@router.get("/config")
async def get_configuration():
    """Get current configuration summary"""
    return admin_dashboard.get_configuration_summary()

@router.post("/config/validate")
async def validate_current_config():
    """Validate current configuration"""
    try:
        validate_config()
        return {"status": "success", "message": "Configuration is valid"}
    except ValueError as e:
        return {"status": "error", "message": str(e)}

@router.get("/env")
async def get_env_file():
    """Get the current .env file content"""
    return admin_dashboard.get_env_file_content()

@router.post("/env/update")
async def update_env_variable(request: EnvFileUpdate):
    result = admin_dashboard.update_env_variable(request.key, request.value)
    # Reload .env after update
    load_dotenv(override=True)
    # If Qdrant connection parameters changed, invalidate cache
    if result["status"] == "success" and request.key in ["QDRANT_URL", "QDRANT_API_KEY", "QDRANT_COLLECTION"]:
        try:
            from ingestion_retrieval.retrieval import invalidate_cache
            invalidate_cache()
            result["message"] += " Cache invalidated for new Qdrant connection."
            logger.info(f"Cache invalidated due to {request.key} change")
        except Exception as e:
            logger.warning(f"Failed to invalidate cache: {e}")
    return result

@router.post("/env/reload")
async def reload_configuration():
    load_dotenv(override=True)
    try:
        from ingestion_retrieval.retrieval import invalidate_cache
        invalidate_cache()
        return {
            "status": "success",
            "message": "Configuration reloaded. New settings will be used for the next request."
        }
    except ValueError as e:
        return {
            "status": "error",
            "message": f"Configuration validation failed: {str(e)}"
        }

@router.post("/cache/invalidate")
async def invalidate_all_caches():
    """Manually invalidate all caches (useful for troubleshooting)"""
    try:
        from ingestion_retrieval.retrieval import invalidate_cache
        invalidate_cache()
        return {
            "status": "success",
            "message": "All caches invalidated successfully. New connections will be established."
        }
    except Exception as e:
        logger.error(f"Failed to invalidate cache: {e}")
        return {
            "status": "error",
            "message": f"Failed to invalidate cache: {str(e)}"
        }

@router.get("/dashboard", response_class=HTMLResponse)
async def admin_dashboard_html():
    """Admin dashboard HTML interface"""
    return """
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <meta name='viewport' content='width=device-width, initial-scale=1.0'>
        <title>HR Assistant Admin Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #FF9800;
                --primary-dark: #F57C00;
                --danger: #D32F2F;
                --danger-dark: #B71C1C;
                --success: #00C853;
                --success-dark: #009624;
                --bg: #121212;
                --bg-card: #181A1B;
                --border: #232323;
                --text: #E0E0E0;
                --text-muted: #9E9E9E;
            }
            html { box-sizing: border-box; }
            *, *:before, *:after { box-sizing: inherit; }
            body {
                font-family: 'Inter', Arial, sans-serif;
                margin: 0;
                padding: 0;
                background: var(--bg);
                min-height: 100vh;
                color: var(--text);
                letter-spacing: 0.01em;
            }
            .container {
                max-width: 930px;
                margin: 48px auto 32px auto;
                padding: 40px 32px 32px 32px;
                background: var(--bg-card);
                border-radius: 20px;
                box-shadow: 0 6px 36px rgba(0,0,0,0.32);
            }
            .card {
                background: var(--bg-card);
                border: 1.5px solid var(--border);
                padding: 32px 28px;
                margin: 28px 0;
                border-radius: 14px;
                box-shadow: 0 2px 18px rgba(0,0,0,0.18);
                transition: box-shadow 0.2s, border 0.2s;
            }
            .card:hover {
                box-shadow: 0 6px 32px rgba(0,0,0,0.22);
                border-color: var(--primary);
            }
            .status-healthy { border-left: 6px solid var(--success); }
            .status-error { border-left: 6px solid var(--danger); }
            .button {
                background: var(--primary);
                color: var(--text);
                padding: 12px 28px;
                border: none;
                border-radius: 7px;
                cursor: pointer;
                margin: 7px 4px 7px 0;
                font-weight: 600;
                font-size: 1.08em;
                letter-spacing: 0.01em;
                transition: background 0.18s, box-shadow 0.18s;
                box-shadow: 0 1px 4px rgba(41,121,255,0.10);
            }
            .button:hover { background: var(--primary-dark); }
            .button-danger { background: var(--danger); }
            .button-danger:hover { background: var(--danger-dark); }
            .button-success { background: var(--success); color: var(--text); }
            .button-success:hover { background: var(--success-dark); }
            .metric {
                display: inline-block;
                margin: 14px 18px 14px 0;
                padding: 18px 28px;
                background: #232323;
                border-radius: 7px;
                color: var(--primary);
                font-size: 1.22em;
                font-weight: 600;
                box-shadow: 0 1px 6px rgba(0,0,0,0.09);
                min-width: 120px;
                text-align: center;
            }
            .env-section { margin: 32px 0 22px 0; }
            .env-input {
                width: 100%;
                padding: 11px;
                margin: 7px 0 15px 0;
                border: 1.5px solid var(--border);
                border-radius: 4px;
                background: var(--bg);
                color: var(--text); 
                font-size: 1.08em;
                font-family: 'Inter', Arial, sans-serif;
            }
            .env-description { color: var(--text-muted); font-size: 1.01em; margin-bottom: 6px; }
            .tab { display: none; }
            .tab.active { display: block; }
            .tab-bar {
                display: flex;
                flex-direction: row;
                justify-content: center;
                align-items: center;
                border-bottom: 2px solid var(--border);
                margin-bottom: 34px;
                gap: 0;
            }
            .tab-button {
                flex: 1 1 0;
                min-width: 150px;
                max-width: 220px;
                height: 54px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: var(--bg-card);
                border: 1.5px solid var(--border); 
                border-bottom: none;
                color: var(--text-muted);
                font-size: 1.13em;
                border-radius: 10px 10px 0 0;
                margin-right: 0;
                margin-bottom: -2px;
                transition: background 0.18s, color 0.18s, border 0.18s;
                font-weight: 600;
                box-sizing: border-box;
                padding: 0 8px;
            }
            .tab-button.active {
                background: var(--primary);
                color: var(--text);
                border-bottom: 2.5px solid var(--primary);
            }
            h1 {
                color: var(--text);
                font-size: 2.5em;
                font-weight: 700;
                margin-bottom: 6px;
                letter-spacing: -0.015em;
            }
            h2 {
                color: var(--primary);
                font-size: 1.4em;
                margin: 18px 0 8px 0;
                font-weight: 600;
            }
            h3 { color: var(--text); font-size: 1.15em; margin: 13px 0 6px 0; font-weight: 600; }
            label { color: var(--primary); font-weight: 600; font-size: 1.05em; }
            select, input[type=text] {
                background: var(--bg);
                color: var(--text);
                border: 1.5px solid var(--border);
                border-radius: 4px;
                padding: 9px 13px;
                margin: 7px 0 18px 0;
                font-size: 1.08em;
                font-family: 'Inter', Arial, sans-serif;
            }
            /* Responsive */
            @media (max-width: 700px) {
                .container { padding: 15px 5px; }
                .card { padding: 18px 8px; }
                h1 { font-size: 1.4em; }
                .metric { padding: 11px 6px; font-size: 1em; min-width: 80px; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div style="display: flex; align-items: center; gap: 28px; margin-bottom: 24px;">
                <img src="/static/Qadri_logo.png" alt="Qadri Logo" style="width: 92px; height: 92px; object-fit: contain;">
                <h1 style="margin: 0;">HR Assistant Admin Dashboard</h1>
            </div>


            <script>
            // Global notification system
            function showNotification(message, type = 'info') {
                const container = document.getElementById('toast-container');
                if (!container) {
                    console.error('Notification container not found');
                    return;
                }
                
                // Create notification element
                const toast = document.createElement('div');
                toast.className = `toast-notification ${type}`;
                toast.innerHTML = `
                    <div class="toast-icon">${type === 'error' ? '❌' : type === 'success' ? '✅' : 'ℹ️'}</div>
                    <div class="toast-message">${message}</div>
                    <button class="toast-close" onclick="this.parentElement.remove()">×</button>
                `;
                
                // Add to container
                container.appendChild(toast);
                
                // Auto-remove after delay
                setTimeout(() => {
                    toast.classList.add('fade-out');
                    setTimeout(() => toast.remove(), 300);
                }, 5000);
            }
            
            // Add styles for notifications
            const style = document.createElement('style');
            style.textContent = `
                .toast-notification {
                    display: flex;
                    align-items: center;
                    padding: 12px 16px;
                    margin-bottom: 12px;
                    border-radius: 4px;
                    color: white;
                    background: #333;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
                    transform: translateX(100%);
                    animation: slideIn 0.3s forwards;
                    min-width: 250px;
                    max-width: 350px;
                }
                .toast-notification.success { background: #4CAF50; }
                .toast-notification.error { background: #F44336; }
                .toast-notification.warning { background: #FF9800; }
                .toast-notification.info { background: #2196F3; }
                .toast-icon { margin-right: 12px; font-size: 1.2em; }
                .toast-message { flex: 1; }
                .toast-close {
                    background: none;
                    border: none;
                    color: white;
                    font-size: 1.2em;
                    cursor: pointer;
                    padding: 0 0 0 12px;
                    margin: 0;
                    opacity: 0.7;
                }
                .toast-close:hover { opacity: 1; }
                @keyframes slideIn {
                    to { transform: translateX(0); }
                }
                .fade-out {
                    animation: fadeOut 0.3s forwards;
                }
                @keyframes fadeOut {
                    to { opacity: 0; transform: translateX(100%); }
                }
            `;
            document.head.appendChild(style);
            
            // Make sure the notification container exists
            document.addEventListener('DOMContentLoaded', () => {
                if (!document.getElementById('toast-container')) {
                    const container = document.createElement('div');
                    container.id = 'toast-container';
                    container.style.position = 'fixed';
                    container.style.top = '30px';
                    container.style.right = '30px';
                    container.style.zIndex = '9999';
                    document.body.appendChild(container);
                }
            });
            
            async function loadPrompt() {
                try {
                    const res = await fetch('/admin/prompt');
                    const data = await res.json();
                    document.getElementById('hr-prompt-textarea').value = data.prompt;
                } catch (e) {
                    document.getElementById('hr-prompt-status').innerText = 'Error loading prompt.';
                }
            }
            async function savePrompt() {
                const prompt = document.getElementById('hr-prompt-textarea').value;
                const res = await fetch('/admin/prompt', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt})
                });
                if (res.ok) {
                    document.getElementById('hr-prompt-status').innerText = 'Saved!';
                } else {
                    document.getElementById('hr-prompt-status').innerText = 'Error saving prompt.';
                }

            }
            window.addEventListener('DOMContentLoaded', loadPrompt);
            async function resetEnvDefault() {
    const status = document.getElementById('env-default-status');
    status.innerText = 'Resetting...';
    try {
        const res = await fetch('/admin/reset_env_default', {method: 'POST'});
        if (res.ok) {
            status.innerText = 'ENV reset to default!';
        } else {
            status.innerText = 'Failed to reset ENV.';
        }
    } catch (e) {
        status.innerText = 'Error resetting ENV.';
    }
}

async function resetPromptDefault() {
    const status = document.getElementById('prompt-default-status');
    status.innerText = 'Resetting...';
    try {
        const res = await fetch('/admin/reset_prompt_default', {method: 'POST'});
        if (res.ok) {
            status.innerText = 'Prompt reset to default!';
            loadPrompt(); // Optionally reload the prompt in the textarea
        } else {
            status.innerText = 'Failed to reset prompt.';
        }
    } catch (e) {
        status.innerText = 'Error resetting prompt.';
    }
}
            </script>
            <!-- Toast Notification -->
            <div id="toast-container" style="position:fixed;top:30px;right:30px;z-index:9999;"></div>
            <div class="tab-bar">
            <div class="tab-button active" onclick="showTab('upload')">Upload & Ingest</div>
            <div class="tab-button" onclick="showTab('status')">System Status</div>
            <div class="tab-button" onclick="showTab('model')">Model</div>
            <div class="tab-button" onclick="showTab('config')">Configuration</div>
            <div class="tab-button" onclick="showTab('prompt')">HR Prompt</div>
        </div>
        <div id="upload" class="tab active">
            <div class="card">
                <h2>Upload & Ingest Documents</h2>
                <form id="upload-form" enctype="multipart/form-data" onsubmit="event.preventDefault(); uploadFiles();">
                    <label for="file-input">Select a folder or files to upload (subfolders supported):</label><br>
                    <input type="file" id="file-input" name="files" webkitdirectory directory multiple style="background:#222; color:#fff; border:1px solid #444; border-radius:3px; padding:7px 10px; margin:10px 0;">
                    <button class="button" type="submit">Upload</button>
                </form>
                <div id="upload-status" style="margin-top:10px;"></div>
                <button class="button button-success" style="margin-top:20px;" onclick="ingestNow()">Ingest Now</button>
                <div id="ingest-status" style="margin-top:10px;"></div>
            </div>
            <div class="card" id="collection-stats-card">
                <h2>Collection Stats</h2>
                <div id="collection-stats-content">Loading...</div>
            </div>
            <div class="card" id="collection-mgmt-card">
                <h2>Collection Management</h2>
                <form id="collection-mgmt-form" onsubmit="event.preventDefault(); handleCollectionAction();">
                    <label for="collection-action">Action:</label>
                    <select id="collection-action">
                        <option value="create">Create</option>
                        <option value="delete">Delete</option>
                    </select>
                    <label for="collection-name" style="margin-left:18px;">Collection Name:</label>
                    <input type="text" id="collection-name" placeholder="Collection name" style="width:200px;">
                    <button class="button" type="submit" style="margin-left:18px;">Go</button>
                </form>
            </div>
        </div>
        <div id="status" class="tab">
            <div class="card">
                <h2>System Status</h2>
                <div id="status-content"></div>
            </div>
            <div class="card">
                <h2>Current Configuration</h2>
                <div id="config-content"></div>
                <div style="color:#ffb366; margin-top:10px; font-size:0.95em;">
                    Note: The LLM shown here is what will be used for <b>new requests</b>.<br>
                    If you just changed the model, the current request may still use the previous LLM until the cache is refreshed.
                </div>
                <div id="active-llm-content" style="margin-top:10px;"></div>
            </div>
        </div>
        <div id="model" class="tab">
            <div class="card">
                <div id="model-content">
                    <!-- Model cards will be populated by JavaScript -->
                    <div style="text-align: center; padding: 40px 0; color: var(--text-muted);">
                        Loading model configurations...
                    </div>
                </div>
                
                <div id="model-status" style="margin-top: 20px;"></div>
            </div>
        </div>
        <div id="config" class="tab">
            <div class="card">   
                <h2>Environment Configuration</h2>
                <p>Edit configuration values below. Changes are saved to the .env file immediately.</p>
                <p><strong>Note:</strong> The current active model and configuration are shown in the System Status tab.</p>
                <button class="button" onclick="resetEnvDefault()">Reset to Default</button>
                <span id="env-default-status"></span>
                <div id="env-content"></div>
            </div>
        </div>
        <div id="prompt" class="tab">
        
            <!-- HR Prompt Card Example (add your actual prompt card here)-->
            <div class="card" id="hr-prompt-card" style="position:relative; min-height:220px;">
                <h2>HR Prompt</h2>
                <div style="margin-bottom:18px;">
                    <label for="hr-prompt-textarea">Prompt Text</label><br>
                    <textarea id="hr-prompt-textarea" class="env-input" rows="12" style="resize:vertical; min-height:100px;"></textarea>
                </div>
                             <!-- Floating Reset Button -->
                <button id="reset-prompt-btn" class="button" style="position:absolute; bottom:18px; right:18px; background:transparent; color:var(--primary); border:2px solid var(--primary); font-weight:600; box-shadow:none; transition:background 0.18s, color 0.18s; z-index:2;"
                    onmouseover="this.style.background='var(--primary)';this.style.color='var(--bg-card)';"
                    onmouseout="this.style.background='transparent';this.style.color='var(--primary)';"
                    onclick="resetPromptDefault()"
                >
                    &#8634; Reset to Default
                </button>
                <span id="prompt-default-status"></span>
                <div style="margin-bottom:12px;">
                    <button class="button button-success" onclick="savePrompt()">Save</button>
                    <span id="hr-prompt-status" style="margin-left:12px; color:var(--text-muted);"></span>
                </div>
   
            </div>

        </div>
        <script>
            // Global variables
            let envData = {};
            let configData = {};
            
            // Show notification function (must be in global scope)
            function showNotification(message, type = 'info') {
                let container = document.getElementById('notification-container');
                if (!container) {
                    // Create notification container if it doesn't exist
                    container = document.createElement('div');
                    container.id = 'notification-container';
                    container.style.cssText = `
                        position: fixed;
                        top: 20px;
                        right: 20px;
                        z-index: 1000;
                        max-width: 400px;
                    `;
                    document.body.appendChild(container);
                }
                
                const notification = document.createElement('div');
                notification.className = `notification ${type}`;
                notification.style.cssText = `
                    background: var(--bg-card);
                    border-left: 4px solid ${type === 'error' ? 'var(--danger)' : type === 'warning' ? 'var(--warning)' : 'var(--success)'};
                    border-radius: 4px;
                    padding: 12px 16px;
                    margin-bottom: 10px;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    max-width: 100%;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    animation: slideIn 0.3s ease-out;
                `;
                
                notification.innerHTML = `
                    <div>${message}</div>
                    <button style="
                        background: none;
                        border: none;
                        color: var(--text-muted);
                        cursor: pointer;
                        font-size: 18px;
                        margin-left: 10px;
                        padding: 0 5px;
                    ">&times;</button>
                `;
                
                // Add click handler for the close button
                notification.querySelector('button').onclick = function() {
                    notification.style.animation = 'fadeOut 0.3s ease-out';
                    setTimeout(() => notification.remove(), 300);
                };
                
                container.appendChild(notification);
                
                // Auto-remove after 5 seconds
                setTimeout(() => { 
                    if (notification.parentNode) {
                        notification.style.animation = 'fadeOut 0.3s ease-out';
                        setTimeout(() => notification.remove(), 300);
                    }
                }, 5000);
            }
            function showTab(tabName) {
                document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
                document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
                document.getElementById(tabName).classList.add('active');
                event.target.classList.add('active');
                if (tabName === 'config') loadEnvConfig();
                if (tabName === 'model') loadModelTab();
                if (tabName === 'upload') loadUploadTab();
            }
            function getStatusColor(percent) {
                return percent > 80 ? '#f44336' : percent > 50 ? '#FFC107' : '#4CAF50';
            }

            async function loadStatus() {
                try {
                    const response = await fetch('/admin/status');
                    const status = await response.json();
                    
                    document.getElementById('status-content').innerHTML = `
                        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 15px;">
                            <!-- System Status -->
                            <div class="metric-card" style="
                                background: rgba(30, 41, 59, 0.7);
                                border-radius: 10px;
                                padding: 15px;
                                border-left: 4px solid #4CAF50;
                                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                            ">
                                <div style="display: flex; align-items: center; justify-content: space-between;">
                                    <div>
                                        <div style="font-size: 0.9em; color: #94a3b8;">System Status</div>
                                        <div style="font-size: 1.4em; font-weight: 600; color: #4CAF50;">${status.status.toUpperCase()}</div>
                                        <div style="font-size: 0.8em; color: #64748b; margin-top: 4px;">${status.uptime}</div>
                                    </div>
                                    <div style="
                                        background: #4CAF5020;
                                        width: 40px;
                                        height: 40px;
                                        border-radius: 50%;
                                        display: flex;
                                        align-items: center;
                                        justify-content: center;
                                        font-size: 1.2em;
                                    ">
                                        ${status.status === 'healthy' ? '✅' : '⚠️'}
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Memory Usage -->
                            <div class="metric-card" style="
                                background: rgba(30, 41, 59, 0.7);
                                border-radius: 10px;
                                padding: 15px;
                                border-left: 4px solid ${getStatusColor(status.memory_usage.percent)};
                                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                            ">
                                <div style="display: flex; align-items: center; justify-content: space-between;">
                                    <div>
                                        <div style="font-size: 0.9em; color: #94a3b8;">Memory Usage</div>
                                        <div style="font-size: 1.4em; font-weight: 600; color: ${getStatusColor(status.memory_usage.percent)};">
                                            ${status.memory_usage.percent}%
                                        </div>
                                        <div style="font-size: 0.8em; color: #64748b; margin-top: 4px;">
                                            ${status.memory_usage.used_gb.toFixed(1)}GB / ${status.memory_usage.total_gb.toFixed(1)}GB
                                        </div>
                                    </div>
                                    <div style="
                                        background: ${getStatusColor(status.memory_usage.percent)}20;
                                        width: 40px;
                                        height: 40px;
                                        border-radius: 50%;
                                        display: flex;
                                        align-items: center;
                                        justify-content: center;
                                        font-size: 1.2em;
                                    ">
                                        💾
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Disk Usage -->
                            <div class="metric-card" style="
                                background: rgba(30, 41, 59, 0.7);
                                border-radius: 10px;
                                padding: 15px;
                                border-left: 4px solid ${getStatusColor(status.disk_usage.percent)};
                                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                            ">
                                <div style="display: flex; align-items: center; justify-content: space-between;">
                                    <div>
                                        <div style="font-size: 0.9em; color: #94a3b8;">Disk Usage</div>
                                        <div style="font-size: 1.4em; font-weight: 600; color: ${getStatusColor(status.disk_usage.percent)};">
                                            ${status.disk_usage.percent}%
                                        </div>
                                        <div style="font-size: 0.8em; color: #64748b; margin-top: 4px;">
                                            ${status.disk_usage.used_gb.toFixed(1)}GB / ${status.disk_usage.total_gb.toFixed(1)}GB
                                        </div>
                                    </div>
                                    <div style="
                                        background: ${getStatusColor(status.disk_usage.percent)}20;
                                        width: 40px;
                                        height: 40px;
                                        border-radius: 50%;
                                        display: flex;
                                        align-items: center;
                                        justify-content: center;
                                        font-size: 1.2em;
                                    ">
                                        💽
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                } catch (error) {
                    console.error('Error loading status:', error);
                    document.getElementById('status-content').innerHTML = `
                        <div style="color: #f44336; padding: 15px; background: rgba(244, 67, 54, 0.1); border-radius: 6px;">
                            Error loading system status: ${error.message}
                        </div>
                    `;
                }
            }
            async function loadConfig() {
                try {
                    const response = await fetch('/admin/config');
                    configData = await response.json();
                    let html = '<div class="metric">';
                    html += '<strong>Active LLM:</strong> ' + configData.llm.active_provider + ' - ' + configData.llm.model;
                    html += '</div><div class="metric">';
                    html += '<strong>Temperature:</strong> ' + configData.llm.temperature;
                    html += '</div><div class="metric">';
                    html += '<strong>Database:</strong> ' + configData.database.collection;
                    html += '</div><div class="metric">';
                    html += '<strong>Server Port:</strong> ' + configData.server.port;
                    html += '</div><div class="metric">';
                    html += '<strong>Workers:</strong> ' + configData.server.workers;
                    html += '</div>';
                    document.getElementById('config-content').innerHTML = html;
                    // Also update model tab current model
                    if (document.getElementById('current-model')) {
                        document.getElementById('current-model').innerText = configData.llm.active_provider + ' - ' + configData.llm.model;
                    }
                } catch (error) {
                    document.getElementById('config-content').innerHTML = '<p style="color: red;">Error loading configuration: ' + error.message + '</p>';
                }
            }
            async function loadActiveLLM() {
                try {
                    const response = await fetch('/admin/llm/active');
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    const data = await response.json();
                    let html = '<b>Currently Active LLM in Memory:</b> ';
                    
                    // Safely handle the response data
                    const provider = data?.provider || 'unknown';
                    const model = data?.model || 'unknown';
                    
                    if (provider && provider !== 'unknown') {
                        // Format provider name nicely (capitalize first letter)
                        const formattedProvider = provider.charAt(0).toUpperCase() + provider.slice(1);
                        html += `${formattedProvider} - ${model || 'default'}`;
                        
                        // Add status indicator
                        if (data.is_initialized === false) {
                            html += ' <span style="color: var(--warning);">(Not initialized yet)</span>';
                        }
                    } else if (data?.error) {
                        // Show error if present
                        html += `<span style="color: var(--danger);">Error: ${data.error}</span>`;
                    } else {
                        // No active LLM loaded yet
                        html += '<span style="color:var(--text-muted);">(No LLM loaded yet)</span>';
                    }
                    
                    // Update the UI
                    const activeLLMContent = document.getElementById('active-llm-content');
                    if (activeLLMContent) {
                        activeLLMContent.innerHTML = html;
                    }
                } catch (error) { 
                    console.error('Error loading active LLM:', error);
                    const activeLLMContent = document.getElementById('active-llm-content'); 
                    if (activeLLMContent) {
                        activeLLMContent.innerHTML = `
                            <div style="color: var(--danger);">
                                Error loading active LLM: ${error.message || 'Unknown error'}
                            </div>`;
                    }
                }
            }

            async function loadModelTab() {
                try {
                    const modelContent = document.getElementById('model-content');
                    if (!modelContent) {
                        console.error('Model content container not found');
                        return;
                    }

                    const response = await fetch('/admin/model/current');
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    } 
                    const data = await response.json();
                    
                    // Debug: Log the received data
                    console.log('Model data received:', data);
                    
                    // Model configuration section
                    let tabsHtml = `
                        <div style="margin-bottom: 20px;">
                            <h3 style="margin-top: 0; color: var(--primary);">Model Configuration</h3>
                            <p style="color: var(--text-muted); margin-bottom: 20px;">
                                Configure your language models and API keys below. Each model can be updated independently.
                            </p>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 20px; margin-bottom: 30px;">
                    `;
                    
                    // Model cards
                    const providers = [
                        { id: 'groq', name: 'Groq', defaultModel: 'llama3-70b-8192' },
                        { id: 'google', name: 'Google', defaultModel: 'gemma-3-12b-it' },
                        { id: 'openrouter', name: 'OpenRouter', defaultModel: 'openai/gpt-4-turpo' },
                        { id: 'openai', name: 'OpenAI', defaultModel: 'gpt-4o' }
                    ];

                    providers.forEach(provider => {
                        const modelKey = `${provider.id}_model`;
                        // Use lowercase for UI consistency, backend will handle the conversion
                        const apiKeyKey = `${provider.id.toLowerCase()}_api_key`;
                        
                        // Debug: Log the keys being accessed
                        console.log(`Processing provider ${provider.id}:`, {
                            modelKey,
                            modelValue: data[modelKey],
                            apiKeyKey,
                            apiKeyValue: data[apiKeyKey] ? '***' + data[apiKeyKey].slice(-4) : 'not found'
                        });
                        
                        const modelValue = data[modelKey] || provider.defaultModel;
                        const apiKeyValue = data[apiKeyKey] || '';
                        const isActive = data.provider === provider.id.toLowerCase();
                        
                        tabsHtml += `
                            <div class="model-card" style="
                                background: var(--bg-card);
                                border: 1.5px solid ${isActive ? 'var(--primary)' : 'var(--border)'};
                                border-radius: 12px;
                                padding: 22px;
                                box-shadow: 0 4px 20px rgba(0,0,0,0.12);
                                transition: all 0.25s ease;
                                position: relative;
                                overflow: hidden;
                                ${isActive ? 'border-left: 4px solid var(--primary);' : ''}
                            ">
                                ${isActive ? `
                                <div style="position: absolute; top: 0; right: 0; background: var(--primary); color: white; padding: 2px 12px; border-bottom-left-radius: 8px; font-size: 0.8em; font-weight: 600;">
                                    ACTIVE
                                </div>
                                ` : ''}
                                
                                <h3 style="margin: 0 0 15px 0; color: ${isActive ? 'var(--primary)' : 'var(--text)'}; font-size: 1.3em;">
                                    ${provider.name}
                                </h3>
                                
                                <div style="margin-bottom: 15px;">
                                    <label for="${provider.id}_model" style="display: block; margin-bottom: 6px; color: var(--text-muted); font-size: 0.9em; font-weight: 500;">Model Name</label>
                                    <input type="text" 
                                           id="${provider.id}_model" 
                                           value="${modelValue}" 
                                           class="env-input" 
                                           style="width: 100%; padding: 10px 12px; border-radius: 6px; background: rgba(255,255,255,0.03);">
                                </div>
                                
                                <div class="api-key-group" style="margin-bottom: 15px; position: relative;">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                        <label for="${provider.id}_api_key_input" style="color: var(--text-muted); font-size: 0.9em; font-weight: 500;">API Key</label>
                                        <div style="display: flex; gap: 8px;">
                                            <button type="button" 
                                                    onclick="toggleApiKeyVisibility(this)" 
                                                    class="toggle-api-key"
                                                    style="
                                                        background: transparent;
                                                        border: 1px solid var(--border);
                                                        color: var(--text-muted);
                                                        border-radius: 4px;
                                                        padding: 2px 8px;
                                                        font-size: 0.8em;
                                                        cursor: pointer;
                                                    ">
                                                Show
                                            </button>
                                            <button type="button" 
                                                    onclick="copyApiKey('${provider.id}_api_key_input')" 
                                                    style="
                                                        background: transparent;
                                                        border: 1px solid var(--border);
                                                        color: var(--text-muted);
                                                        border-radius: 4px;
                                                        padding: 2px 8px;
                                                        font-size: 0.8em;
                                                        cursor: pointer;
                                                    ">
                                                Copy 
                                            </button>
                                        </div>
                                    </div>
                                    <input type="password" 
                                           id="${provider.id}_api_key_input" 
                                           name="${provider.id}_api_key"
                                           value="${apiKeyValue}" 
                                           class="env-input api-key-input" 
                                           style="width: 100%; padding: 10px 12px; border-radius: 6px; background: rgba(255,255,255,0.03);"
                                           placeholder="${provider.name} API key"
                                           autocomplete="off">
                                </div>
                                
                                <div style="display: flex; gap: 10px; margin-top: 15px;">
                                    <button type="button"
                                            onclick="setActiveModel('${provider.id}')"
                                            style="
                                                flex: 1;
                                                padding: 10px;
                                                background: ${isActive ? 'var(--success)' : 'var(--primary)'};
                                                border: 1px solid ${isActive ? 'var(--success)' : 'var(--primary)'};
                                                color: white;
                                                font-weight: 500;
                                                border-radius: 6px;
                                                cursor: pointer;
                                                transition: all 0.2s;
                                            "
                                            onmouseover="this.style.opacity='0.9'"
                                            onmouseout="this.style.opacity='1'">
                                        ${isActive ? '✓ Active' : 'Set as Active'}
                                    </button>
                                    <button type="button"
                                            onclick="saveModelConfig('${provider.id}')"
                                            style="
                                                padding: 10px 15px;
                                                background: rgba(255,255,255,0.05);
                                                border: 1px solid var(--border);
                                                color: var(--text);
                                                border-radius: 6px;
                                                cursor: pointer;
                                                transition: all 0.2s;
                                            "
                                            onmouseover="this.style.opacity='0.9'"
                                            onmouseout="this.style.opacity='1'">
                                        Update
                                    </button>
                                </div>
                                <div id="${provider.id}-status" style="margin-top: 8px; font-size: 0.85em; min-height: 20px;"></div>
                            </div>
                        `;
                    });

                    // Close model cards grid
                    tabsHtml += `
                            </div>
                        </div>
                        <input type="hidden" id="model-priority-current" value="${data.MODEL_PRIORITY || '1,2,3'}">
                    `;

                    // Only update the content if we have a valid container
                    modelContent.innerHTML = tabsHtml;
                    
                } catch (e) {
                    console.error('Error loading model tab:', e);
                    const errorMessage = `Failed to load model info: ${e.message}`;
                    console.error(errorMessage);
                    
                    // Try to update the status element if it exists
                    const statusElement = document.getElementById('model-status');
                    if (statusElement) {
                        statusElement.innerHTML = `<span style="color:red;">${errorMessage}</span>`;
                    } else {
                        // If status element doesn't exist, log to console
                        console.error('Status element not found for error display');
                    }
                }
            }
            // Toggle API key visibility
            function toggleApiKeyVisibility(button) {
                try {
                    // Find the input element - it's the previous sibling of the button's parent
                    const inputGroup = button.closest('.api-key-group');
                    if (!inputGroup) {
                        console.error('API key group not found');
                        return;
                    }
                    const input = inputGroup.querySelector('input[type="password"], input[type="text"]');
                    if (input) {
                        if (input.type === 'password') {
                            input.type = 'text';
                            button.textContent = 'Hide';
                        } else {
                            input.type = 'password';
                            button.textContent = 'Show';
                        }
                    } else {
                        console.error('Could not find API key input field in group');
                    }
                } catch (error) {
                    console.error('Error in toggleApiKeyVisibility:', error);
                }
            }

            // Copy API key to clipboard
            async function copyApiKey(inputId) {
                const input = document.getElementById(inputId);
                try {
                    await navigator.clipboard.writeText(input.value);
                    const copyBtn = input.parentElement.querySelector('button[onclick^="copyApiKey"]');
                    const originalText = copyBtn.textContent;
                    copyBtn.textContent = 'Copied!';
                    copyBtn.style.borderColor = 'var(--success)';
                    copyBtn.style.color = 'var(--success)';
                    setTimeout(() => { 
                        copyBtn.textContent = originalText;
                        copyBtn.style.borderColor = 'var(--border)';
                        copyBtn.style.color = 'var(--text-muted)';
                    }, 2000);
                } catch (err) {
                    console.error('Failed to copy:', err);
                }
            }

            // Set the active model
            async function setActiveModel(provider) {
                const statusElement = document.getElementById(`${provider}-status`);
                const button = document.querySelector(`button[onclick^="setActiveModel('${provider}')"]`);
                
                if (!button) {
                    console.error('Active model button not found for provider:', provider);
                    return;
                }
                
                const originalButtonText = button.textContent;
                button.disabled = true;
                button.textContent = 'Activating...';
                if (statusElement) {
                    statusElement.innerHTML = '<span style="color: var(--text-muted);">Activating model...</span>';
                }
                
                try {
                    // First save the current configuration with isFromSetActive flag
                    await saveModelConfig(provider, false, true);
                    
                    // Then set as active
                    const response = await fetch('/admin/model/set-active', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ provider: provider })
                    });
                    
                    const result = await response.json();
                    
                    if (!response.ok) {
                        throw new Error(result.detail || 'Failed to set active model');
                    }
                    
                    // Update the UI to show the active status
                    const providerCards = document.querySelectorAll('.model-card');
                    providerCards.forEach(card => {
                        const cardProvider = card.querySelector('h3')?.textContent?.toLowerCase();
                        if (cardProvider === provider.toLowerCase()) {
                            card.style.borderLeft = '4px solid var(--primary)';
                            const activeBadge = card.querySelector('.active-badge');
                            if (activeBadge) {
                                activeBadge.style.display = 'block';
                            }
                        } else {
                            card.style.borderLeft = '1.5px solid var(--border)';
                            const otherBadge = card.querySelector('.active-badge');
                            if (otherBadge) {
                                otherBadge.style.display = 'none';
                            }
                        }
                    });
                    
                    // Update the active model display
                    await loadActiveLLM();
                    
                    // Show success notification
                    showNotification(
                        `${provider.charAt(0).toUpperCase() + provider.slice(1)} is now the active model`,
                        'success'
                    );
                    
                    // Update the button text
                    button.textContent = '✓ Active';
                    button.style.background = 'var(--success)';
                    button.style.borderColor = 'var(--success)';
                    
                } catch (error) {
                    console.error('Error setting active model:', error);
                    if (statusElement) {
                        statusElement.innerHTML = `<span style="color: var(--danger);">Error: ${error.message}</span>`;
                    }
                    showNotification(`Error: ${error.message}`, 'error');
                } finally {
                    button.disabled = false;
                    button.textContent = originalButtonText;
                }
            }
            
            // Save model configuration
            async function saveModelConfig(provider, showNotif = true, isFromSetActive = false) {
                const modelInput = document.getElementById(`${provider}_model`);
                const apiKeyInput = document.getElementById(`${provider}_api_key_input`);
                const statusElement = document.getElementById(`${provider}-status`);
                
                // Only get the save button, not the set active button
                const buttons = document.querySelectorAll(`button[onclick^="saveModelConfig('${provider}')"]`);
                const button = buttons[0]; // Use the first matching button
                
                if (!button) {
                    console.error('Save button not found for provider:', provider);
                    return;
                }
                
                // Update UI immediately for better UX
                const originalButtonText = button.textContent;
                button.disabled = true;
                button.textContent = 'Saving...';
                if (statusElement) {
                    statusElement.innerHTML = '';
                }
                
                try {
                    // Get current model values
                    const groqModel = provider === 'groq' ? modelInput.value : document.getElementById('groq_model')?.value || 'llama3-70b-8192';
                    const googleModel = provider === 'google' ? modelInput.value : document.getElementById('google_model')?.value || 'gemma-3-12b-it';
                    const openrouterModel = provider === 'openrouter' ? modelInput.value : document.getElementById('openrouter_model')?.value || 'openai/gpt-4-turpo';
                    
                    // Prepare payload for the model update
                    const payload = {
                        provider: provider,
                        groq_model: groqModel,
                        google_model: googleModel,
                        openrouter_model: openrouterModel,
                        embedding_provider: 'google',  // Default value, can be updated if needed
                        embedding_model_google: 'models/embedding-001',  // Default value
                        embedding_model_cohere: ''  // Default value
                    };
                    
                    // Update the model configuration
                    const updateResponse = await fetch('/admin/model/update', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    
                    const result = await updateResponse.json(); 
                    
                    if (!updateResponse.ok) {
                        throw new Error(result.detail || 'Failed to update model configuration');
                    }
                    
                    // Update the API key in the environment if provided
                    if (apiKeyInput && apiKeyInput.value) {
                        try {
                            const keyResponse = await fetch('/admin/env/update', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    // Convert to uppercase for backend compatibility
                                    key: `${provider.toUpperCase()}_API_KEY`,
                                    value: apiKeyInput.value,
                                    description: `${provider.charAt(0).toUpperCase() + provider.slice(1)} API Key`
                                })
                            });
                            
                            if (!keyResponse.ok) {
                                const keyResult = await keyResponse.json();
                                throw new Error(keyResult.detail || 'Failed to update API key');
                            }
                            
                            // Update the model name in environment
                            const modelKey = `${provider.toUpperCase()}_MODEL`;
                            const modelResponse = await fetch('/admin/env/update', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    key: modelKey,
                                    value: modelInput.value,
                                    description: `${provider.charAt(0).toUpperCase() + provider.slice(1)} Model`
                                })
                            });
                            
                            if (!modelResponse.ok) {
                                const modelResult = await modelResponse.json();
                                console.warn('Model name update warning:', modelResult.detail);
                            }
                            
                        } catch (keyError) {
                            console.warn('API key or model update warning:', keyError);
                            // Don't fail the entire operation if just the API key or model update fails
                            if (typeof window.showNotification === 'function') {
                                window.showNotification(`Warning: ${keyError.message}`, 'warning');
                            } else {
                                console.log('Notification system not available');
                            }
                        }
                    }
                    
                    // Only show success notification if not called from setActiveModel or if explicitly requested
                    if (showNotif && !isFromSetActive) {
                        if (typeof window.showNotification === 'function') {
                            window.showNotification(`${provider.charAt(0).toUpperCase() + provider.slice(1)} settings saved successfully!`, 'success');
                        } else {
                            console.log('Notification system not available');
                        }
                    }
                    
                } catch (error) {
                    console.error('Error saving model config:', error);
                    const errorMessage = error.message || 'An unknown error occurred';
                    statusElement.innerHTML = `<span style="color: var(--danger);">Error: ${errorMessage}</span>`;
                    showNotification(`Error: ${errorMessage}`, 'error');
                } finally {
                    button.disabled = false;
                    button.textContent = originalButtonText;
                    
                    // Reload the model tab to show updated status
                    loadModelTab();
                }
            }
            
            // Show notification
            function showNotification(message, type = 'info') {
                const container = document.getElementById('notification-container');
                if (!container) return;
                
                const notification = document.createElement('div');
                notification.className = `notification ${type}`;
                notification.style.cssText = `
                    background: var(--bg-card);
                    border-left: 4px solid ${type === 'error' ? 'var(--danger)' : 'var(--success)'};
                    border-radius: 4px;
                    padding: 12px 16px;
                    margin-bottom: 10px;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    max-width: 400px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    animation: slideIn 0.3s ease-out;
                `;
                
                notification.innerHTML = `
                    <div>${message}</div>
                    <button onclick="this.parentElement.remove()" style="
                        background: none;
                        border: none;
                        color: var(--text-muted);
                        cursor: pointer;
                        font-size: 18px;
                        margin-left: 10px;
                        padding: 0 5px;
                    ">&times;</button>
                `;
                
                container.appendChild(notification);
                
                // Auto-remove after 5 seconds
                setTimeout(() => { 
                    notification.style.animation = 'fadeOut 0.3s ease-out';
                    setTimeout(() => notification.remove(), 300);
                }, 5000);
            }
            
            // Add notification container if it doesn't exist
            if (!document.getElementById('notification-container')) {
                const container = document.createElement('div');
                container.id = 'notification-container';
                container.style.cssText = `
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    z-index: 1000;
                `;
                document.body.appendChild(container);
                
                // Add styles for notifications
                const style = document.createElement('style');
                style.textContent = `
                    @keyframes slideIn {
                        from { transform: translateX(100%); opacity: 0; }
                        to { transform: translateX(0); opacity: 1; }
                    }
                    @keyframes fadeOut {
                        from { opacity: 1; transform: translateX(0); }
                        to { opacity: 0; transform: translateX(100%); }
                    }
                    .notification {
                        transition: all 0.3s ease;
                    }
                `;
                document.head.appendChild(style);
            }
            
            // Save model configuration
            async function saveEmbeddingConfig() {
                const embedding_provider = document.getElementById('embedding_provider').value;
                const embedding_model_google = document.getElementById('embedding_model_google')?.value || '';
                const embedding_model_cohere = document.getElementById('embedding_model_cohere')?.value || '';
                
                const payload = {
                    provider: 'embedding',
                    embedding_provider,
                    embedding_model_google,
                    embedding_model_cohere
                };

                const statusElement = document.getElementById('embedding-status');
                statusElement.innerHTML = '<span style="color:orange;">Saving...</span>';
                
                try {
                    const response = await fetch('/admin/model/update', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const result = await response.json();
                    if (result.status === 'success') {
                        statusElement.innerHTML = '<span style="color:lightgreen;">' + result.message + '</span>';
                        await loadConfig();
                    } else {
                        statusElement.innerHTML = '<span style="color:red;">' + result.message + '</span>';
                    }
                } catch (e) {
                    statusElement.innerHTML = '<span style="color:red;">Failed to save: ' + e.message + '</span>';
                }
            }

            async function toggleEmbeddingModelInput() {
                const embeddingProvider = document.getElementById('embedding_provider').value;
                const embeddingModelInputs = document.getElementById('embedding-model-inputs');
                
                let html = '';
                if (embeddingProvider === 'google') {
                    html = `
                        <div style="margin-bottom: 15px;">
                            <label for="embedding_model_google">Google Embedding Model</label>
                            <input type="text" 
                                   id="embedding_model_google" 
                                   class="env-input" 
                                   placeholder="e.g., models/embedding-001"
                                   style="width: 100%;">
                        </div>
                    `;
                } else if (embeddingProvider === 'cohere') {
                    html = `
                        <div style="margin-bottom: 15px;">
                            <label for="embedding_model_cohere">Cohere Embedding Model</label>
                            <input type="text" 
                                   id="embedding_model_cohere" 
                                   class="env-input" 
                                   placeholder="e.g., embed-english-v3.0"
                                   style="width: 100%;">
                            <div style="margin-top: 10px; position: relative;">
                                <label for="cohere_api_key" style="display: block; margin-bottom: 5px;">Cohere API Key</label>
                                <div style="display: flex; gap: 5px;">
                                    <input type="password" 
                                           id="cohere_api_key" 
                                           class="env-input" 
                                           style="flex-grow: 1;"
                                           placeholder="Enter Cohere API key">
                                    <button class="button" 
                                            onclick="toggleApiKeyVisibility(this)" 
                                            style="min-width: 80px;">
                                        Show
                                    </button>
                                    <button class="button" 
                                            onclick="copyApiKey('cohere_api_key')" 
                                            style="min-width: 80px;">
                                        Copy
                                    </button>
                                </div>
                            </div>
                        </div>
                    `;
                }
                
                if (embeddingModelInputs) {
                    embeddingModelInputs.innerHTML = html;
                }
            }
            // Add CSS for the configuration tab
            const configStyles = document.createElement('style');
            configStyles.textContent = `
                .config-container {
                    max-width: 1000px;
                    margin: 0 auto;
                }
                .config-section {
                    background: #1e1e1e;
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 25px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                    border: 1px solid #2a2a2a;
                }
                .section-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                    padding-bottom: 12px;
                    border-bottom: 1px solid #2a2a2a;
                }
                .section-title {
                    font-size: 1.2em;
                    color: var(--primary);
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                .config-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                    gap: 20px;
                }
                .config-item {
                    background: #252525;
                    border-radius: 8px;
                    padding: 15px;
                    border: 1px solid #2f2f2f;
                }
                .config-label {
                    font-size: 0.85em;
                    color: #9e9e9e;
                    margin-bottom: 8px;
                }
                .config-input {
                    width: 100%;
                    padding: 10px 12px;
                    background: #1a1a1a;
                    border: 1px solid #333;
                    border-radius: 6px;
                    color: #e0e0e0;
                    font-family: 'Inter', sans-serif;
                    font-size: 0.95em;
                }
                .config-input:focus {
                    outline: none;
                    border-color: var(--primary);
                }
                .config-description {
                    font-size: 0.8em;
                    color: #777;
                    margin-top: 8px;
                }
                .config-actions {
                    margin-top: 15px;
                    text-align: right;
                }
                .empty-state {
                    text-align: center;
                    padding: 40px 20px;
                    background: #1e1e1e;
                    border-radius: 8px;
                    border: 1px dashed #333;
                }
                @media (max-width: 768px) {
                    .config-grid {
                        grid-template-columns: 1fr;
                    }
                }
            `;
            document.head.appendChild(configStyles);

            async function loadEnvConfig() {
                try {
                    const response = await fetch('/admin/env');
                    envData = await response.json();
                    const envContent = document.getElementById('env-content');
                    
                    if (Object.keys(envData.variables).length === 0) {
                        envContent.innerHTML = `
                            <div class="empty-state">
                                <h3>No Configuration Found</h3>
                                <p>The system is using default configuration values.</p>
                                <p>Create a .env file in the project root to customize settings.</p>
                            </div>
                        `;
                        return;
                    }
                    
                    // Filter out model-related environment variables
                    const modelRelatedKeys = [
                        'GROQ_API_KEY', 'GROQ_MODEL',
                        'GOOGLE_API_KEY', 'GOOGLE_MODEL',
                        'OPENROUTER_API_KEY', 'OPENROUTER_MODEL',
                        'MODEL_PRIORITY', 'TEMPERATURE',
                        'OPENAI_API_KEY', 'OPENAI_MODEL',

                    ];
                    
                    // Categorize variables
                    const categories = {
                        database: { title: 'Database', items: [] },
                        server: { title: 'Server', items: [] },
                        other: { title: 'Other', items: [] }
                    };
                    
                    let hasNonModelVars = false;
                    
                    for (const [key, value] of Object.entries(envData.variables)) {
                        if (modelRelatedKeys.includes(key)) continue;
                        
                        hasNonModelVars = true;
                        const description = envData.descriptions[key] || 'No description available';
                        const item = { key, value, description };
                        
                        // Categorize the variable
                        if (key.startsWith('QDRANT_') || key.includes('DATABASE') || key.includes('DB_')) {
                            categories.database.items.push(item);
                        } else if (key === 'PORT' || key === 'WORKERS' || key.includes('HOST') || key.includes('URL')) {
                            categories.server.items.push(item);
                        } else {
                            categories.other.items.push(item);
                        }
                    }
                    
                    let html = '<div class="config-container">';
                    
                    // Render each category
                    for (const [category, data] of Object.entries(categories)) {
                        if (data.items.length === 0) continue;
                        
                        html += `
                            <div class="config-section">
                                <div class="section-header">
                                    <h3 class="section-title">${data.title} Configuration</h3>
                                </div>
                                <div class="config-grid">
                        `;
                        
                        // Add config items
                        data.items.forEach(item => {
                            html += `
                                <div class="config-item">
                                    <div class="config-label">${item.key}</div>
                                    <input 
                                        type="text" 
                                        class="config-input" 
                                        id="env_${item.key}" 
                                        value="${item.value}"
                                        placeholder="${item.key}"
                                    >
                                    <div class="config-description">${item.description}</div>
                                    <div class="config-actions">
                                        <button class="button button-small" onclick="updateEnvVar('${item.key}')">
                                            Update
                                        </button>
                                    </div>
                                </div>
                            `;
                        });
                        
                        html += '</div></div>';
                    }
                    
                    if (!hasNonModelVars) {
                        html += `
                            <div class="empty-state">
                                <h3>No Configuration Variables</h3>
                                <p>All model-related configurations are managed in the <strong>Model</strong> tab.</p>
                            </div>
                        `;
                    }
                    
                    html += '</div>';
                    envContent.innerHTML = html;
                    
                } catch (error) {
                    document.getElementById('env-content').innerHTML = `
                        <div class="empty-state" style="border-color: #ff6b6b;">
                            <h3>Error Loading Configuration</h3>
                            <p>${error.message}</p>
                        </div>
                    `;
                }
            }
            async function updateEnvVar(key) {
                const value = document.getElementById(`env_${key}`).value;
                const response = await fetch('/admin/env/update', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({key: key, value: value})
                });
                const result = await response.json();
                alert(result.message);
            }
            async function reloadConfig() {
                const response = await fetch('/admin/env/reload', {method: 'POST'});
                const result = await response.json();
                alert(result.message);
            }
            async function invalidateCache() {
                const response = await fetch('/admin/cache/invalidate', {method: 'POST'});
                const result = await response.json();
                alert(result.message);
            }
            async function createBackup() {
                const response = await fetch('/admin/backup', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({include_vectors: true, include_uploads: true})
                });
                const result = await response.json();
                alert(result.message);
            }
            async function cleanupBackups() {
                const response = await fetch('/admin/backup/cleanup', {method: 'POST'});
                const result = await response.json();
                alert(result.message);
            }
            async function cleanupLogs() {
                const response = await fetch('/admin/logs/cleanup?days_to_keep=30', {method: 'POST'});
                const result = await response.json();
                alert(result.message);
            }
            async function validateConfig() {
                const response = await fetch('/admin/config/validate', {method: 'POST'});
                const result = await response.json();
                alert(result.message);
            }
            function loadUploadTab() {
                document.getElementById('upload-status').innerHTML = '';
                document.getElementById('ingest-status').innerHTML = '';
            }
            // Toast notification system
            function showToast(message, type = 'info') {
                const toastContainer = document.getElementById('toast-container');
                const toast = document.createElement('div');
                toast.textContent = message;
                toast.style.background = type === 'success' ? '#28a745' : (type === 'error' ? '#dc3545' : '#ff6600');
                toast.style.color = '#fff';
                toast.style.padding = '16px 28px';
                toast.style.marginTop = '12px';
                toast.style.borderRadius = '8px';
                toast.style.boxShadow = '0 2px 12px rgba(0,0,0,0.25)';
                toast.style.fontWeight = 'bold';
                toast.style.fontSize = '1.1em';
                toast.style.opacity = '0.97';
                toast.style.transition = 'opacity 0.4s';
                toastContainer.appendChild(toast);
                setTimeout(() => {
                    toast.style.opacity = '0';
                    setTimeout(() => toast.remove(), 400);
                }, 3000);
            }
            // Upload files (update to use toast)
            async function uploadFiles() {
                const input = document.getElementById('file-input');
                const files = input.files;
                if (!files.length) {
                    showToast('Please select at least one file to upload.', 'error');
                    return;
                }
                const formData = new FormData();
                for (let i = 0; i < files.length; i++) {
                    formData.append('files', files[i]);
                }
                // Optionally, allow folder_name input (default to Data)
                formData.append('folder_name', 'Data');
                document.getElementById('upload-status').innerHTML = 'Uploading...';
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    if (response.ok) {
                        showToast(result.message || 'Upload successful!', 'success');
                        document.getElementById('upload-status').innerHTML = '<span style="color:#28a745">' + (result.message || 'Upload successful!') + '</span>';
                    } else {
                        showToast(result.detail || 'Upload failed.', 'error');
                        document.getElementById('upload-status').innerHTML = '<span style="color:#dc3545">' + (result.detail || 'Upload failed.') + '</span>';
                    }
                } catch (e) {
                    showToast('Upload failed: ' + e, 'error');
                    document.getElementById('upload-status').innerHTML = '<span style="color:#dc3545">Upload failed: ' + e + '</span>';
                }
            }
            // Spinner HTML
            const spinnerHTML = `
              <div style="display:flex;align-items:center;justify-content:center;margin:20px 0;">
                <div class="spinner" style="width:36px;height:36px;border:4px solid #ff6600;border-top:4px solid #fff;border-radius:50%;animation:spin 1s linear infinite;"></div>
                <span style="margin-left:16px;font-size:1.1em;color:#ffb366;">Ingestion in progress...</span>
              </div>
            `;
            // Add spinner CSS
            (function(){
              const style = document.createElement('style');
              style.innerHTML = `@keyframes spin { 0% { transform: rotate(0deg);} 100% { transform: rotate(360deg);} }`;
              document.head.appendChild(style);
            })();
            // Ingest now (improved UX)
            async function ingestNow() {
                const ingestStatus = document.getElementById('ingest-status');
                ingestStatus.innerHTML = spinnerHTML;
                try {
                    const response = await fetch('/ingest', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ directory_path: '' })
                    });
                    const result = await response.json();
                    ingestStatus.innerHTML = '';
                    if (response.ok) {
                        showToast('Ingestion complete! ' + (result.chunks_ingested || 0) + ' chunks processed.', 'success');
                    } else {
                        showToast(result.detail || 'Ingestion failed.', 'error');
                    }
                } catch (e) {
                    ingestStatus.innerHTML = '';
                    showToast('Ingestion failed: ' + e, 'error');
                }
            }
            // Fetch and display collection stats
            async function fetchCollectionStats() {
                const statsDiv = document.getElementById('collection-stats-content');
                statsDiv.innerHTML = spinnerHTML;
                try {
                    const res = await fetch('/collections/stats');
                    const data = await res.json();
                    if (res.ok && data.collections && data.collections.length) {
                        let html = `<table style="width:100%;background:#222;border-radius:6px;overflow:hidden;">
                            <tr style="background:#333;"><th style="padding:8px 12px;text-align:left;">Collection</th><th style="padding:8px 12px;text-align:left;">Chunks</th></tr>`;
                        for (const c of data.collections) {
                            html += `<tr><td style="padding:8px 12px;">${c.name}</td><td style="padding:8px 12px;">${c.chunks}</td></tr>`;
                        }
                        html += '</table>';
                        statsDiv.innerHTML = html;
                    } else {
                        statsDiv.innerHTML = '<span style="color:#dc3545">No collections found.</span>';
                    }
                } catch (e) {
                    statsDiv.innerHTML = '<span style="color:#dc3545">Failed to load stats.</span>';
                }
            }
            // Call on load
            fetchCollectionStats();
            // Also refresh after upload/ingest/collection action
            function refreshStatsAfterAction() {
                setTimeout(fetchCollectionStats, 1000);
            }
            // Collection management logic
            async function handleCollectionAction() {
                const action = document.getElementById('collection-action').value;
                const name = document.getElementById('collection-name').value.trim();
                if (!name) {
                    showToast('Please enter a collection name.', 'error');
                    return;
                }
                if (action === 'delete') {
                    if (!confirm(`Are you sure you want to DELETE the collection '${name}'? This cannot be undone!`)) {
                        return;
                    }
                }
                const formData = new FormData();
                formData.append('action', action);
                // For now, backend uses config.database.collection_name, so warn if name != current
                // (Optionally, you can update backend to accept collection name as param)
                showToast((action === 'delete' ? 'Deleting' : 'Creating') + ` collection '${name}'...`, 'info');
                try {
                    const res = await fetch('/collection', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await res.json();
                    if (res.ok) {
                        showToast(data.status || 'Action complete.', 'success');
                        refreshStatsAfterAction();
                    } else {
                        showToast(data.detail || 'Action failed.', 'error');
                    }
                } catch (e) {
                    showToast('Action failed: ' + e, 'error');
                }
            }
            loadStatus(); 
            loadConfig();
            loadActiveLLM();
            setInterval(() => {
                if (document.getElementById('status').classList.contains('active')) {
                    loadStatus();
                    loadConfig();
                    loadActiveLLM();
                }
            }, 30000);
        </script>
    </body>
    </html>
    
""" 

@router.get("/model/current")
async def get_current_model_config():
    try:
        cfg = get_config()
        
        # Get the active LLM configuration with error handling
        llm_config = cfg.get_active_llm_config()
        if not isinstance(llm_config, dict):
            llm_config = {}
            
        # Get API keys from environment
        env_vars = admin_dashboard.read_env_file()
        
        # Get model priority with fallback
        model_priority = getattr(cfg.llm, 'model_priority', '1,2,3')
        
        # Build the response with fallback values
        response = {
            "provider": llm_config.get("provider", ""),
            "priority": model_priority,
            "groq_model": getattr(cfg.llm, 'groq_model', 'llama3-70b-8192'),
            "google_model": getattr(cfg.llm, 'google_model', 'gemma-3-12b-it'),
            "openrouter_model": getattr(cfg.llm, 'openrouter_model', 'openai/gpt-4-turpo'),
            "groq_api_key": env_vars.get("GROQ_API_KEY", ""),
            "google_api_key": env_vars.get("GOOGLE_API_KEY", ""),
            "openrouter_api_key": env_vars.get("OPENROUTER_API_KEY", ""),
            "status": "success"
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error in get_current_model_config: {str(e)}")
        # Return a safe default configuration if there's an error
        return {
            "provider": "error",
            "priority": "1,2,3",
            "groq_model": "llama3-70b-8192",
            "google_model": "gemma-3-12b-it",
            "openrouter_model": "openai/gpt-4-turpo",
            "groq_api_key": "",
            "google_api_key": "",
            "openrouter_api_key": "",
            "status": "error",
            "error": str(e)
        }

@router.post("/model/update")
async def update_model_config(request: ModelUpdateRequest = Body(...)):
    provider_map = {"openrouter": 1, "google": 2, "groq": 3 , "openai": 4}
    new_priority = provider_map.get(request.provider.lower(), 1)
    env_vars = admin_dashboard.read_env_file()
    
    # Update model priority and model settings
    env_vars["MODEL_PRIORITY"] = str(new_priority)
    env_vars["GROQ_MODEL"] = request.groq_model
    env_vars["GOOGLE_MODEL"] = request.google_model
    env_vars["OPENROUTER_MODEL"] = request.openrouter_model
    env_vars["OPENAI_MODEL"] = request.openai_model
    
    # Save and reload environment
    admin_dashboard.write_env_file(env_vars)
    load_dotenv(override=True)
    
    try:
        from ingestion_retrieval.retrieval import invalidate_cache
        invalidate_cache()
        return {
            "status": "success", 
            "message": "Model and embedding settings updated. New settings will be used for the next request."
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to update config: {str(e)}"} 

@router.get("/llm/active")
async def get_active_llm():
    try:
        cfg = get_config()
        active_llm = cfg.get_active_llm_config()
        
        # Ensure we always return a consistent structure
        provider = active_llm.get("provider", "unknown")
        model = active_llm.get("model", "")
        
        # If model is empty but we have provider-specific model settings
        if not model and provider == "groq":
            model = cfg.llm.groq_model
        elif not model and provider == "google":
            model = cfg.llm.google_model
        elif not model and provider == "openrouter":
            model = cfg.llm.openrouter_model
            
        return {
            "provider": provider,
            "model": model or "default",
            "is_initialized": active_llm.get("is_initialized", False)
        }
    except Exception as e:
        logger.error(f"Error getting active LLM: {str(e)}")
        return {
            "provider": "error",
            "model": "unknown",
            "error": str(e)
        }

@router.post("/model/set-active")
async def set_active_model(request: dict = Body(...)):
    try:
        provider = request.get('provider')
        if not provider:
            raise HTTPException(status_code=400, detail="Provider is required")
        
        # Update the model priority based on the provider
        provider_map = {"openrouter": 1, "google": 2, "groq": 3, "openai": 4}
        if provider.lower() not in provider_map:
            raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}")
            
        # Update the environment variables
        env_vars = admin_dashboard.read_env_file()
        env_vars["MODEL_PRIORITY"] = str(provider_map[provider.lower()])
        admin_dashboard.write_env_file(env_vars)
        
        # Reload the environment
        load_dotenv(override=True)
        
        # Invalidate the LLM cache to ensure the new model is used
        try:
            from ingestion_retrieval.retrieval import invalidate_cache
            invalidate_cache()
            
            # Get the current config to return the active model info
            cfg = get_config()
            llm_config = cfg.get_active_llm_config()
            
            return {
                "status": "success",
                "message": f"Successfully activated {provider} model",
                "provider": provider,
                "model": llm_config.get('model_name', 'unknown')
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to update model cache: {str(e)}"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to set active model: {str(e)}"
        ) 