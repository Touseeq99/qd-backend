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
    last_backup: str
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
            "OPENAI_API_KEY": "OpenAI API key for LLM (optional)",
            "MODEL_PRIORITY": "LLM priority: 1=Groq, 2=Google, 3=OpenAI",
            "GROQ_MODEL": "Groq model name (e.g., llama3-70b-8192)",
            "GOOGLE_MODEL": "Google model name (e.g., gemma-3-12b-it)",
            "OPENAI_MODEL": "OpenAI model name (e.g., gpt-4)",
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
                last_backup=self._get_last_backup_time(),
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
                last_backup="unknown",
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
        return {
            "database": {
                "url": self.config.database.url,
                "collection": self.config.database.collection_name,
                "vector_size": self.config.database.vector_size
            },
            "llm": {
                "active_provider": self.config.get_active_llm_config()["provider"],
                "model": self.config.get_active_llm_config()["model"],
                "temperature": self.config.llm.temperature
            },
            "embedding": {
                "model": self.config.embedding.google_model
            },
            "server": {
                "port": self.config.server.port,
                "workers": self.config.server.workers,
                "log_level": self.config.server.log_level
            },
            "file_upload": {
                "max_size_mb": self.config.file_upload.max_file_size_mb,
                "allowed_extensions": self.config.file_upload.allowed_extensions
            },
            "backup": {
                "enabled": self.config.backup.auto_backup_enabled,
                "frequency_hours": self.config.backup.backup_frequency_hours,
                "retention_days": self.config.backup.backup_retention_days
            }
        }

class ModelUpdateRequest(BaseModel):
    provider: str
    groq_model: str
    google_model: str
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
                border-bottom: 2px solid var(--border);
                margin-bottom: 34px;
            }
            .tab-button {
                background: var(--bg-card);
                border: 1.5px solid var(--border);
                border-bottom: none;
                padding: 15px 36px 13px 36px;
                cursor: pointer;
                color: var(--text-muted);
                font-size: 1.13em;
                border-radius: 10px 10px 0 0;
                margin-right: 4px;
                margin-bottom: -2px;
                transition: background 0.18s, color 0.18s, border 0.18s;
                font-weight: 600;
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
                <h2>Model Management</h2>
                <div id="model-content">
                    <form id="model-form" onsubmit="event.preventDefault(); saveModelConfig();">
                        <label for="provider">LLM Provider:</label>
                        <select id="provider">
                            <option value="groq">Groq</option>
                            <option value="google">Google</option>
                        </select><br>
                        <label for="groq_model">Groq Model Name:</label>
                        <input type="text" id="groq_model" placeholder="llama3-70b-8192"><br>
                        <label for="google_model">Google Model Name:</label>
                        <input type="text" id="google_model" placeholder="gemma-3-12b-it"><br>
                        <label for="embedding_provider">Embedding Provider:</label>
                        <select id="embedding_provider" onchange="toggleEmbeddingModelInput()">
                            <option value="google">Google</option>
                            <option value="cohere">Cohere</option>
                        </select><br>
                        <div id="embedding-model-inputs">
                            <label for="embedding_model_google">Google Embedding Model Name:</label>
                            <input type="text" id="embedding_model_google" placeholder="models/embedding-001"><br>
                            <label for="embedding_model_cohere" style="display:none;">Cohere Embedding Model Name:</label>
                            <input type="text" id="embedding_model_cohere" placeholder="embed-english-v3.0" style="display:none;"><br>
                        </div>
                        <button class="button" type="submit">Save Model Settings</button>
                    </form>
                    <div id="model-status"></div>
                    <div style="margin-top: 18px;">
                        <strong>Current Active Model:</strong> <span id="current-model"></span>
                    </div>
                </div>
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
            let envData = {};
            let configData = {};
            function showTab(tabName) {
                document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
                document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
                document.getElementById(tabName).classList.add('active');
                event.target.classList.add('active');
                if (tabName === 'config') loadEnvConfig();
                if (tabName === 'model') loadModelTab();
                if (tabName === 'upload') loadUploadTab();
            }
            async function loadStatus() {
                const response = await fetch('/admin/status');
                const status = await response.json();
                document.getElementById('status-content').innerHTML = `
                    <div class="metric">Status: <strong>${status.status}</strong></div>
                    <div class="metric">Uptime: ${status.uptime}</div>
                    <div class="metric">Memory: ${status.memory_usage.percent}%</div>
                    <div class="metric">Disk: ${status.disk_usage.percent}%</div>
                    <div class="metric">Last Backup: ${status.last_backup}</div>
                `;
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
                    html += '<strong>Embedding Model:</strong> ' + configData.embedding.model;
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
                const response = await fetch('/admin/llm/active');
                const data = await response.json();
                let html = '<b>Currently Active LLM in Memory:</b> ';
                if (data.provider) {
                    html += data.provider + ' - ' + data.model;
                } else {
                    html += '<span style="color:#bbb;">(No LLM cached yet)</span>';
                }
                document.getElementById('active-llm-content').innerHTML = html;
            }
            async function loadModelTab() {
                // Fetch current model/provider/embedding info from backend
                try {
                    const response = await fetch('/admin/model/current');
                    const data = await response.json();
                    document.getElementById('provider').value = data.provider;
                    document.getElementById('groq_model').value = data.groq_model;
                    document.getElementById('google_model').value = data.google_model;
                    document.getElementById('embedding_provider').value = data.embedding_provider || 'google';
                    toggleEmbeddingModelInput();
                    document.getElementById('embedding_model_google').value = data.embedding_model_google || '';
                    document.getElementById('embedding_model_cohere').value = data.embedding_model_cohere || '';
                    document.getElementById('current-model').innerText = data.provider + ' - ' + (data.provider === 'groq' ? data.groq_model : data.google_model);
                } catch (e) {
                    document.getElementById('model-status').innerHTML = '<span style="color:red;">Failed to load model info</span>';
                }
            }
            async function saveModelConfig() {
                const provider = document.getElementById('provider').value;
                const groq_model = document.getElementById('groq_model').value;
                const google_model = document.getElementById('google_model').value;
                const embedding_provider = document.getElementById('embedding_provider').value;
                let embedding_model_google = document.getElementById('embedding_model_google').value;
                let embedding_model_cohere = document.getElementById('embedding_model_cohere').value;
                const payload = { provider, groq_model, google_model, embedding_provider, embedding_model_google, embedding_model_cohere };
                document.getElementById('model-status').innerHTML = '<span style="color:orange;">Saving...</span>';
                try {
                    const response = await fetch('/admin/model/update', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const result = await response.json();
                    if (result.status === 'success') {
                        document.getElementById('model-status').innerHTML = '<span style="color:lightgreen;">' + result.message + '</span>';
                        await loadModelTab();
                        await loadConfig();
                    } else {
                        document.getElementById('model-status').innerHTML = '<span style="color:red;">' + result.message + '</span>';
                    }
                } catch (e) {
                    document.getElementById('model-status').innerHTML = '<span style="color:red;">Failed to save</span>';
                }
            }
            async function toggleEmbeddingModelInput() {
                const embeddingProvider = document.getElementById('embedding_provider').value;
                const embeddingModelInputs = document.getElementById('embedding-model-inputs');
                const googleModelInput = document.getElementById('embedding_model_google');
                const cohereModelInput = document.getElementById('embedding_model_cohere');

                if (embeddingProvider === 'google') {
                    googleModelInput.style.display = 'block';
                    cohereModelInput.style.display = 'none';
                    googleModelInput.setAttribute('required', 'required');
                    cohereModelInput.removeAttribute('required');
                } else if (embeddingProvider === 'cohere') {
                    googleModelInput.style.display = 'none';
                    cohereModelInput.style.display = 'block';
                    googleModelInput.removeAttribute('required');
                    cohereModelInput.setAttribute('required', 'required');
                }
            }
            async function loadEnvConfig() {
                try {
                    const response = await fetch('/admin/env');
                    envData = await response.json();
                    if (Object.keys(envData.variables).length === 0) {
                        document.getElementById('env-content').innerHTML = `
                            <div class="env-section">
                                <p><strong>No .env file found.</strong> The system is using default configuration values.</p>
                                <p>To customize settings, create a .env file in the project root with the following variables:</p>
                                <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; font-family: monospace; font-size: 0.9em;">
                                    <div># LLM Configuration</div>
                                    <div>GROQ_API_KEY=your_api_key_here</div>
                                    <div>GROQ_MODEL=llama3-70b-8192</div>
                                    <div>GOOGLE_API_KEY=your_api_key_here</div>
                                    <div>GOOGLE_MODEL=gemma-3-12b-it</div>
                                    <div>MODEL_PRIORITY=1</div>
                                    <div>TEMPERATURE=0.1</div>
                                    <div><br></div>
                                    <div># Database Configuration</div>
                                    <div>QDRANT_URL=http://localhost:6333</div>
                                    <div>QDRANT_COLLECTION=hr_documents</div>
                                    <div><br></div>
                                    <div># Embedding Configuration</div>
                                    <div>EMBEDDING_MODEL=models/embedding-001</div>
                                    <div><br></div>
                                    <div># Server Configuration</div>
                                    <div>PORT=8000</div>
                                    <div>WORKERS=4</div>
                                </div>
                                <p><strong>Current active configuration is shown in the System Status tab.</strong></p>
                            </div>
                        `;
                        return;
                    }
                    let html = '<div class="env-section">';
                    for (const [key, value] of Object.entries(envData.variables)) {
                        const description = envData.descriptions[key] || 'No description available';
                        html += `
                            <div class="env-description">${description}</div>
                            <input type="text" class="env-input" id="env_${key}" value="${value}" placeholder="${key}">
                            <button class="button button-success" onclick="updateEnvVar('${key}')">Update</button>
                            <br><br>
                        `;
                    }
                    html += '</div>';
                    document.getElementById('env-content').innerHTML = html;
                } catch (error) {
                    document.getElementById('env-content').innerHTML = '<p style="color: red;">Error loading environment configuration: ' + error.message + '</p>';
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
    cfg = get_config()
    # Determine embedding provider and model
    embedding_provider = "google"
    embedding_model_google = cfg.embedding.google_model
    embedding_model_cohere = cfg.embedding.cohere_model if hasattr(cfg.embedding, 'cohere_model') else ""
    # If cohere_model is set and not default, prefer cohere
    if embedding_model_cohere and embedding_model_cohere != "embed-english-v3.0":
        embedding_provider = "cohere"
    return {
        "provider": cfg.get_active_llm_config()["provider"],
        "priority": cfg.llm.model_priority,
        "groq_model": cfg.llm.groq_model,
        "google_model": cfg.llm.google_model,
        "embedding_provider": embedding_provider,
        "embedding_model_google": embedding_model_google,
        "embedding_model_cohere": embedding_model_cohere
    }

@router.post("/model/update")
async def update_model_config(request: ModelUpdateRequest = Body(...)):
    provider_map = {"groq": 1, "google": 2}
    new_priority = provider_map.get(request.provider.lower(), 1)
    env_vars = admin_dashboard.read_env_file()
    env_vars["MODEL_PRIORITY"] = str(new_priority)
    env_vars["GROQ_MODEL"] = request.groq_model
    env_vars["GOOGLE_MODEL"] = request.google_model
    # Embedding provider/model logic
    if request.embedding_provider == "google":
        env_vars["EMBEDDING_MODEL"] = request.embedding_model_google
        env_vars["COHERE_EMBEDDING_MODEL"] = ""
    elif request.embedding_provider == "cohere":
        env_vars["EMBEDDING_MODEL"] = ""
        env_vars["COHERE_EMBEDDING_MODEL"] = request.embedding_model_cohere
    admin_dashboard.write_env_file(env_vars)
    load_dotenv(override=True)
    try:
        from ingestion_retrieval.retrieval import invalidate_cache
        invalidate_cache()
        return {"status": "success", "message": "Model and embedding settings updated. New settings will be used for the next request."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to update config: {str(e)}"} 

@router.get("/llm/active")
async def get_active_llm():
    try:
        from ingestion_retrieval import retrieval
        llm = getattr(retrieval, '_llm_cache', None)
        if llm is None:
            return {"provider": None, "model": None}
        # Try to get provider/model info from the LLM object
        provider = getattr(llm, '__class__', type(llm)).__name__
        model = getattr(llm, 'model', None) or getattr(llm, '_model', None)
        return {"provider": provider, "model": model}
    except Exception as e:
        return {"provider": None, "model": None, "error": str(e)} 