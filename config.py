"""
Centralized Configuration Management for HR Assistant
This file makes it easy for non-technical teams to modify application settings.
"""

import os
from typing import List, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv(override=True)

def get_env_value(key: str, default: str) -> str:  
    """Get environment variable value, stripping comments and whitespace"""
    load_dotenv(override=True)
    value = os.getenv(key, default)
    if value is not None:
        # Remove comments (everything after #)
        value = value.split('#')[0]
        # Strip whitespace
        value = value.strip()
    return value 

@dataclass 
class DatabaseConfig:
    """Qdrant database configuration"""
    url: str = field(default_factory=lambda: get_env_value("QDRANT_URL", "http://localhost:6333"))
    api_key: Optional[str] = field(default_factory=lambda: get_env_value("QDRANT_API_KEY", ""))
    collection_name: str = field(default_factory=lambda: get_env_value("QDRANT_COLLECTION", "hr_documents"))
    vector_size: int = field(default_factory=lambda: int(get_env_value("VECTOR_SIZE", "768")))

@dataclass
class LLMConfig:
    """Language model configuration"""
    # Primary LLM (OpenRouter)
    openrouter_api_key: Optional[str] = field(default_factory=lambda: get_env_value("OPENROUTER_API_KEY", ""))
    openrouter_model: str = field(default_factory=lambda: get_env_value("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324:free"))
    
    # Fallback LLM (Google)
    google_api_key: Optional[str] = field(default_factory=lambda: get_env_value("GOOGLE_API_KEY", ""))
    google_model: str = field(default_factory=lambda: get_env_value("GOOGLE_MODEL", "gemma-3-12b-it"))
    
    # Legacy LLM (Groq) - Kept for backward compatibility
    groq_api_key: Optional[str] = field(default_factory=lambda: get_env_value("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: get_env_value("GROQ_MODEL", "llama3-70b-8192"))

    openai_api_key: Optional[str] = field(default_factory=lambda: get_env_value("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: get_env_value("OPENAI_MODEL", "gpt-4o"))

    # Model selection priority (1=OpenRouter, 2=Google, 3=Groq)
    model_priority: int = field(default_factory=lambda: int(get_env_value("MODEL_PRIORITY", "1")))
    temperature: float = field(default_factory=lambda: float(get_env_value("TEMPERATURE", "0.8")))

@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    # Primary embedding (SentenceTransformer)
    sentence_transformer_model: str = field(
        default_factory=lambda: get_env_value(
            "SENTENCE_TRANSFORMER_MODEL", 
            "BAAI/bge-base-en-v1.5"
        )
    )


@dataclass 
class ServerConfig: 
    """Server configuration"""
    port: int = field(default_factory=lambda: int(get_env_value("PORT", "8000")))
    workers: int = field(default_factory=lambda: int(get_env_value("WORKERS", "4"))) 
    timeout: int = field(default_factory=lambda: int(get_env_value("TIMEOUT", "120")))
    keep_alive: int = field(default_factory=lambda: int(get_env_value("KEEP_ALIVE", "30")))
    log_level: str = field(default_factory=lambda: get_env_value("LOG_LEVEL", "INFO"))

@dataclass
class FileUploadConfig:
    """File upload configuration"""
    max_file_size_mb: int = field(default_factory=lambda: int(get_env_value("MAX_FILE_SIZE_MB", "200")))
    allowed_extensions: List[str] = field(default_factory=lambda: get_env_value("ALLOWED_FILE_EXTENSIONS", ".pdf,.docx,.doc,.txt,.md").split(","))
    upload_dir: str = field(default_factory=lambda: get_env_value("UPLOAD_DIR", "uploaded_folders"))

@dataclass
class VectorSearchConfig:
    """Vector search configuration"""
    chunk_size: int = field(default_factory=lambda: int(get_env_value("CHUNK_SIZE", "600")))
    chunk_overlap: int = field(default_factory=lambda: int(get_env_value("CHUNK_OVERLAP", "150")))
    search_k: int = field(default_factory=lambda: int(get_env_value("SEARCH_K", "23")))
    fetch_k: int = field(default_factory=lambda: int(get_env_value("FETCH_K", "10")))
    score_threshold: float = field(default_factory=lambda: float(get_env_value("SCORE_THRESHOLD", "0.3")))
    # HNSW optimization parameters
    hnsw_ef: int = field(default_factory=lambda: int(get_env_value("HNSW_EF", "32")))
    # Caching parameters
    cache_size: int = field(default_factory=lambda: int(get_env_value("CACHE_SIZE", "1024")))  
 
@dataclass
class SecurityConfig:
    """Security configuration"""
    allowed_origins: List[str] = field(default_factory=lambda: get_env_value("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8080,http://localhost:8000,http://127.0.0.1:8000,http://127.0.0.1:3000,http://127.0.0.1:8080,*").split(","))
    rate_limit_per_minute: int = field(default_factory=lambda: int(get_env_value("RATE_LIMIT_PER_MINUTE", "60")))
    enable_cors: bool = field(default_factory=lambda: get_env_value("ENABLE_CORS", "true").lower() == "true")

@dataclass
class MonitoringConfig:
    """Monitoring and analytics configuration"""
    enable_request_logging: bool = field(default_factory=lambda: get_env_value("ENABLE_REQUEST_LOGGING", "true").lower() == "true")
    enable_performance_monitoring: bool = field(default_factory=lambda: get_env_value("ENABLE_PERFORMANCE_MONITORING", "true").lower() == "true")
    log_file_path: str = field(default_factory=lambda: get_env_value("LOG_FILE_PATH", "logs/hr_assistant.log"))

@dataclass
class BackupConfig:
    """Backup configuration"""
    auto_backup_enabled: bool = field(default_factory=lambda: get_env_value("AUTO_BACKUP_ENABLED", "true").lower() == "true")
    backup_frequency_hours: int = field(default_factory=lambda: int(get_env_value("BACKUP_FREQUENCY_HOURS", "24")))
    backup_retention_days: int = field(default_factory=lambda: int(get_env_value("BACKUP_RETENTION_DAYS", "30")))
    backup_dir: str = field(default_factory=lambda: get_env_value("BACKUP_DIR", "backups"))

class Config:
    """Main configuration class that combines all settings"""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.llm = LLMConfig()
        self.embedding = EmbeddingConfig()
        self.server = ServerConfig()
        self.file_upload = FileUploadConfig()
        self.vector_search = VectorSearchConfig() 
        self.security = SecurityConfig()
        self.monitoring = MonitoringConfig()
        self.backup = BackupConfig()
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Check required API keys based on model priority
        if self.llm.model_priority == 1 and not self.llm.groq_api_key:
            errors.append("GROQ_API_KEY is required when MODEL_PRIORITY=1")
        elif self.llm.model_priority == 2 and not self.llm.google_api_key:
            errors.append("GOOGLE_API_KEY is required when MODEL_PRIORITY=2")
        
        
        # Check database URL
        if not self.database.url:
            errors.append("QDRANT_URL is required")
        
        return errors
    
    def get_active_llm_config(self) -> dict:
        logging.info(
            f"[LLM CONFIG] model_priority={self.llm.model_priority}, "
            f"openrouter_api_key={'set' if self.llm.openrouter_api_key else 'unset'}, "
            f"google_api_key={'set' if self.llm.google_api_key else 'unset'}, "
            f"groq_api_key={'set' if self.llm.groq_api_key else 'unset'}"
        )
        
        # Priority 1: OpenRouter
        if self.llm.model_priority == 1 and self.llm.openrouter_api_key:
            logging.info("[LLM CONFIG] Selecting OpenRouter as provider")
            return {
                "provider": "openrouter",
                "api_key": self.llm.openrouter_api_key,
                "model": self.llm.openrouter_model,
            }
        # Priority 2: Google
        elif self.llm.model_priority == 2 and self.llm.google_api_key:
            logging.info("[LLM CONFIG] Selecting Google as provider")
            return {
                "provider": "google",
                "api_key": self.llm.google_api_key,
                "model": self.llm.google_model,
            }
        # Priority 3: Groq (legacy)
        elif self.llm.model_priority == 3 and self.llm.groq_api_key:
            logging.info("[LLM CONFIG] Selecting Groq as provider")
            return {
                "provider": "groq",
                "api_key": self.llm.groq_api_key,
                "model": self.llm.groq_model,
            }
        elif self.llm.model_priority == 4 and self.llm.openai_api_key:
            logging.info("[LLM CONFIG] Selecting OpenAI as provider")
            return {
                "provider": "openai",
                "api_key": self.llm.openai_api_key,
                "model": self.llm.openai_model,            }
        
        # Fallback to next available provider based on priority
        fallback_order = [
            ("openrouter", self.llm.openrouter_api_key, self.llm.openrouter_model, "OpenRouter"),
            ("google", self.llm.google_api_key, self.llm.google_model, "Google"),
            ("groq", self.llm.groq_api_key, self.llm.groq_model, "Groq")
        ]
        
        for provider, api_key, model, provider_name in fallback_order:
            if api_key:
                logging.info(f"[LLM CONFIG] Fallback: Selecting {provider_name} as provider")
                config = {
                    "provider": provider,
                    "api_key": api_key,
                    "model": model,
                }
                if provider == "openrouter":
                    config.update({
                        "site_url": self.llm.openrouter_site_url,
                        "site_name": self.llm.openrouter_site_name
                    })
                return config
        
        logging.error("[LLM CONFIG] No valid LLM configuration found!")
        raise ValueError("No valid LLM configuration found. Please check your API keys and configuration.")

# Remove the global config instance
# config = Config()

def get_config() -> Config:
    """Get a fresh configuration instance (always reloads .env)"""
    return Config()

def validate_config() -> None:
    """Validate configuration and raise errors if invalid"""
    errors = get_config().validate()
    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}") 