# HR Assistant - Complete Maintenance Guide

## 📋 Table of Contents
1. [System Architecture](#system-architecture)
2. [How Everything Works](#how-everything-works)
3. [Frontend Admin Dashboard](#frontend-admin-dashboard)
4. [Configuration Management](#configuration-management)
5. [Docker Management](#docker-management)
6. [Troubleshooting](#troubleshooting)
7. [Performance Optimization](#performance-optimization)
8. [Backup & Recovery](#backup--recovery)

---

## 🏗️ System Architecture

### **Core Components:**
```
HR Assistant
├── FastAPI Backend (main.py)
├── Admin Interface (admin_interface.py)
├── Configuration Management (config.py)
├── Document Processing (ingestion_retrieval/)
├── Vector Database (Qdrant)
├── LLM Providers (Groq/Google)
└── Docker Container
```

### **Technology Stack:**
- **Backend:** FastAPI + Gunicorn
- **Vector Database:** Qdrant
- **LLM:** Groq (primary), Google (fallback)
- **Embeddings:** Google/Cohere
- **Container:** Docker + Docker Compose
- **Admin UI:** HTML + JavaScript

---

## ⚙️ How Everything Works

### **1. Request Flow:**
```
User Question → FastAPI → Vector Search → LLM → Response
     ↓              ↓           ↓         ↓        ↓
   Frontend    Authentication  Qdrant   Groq/   Formatted
   Interface                   Search   Google   Answer
```

### **2. Document Processing:**
```
Upload → Chunking → Embedding → Vector Storage → Retrieval
   ↓        ↓         ↓           ↓            ↓
Files  512 chars   Google/    Qdrant DB   Similarity
       overlap     Cohere     Collection   Search
```

### **3. Caching Strategy:**
- **LLM Cache:** Global singleton for model instances
- **Vector Store:** Persistent Qdrant client connections
- **Chain Cache:** LRU cache for retrieval chains
- **Thread Safety:** Locked cache operations

### **4. Worker Configuration:**
```python
# Current: 17 workers, 2 threads each = 34 concurrent handlers
workers = 17
threads = 2
worker_connections = 1000
backlog = 2048
```

---

## 🖥️ Frontend Admin Dashboard

### **Access URL:**
```
http://localhost:8000/admin/dashboard
```

### **Dashboard Tabs:**

#### **1. Upload & Ingest Tab**
- **File Upload:** Drag & drop or select files/folders
- **Ingest Documents:** Process uploaded files into vector database
- **Collection Stats:** View document counts and storage info
- **Collection Management:** Create/delete vector collections

#### **2. System Status Tab**
- **Real-time Metrics:** Uptime, memory, disk usage
- **Configuration Summary:** Current settings overview
- **Active LLM:** Shows which model is currently active
- **Connection Status:** Database and API health

#### **3. Model Tab**
- **LLM Provider Selection:** Groq or Google
- **Model Configuration:** Set model names and parameters
- **Embedding Provider:** Google or Cohere embeddings
- **Real-time Updates:** Changes apply immediately

#### **4. Configuration Tab**
- **Environment Variables:** Edit all settings via web interface
- **Validation:** Check configuration before applying
- **Backup/Restore:** Manage configuration backups
- **Cache Management:** Invalidate caches when needed

#### **5. Quick Actions Tab**
- **System Maintenance:** Log cleanup, cache invalidation
- **Backup Operations:** Create manual backups
- **Service Restart:** Restart Docker container
- **Health Checks:** Verify system components

---

## ⚙️ Configuration Management

### **Key Configuration Files:**

#### **1. .env File (Main Configuration)**
```bash
# Database Configuration
QDRANT_URL=https://your-qdrant-instance:6333
QDRANT_API_KEY=your-api-key
QDRANT_COLLECTION=hr_documents

# LLM Configuration
GROQ_API_KEY=your-groq-key
GOOGLE_API_KEY=your-google-key
MODEL_PRIORITY=1  # 1=Groq, 2=Google
GROQ_MODEL=llama3-70b-8192
GOOGLE_MODEL=gemma-3-12b-it

# Performance Settings
WORKERS=17
TIMEOUT=180
KEEP_ALIVE=45
PORT=8000

# File Upload Settings
MAX_FILE_SIZE_MB=200
ALLOWED_FILE_EXTENSIONS=.pdf,.docx,.doc,.txt,.md
UPLOAD_DIR=uploaded_folders

# Vector Search Settings
CHUNK_SIZE=512
CHUNK_OVERLAP=128
SEARCH_K=4
FETCH_K=15
SCORE_THRESHOLD=0.5
```

#### **2. docker-compose.yml (Container Settings)**
```yaml
deploy:
  resources:
    limits:
      cpus: '16.0'    # Use all 16 cores
      memory: 14G     # Use all 14GB RAM
    reservations:
      cpus: '2.0'     # Reserve 2 cores for system
      memory: 2G      # Reserve 2GB for system
```

#### **3. gunicorn.conf.py (Server Settings)**
```python
workers = 17
threads = 2
worker_connections = 1000
timeout = 180
keepalive = 45
```

### **Configuration via Frontend:**

#### **Step 1: Access Configuration Tab**
1. Go to `http://localhost:8000/admin/dashboard`
2. Click "Configuration" tab

#### **Step 2: Update Settings**
1. **Environment Variables:**
   - Find the variable you want to change
   - Click "Edit" button
   - Enter new value
   - Click "Save"

2. **Model Configuration:**
   - Go to "Model" tab
   - Select LLM provider (Groq/Google)
   - Enter model names
   - Click "Save Model Settings"

#### **Step 3: Apply Changes**
1. **Reload Configuration:**
   - Click "Reload Configuration" button
   - Wait for confirmation message

2. **Invalidate Cache (if needed):**
   - Click "Invalidate All Caches"
   - This ensures new settings take effect

#### **Step 4: Verify Changes**
1. Check "System Status" tab
2. Verify new configuration is active
3. Test with a sample question

---

## 🐳 Docker Management

### **Current Docker Configuration:**
```yaml
# Container Resources
cpus: '16.0'        # 16 cores
memory: 14G         # 14GB RAM
workers: 17         # 17 worker processes
shm_size: '512mb'   # Shared memory
```

### **Docker Commands:**

#### **1. Start the Application:**
```bash
# Start with Docker Compose
docker-compose up -d

# Check status
docker-compose ps
```

#### **2. Stop the Application:**
```bash
# Stop gracefully
docker-compose down

# Force stop
docker-compose down --force
```

#### **3. Restart the Application:**
```bash
# Restart with new configuration
docker-compose down
docker-compose up -d

# Or restart in one command
docker-compose restart
```

#### **4. View Logs:**
```bash
# View real-time logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f hr-assistant

# View last 100 lines
docker-compose logs --tail=100 hr-assistant
```

#### **5. Update Configuration:**
```bash
# After changing .env or docker-compose.yml
docker-compose down
docker-compose up -d --build
```

### **Docker via Frontend:**

#### **Restart from Admin Dashboard:**
1. Go to `http://localhost:8000/admin/dashboard`
2. Click "Quick Actions" tab
3. Click "Restart Services" button
4. Wait for restart confirmation

#### **Monitor Container Status:**
1. Check "System Status" tab
2. View uptime and resource usage
3. Monitor active connections

---

## 🔧 Troubleshooting

### **Common Issues & Solutions:**

#### **1. Application Won't Start:**
```bash
# Check Docker status
docker-compose ps

# View error logs
docker-compose logs hr-assistant

# Check resource availability
docker system df
```

#### **2. High Memory Usage:**
```bash
# Check memory usage
docker stats hr-assistant

# Reduce workers if needed
# Edit .env: WORKERS=12
```

#### **3. Slow Response Times:**
```bash
# Check LLM API status
curl -f http://localhost:8000/health

# Invalidate caches
# Use admin dashboard: "Invalidate All Caches"
```

#### **4. Configuration Not Applied:**
```bash
# Reload configuration
# Use admin dashboard: "Reload Configuration"

# Or restart container
docker-compose restart
```

#### **5. File Upload Issues:**
```bash
# Check upload directory permissions
ls -la uploaded_folders/

# Check disk space
df -h
```

### **Health Check Endpoints:**
```bash
# Application health
curl http://localhost:8000/health

# Admin status
curl http://localhost:8000/admin/status

# Configuration validation
curl http://localhost:8000/admin/config/validate
```

---

## ⚡ Performance Optimization

### **Current Performance Settings:**
- **Concurrent Users:** 50-80 users
- **Response Time:** 2-6 seconds
- **Memory Usage:** 8-12GB under load
- **CPU Usage:** 60-90% under load

### **Optimization Recommendations:**

#### **1. For Higher Concurrency:**
```bash
# Increase workers
WORKERS=20

# Increase timeout
TIMEOUT=240

# Optimize memory
shm_size: '1gb'
```

#### **2. For Lower Latency:**
```bash
# Use Groq (faster than Google)
MODEL_PRIORITY=1
GROQ_MODEL=llama3-70b-8192

# Reduce search parameters
SEARCH_K=3
FETCH_K=10
```

#### **3. For Better Resource Usage:**
```bash
# Enable request logging
ENABLE_REQUEST_LOGGING=true

# Enable performance monitoring
ENABLE_PERFORMANCE_MONITORING=true
```

---

## 💾 Backup & Recovery

### **Backup Types:**

#### **1. Configuration Backup:**
```bash
# Automatic backup on config changes
# Location: .env.env.backup

# Manual backup via admin dashboard
# Go to "Quick Actions" → "Create Backup"
```

#### **2. Data Backup:**
```bash
# Vector database backup
# Use admin dashboard: "Create Backup" with "Include Vectors"

# Upload files backup
# Use admin dashboard: "Create Backup" with "Include Uploads"
```

#### **3. System Backup:**
```bash
# Full system backup
docker-compose down
tar -czf backup-$(date +%Y%m%d).tar.gz .
docker-compose up -d
```

### **Recovery Procedures:**

#### **1. Configuration Recovery:**
```bash
# Restore from backup
cp .env.env.backup .env

# Restart application
docker-compose restart
```

#### **2. Data Recovery:**
```bash
# Restore from admin dashboard
# Go to "Quick Actions" → "Restore Backup"
```

#### **3. Full System Recovery:**
```bash
# Stop application
docker-compose down

# Restore from backup
tar -xzf backup-YYYYMMDD.tar.gz

# Restart application
docker-compose up -d
```

---

## 📊 Monitoring & Maintenance

### **Daily Tasks:**
1. Check system status via admin dashboard
2. Monitor response times
3. Review error logs
4. Check disk space

### **Weekly Tasks:**
1. Clean up old logs
2. Review backup retention
3. Update configuration if needed
4. Monitor performance metrics

### **Monthly Tasks:**
1. Full system backup
2. Performance review
3. Configuration optimization
4. Security updates

---

## 🚀 Quick Reference Commands

### **Essential Commands:**
```bash
# Start application
docker-compose up -d

# Stop application
docker-compose down

# Restart application
docker-compose restart

# View logs
docker-compose logs -f

# Check status
docker-compose ps

# Update configuration
docker-compose down && docker-compose up -d
```

### **Admin Dashboard URLs:**
- **Main Dashboard:** http://localhost:8000/admin/dashboard
- **System Status:** http://localhost:8000/admin/status
- **Configuration:** http://localhost:8000/admin/config
- **Health Check:** http://localhost:8000/health

### **Emergency Contacts:**
- **System Admin:** [Your Contact]
- **Technical Support:** [Support Contact]
- **Documentation:** This guide

---

## 📝 Notes

- **Always backup before major changes**
- **Test configuration changes in staging first**
- **Monitor system resources during peak usage**
- **Keep this guide updated with any changes**

---

*Last Updated: [Current Date]*
*Version: 1.0* 