import multiprocessing
import os

# Server Socket
bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
backlog = 2048  # Reasonable backlog for 10-15 concurrent users

# Worker Processes
workers_str = os.getenv('WORKERS', '').split('#')[0].strip()
workers = int(workers_str) if workers_str.isdigit() else (multiprocessing.cpu_count() * 2 + 1)
worker_class = 'uvicorn.workers.UvicornWorker'
worker_connections = 1000
max_requests = 1000  # Restart workers after this many requests
max_requests_jitter = 100  # Random jitter to prevent thundering herd
timeout = int(os.getenv('TIMEOUT', '120'))
keepalive = int(os.getenv('KEEP_ALIVE', '30'))
threads = 2  # 2 threads per worker for I/O bound operations

# Memory Management
preload_app = True  # Load application before forking workers
max_requests = 1000
max_requests_jitter = 200
worker_tmp_dir = '/dev/shm'  # Use shared memory for worker temp files

# Security
limit_request_line = 4094  # Maximum size of HTTP request line
limit_request_fields = 100  # Limit number of HTTP headers
limit_request_field_size = 8190  # Limit size of HTTP request headers

# Debugging
reload = os.getenv('RELOAD', 'false').lower() == 'false'
reload_engine = 'auto'

# Logging
loglevel = os.getenv('LOG_LEVEL', 'info')
accesslog = '-'
errorlog = '-'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(L)s %(D)s'

# Process Naming
proc_name = 'hr_assistant'

# Server Hooks
def on_starting(server):
    server.log.info(f"Starting HR Assistant API Server with {workers} workers and {threads} threads")

def on_reload(server):
    server.log.info("Reloading HR Assistant API Server")

def when_ready(server):
    server.log.info("HR Assistant API Server is ready and accepting connections")

def worker_int(worker):
    worker.log.info(f"Worker {worker.pid} received INT or QUIT signal")

def worker_abort(worker):
    worker.log.warning(f"Worker {worker.pid} received SIGABRT signal")
