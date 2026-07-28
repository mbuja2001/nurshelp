# Phase 1 Implementation: Detailed Step-by-Step Guide

## Overview
This document provides exact implementation steps for:
1. **Measurement Phase**: Concurrent capacity testing (BEFORE queue implementation)
2. **Implementation Phase 1a**: Request queue + retry logic (Python backend)
3. **Implementation Phase 1b**: Async API updates (Node.js backend)
4. **Validation Phase**: Concurrent capacity testing (AFTER queue implementation)

---

## PART 1: MEASURE BASELINE CONCURRENT CAPACITY

### Goal
Determine how many concurrent requests your current setup can handle before failures spike, and at what concurrency level the 50% failure rate kicks in.

### Test Script Architecture
```
┌─────────────────────────────────────────┐
│  test_concurrent_capacity.py            │
│  (Python - submits triage requests)     │
└─────────────────────────────────────────┘
              ↓ (HTTP POST)
┌─────────────────────────────────────────┐
│  Current /llm-triage endpoint           │
│  (Flask - SYNCHRONOUS, no queue yet)    │
└─────────────────────────────────────────┘
              ↓ (Direct call)
┌─────────────────────────────────────────┐
│  gguf_generate(max_tokens=3100)         │
│  (MedGemma inference)                   │
└─────────────────────────────────────────┘
```

### Step 1.1: Create Test Data File

**File**: `backend/ML/test_data_concurrent.json`

```json
{
  "test_cases": [
    {
      "id": "case_1",
      "patient": {
        "id_number": "PT-001",
        "age": 45,
        "gender": "male",
        "symptoms": "Sudden onset chest pain radiating to left arm. SOB. Diaphoretic. Nausea."
      },
      "vitals": {
        "temp": 36.8,
        "bp_systolic": 155,
        "bp_diastolic": 95,
        "hr": 105,
        "o2": 92,
        "resp": 22
      },
      "chiefComplaint": "Acute chest pain with SOB",
      "triageNotes": "Patient appears in distress, clutching chest",
      "medicalHistory": "HTN, Type 2 DM"
    },
    {
      "id": "case_2",
      "patient": {
        "id_number": "PT-002",
        "age": 28,
        "gender": "female",
        "symptoms": "Fever 39.2°C, rash on trunk, neck stiffness"
      },
      "vitals": {
        "temp": 39.2,
        "bp_systolic": 118,
        "bp_diastolic": 72,
        "hr": 118,
        "o2": 98,
        "resp": 20
      },
      "chiefComplaint": "Fever with rash and neck pain",
      "triageNotes": "Petechial rash on arms",
      "medicalHistory": "None"
    },
    {
      "id": "case_3",
      "patient": {
        "id_number": "PT-003",
        "age": 67,
        "gender": "male",
        "symptoms": "Sudden weakness left side, slurred speech"
      },
      "vitals": {
        "temp": 37.1,
        "bp_systolic": 172,
        "bp_diastolic": 98,
        "hr": 92,
        "o2": 96,
        "resp": 18
      },
      "chiefComplaint": "Acute neurological symptoms",
      "triageNotes": "Facial droop on left, arm weakness",
      "medicalHistory": "HTN, AFib"
    },
    {
      "id": "case_4",
      "patient": {
        "id_number": "PT-004",
        "age": 52,
        "gender": "female",
        "symptoms": "Severe abdominal pain, vomiting"
      },
      "vitals": {
        "temp": 38.5,
        "bp_systolic": 105,
        "bp_diastolic": 68,
        "hr": 128,
        "o2": 97,
        "resp": 24
      },
      "chiefComplaint": "Severe RUQ pain",
      "triageNotes": "Positive Murphy's sign, guarding",
      "medicalHistory": "Obesity, gallstones"
    }
  ]
}
```

### Step 1.2: Create Concurrent Test Script

**File**: `backend/ML/test_concurrent_capacity.py`

```python
"""
Test script to measure concurrent capacity before queue implementation.
Submits N simultaneous triage requests and measures:
- Success/failure rate
- Inference time distribution
- Queue wait time (if applicable)
- JSON parse error types
"""

import asyncio
import aiohttp
import json
import time
import statistics
from typing import List, Dict, Tuple
from datetime import datetime
import sys

# Configuration
FLASK_URL = "http://localhost:5000/llm-triage"
TEST_DATA_FILE = "test_data_concurrent.json"
CONCURRENCY_LEVELS = [1, 2, 3, 4, 5, 10]  # Test at increasing load

class ConcurrencyTester:
    def __init__(self, flask_url: str, test_data_file: str):
        self.flask_url = flask_url
        self.test_data_file = test_data_file
        self.results = {
            "concurrency_level": None,
            "total_requests": 0,
            "successful": 0,
            "failed": 0,
            "success_rate": 0.0,
            "inference_times": [],
            "error_types": {},
            "raw_results": []
        }
    
    async def submit_triage_request(self, session: aiohttp.ClientSession, 
                                   request_id: int, test_case: Dict) -> Dict:
        """Submit single triage request and measure time"""
        start_time = time.time()
        
        try:
            async with session.post(
                self.flask_url,
                json=test_case,
                timeout=aiohttp.ClientTimeout(total=120)  # 2 min timeout per request
            ) as resp:
                elapsed = time.time() - start_time
                
                if resp.status == 200:
                    result = await resp.json()
                    return {
                        "request_id": request_id,
                        "status": "success",
                        "inference_time": elapsed,
                        "response_size": len(json.dumps(result)),
                        "esi_level": result.get("esi_level_from_llm"),
                        "specialty": result.get("specialty"),
                        "error": None
                    }
                else:
                    error_text = await resp.text()
                    return {
                        "request_id": request_id,
                        "status": "http_error",
                        "http_status": resp.status,
                        "inference_time": elapsed,
                        "error": error_text[:200]
                    }
        
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            return {
                "request_id": request_id,
                "status": "timeout",
                "inference_time": elapsed,
                "error": "Request exceeded 120s timeout"
            }
        
        except json.JSONDecodeError as e:
            elapsed = time.time() - start_time
            return {
                "request_id": request_id,
                "status": "json_parse_error",
                "inference_time": elapsed,
                "error": f"JSON parse failed: {str(e)[:100]}"
            }
        
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "request_id": request_id,
                "status": "exception",
                "inference_time": elapsed,
                "error": str(e)[:100]
            }
    
    async def run_concurrent_test(self, concurrency_level: int, 
                                 test_cases: List[Dict]) -> Dict:
        """Run N concurrent requests"""
        self.results["concurrency_level"] = concurrency_level
        self.results["total_requests"] = concurrency_level
        self.results["successful"] = 0
        self.results["failed"] = 0
        self.results["inference_times"] = []
        self.results["error_types"] = {}
        self.results["raw_results"] = []
        
        print(f"\n{'='*80}")
        print(f"CONCURRENCY TEST: {concurrency_level} simultaneous requests")
        print(f"{'='*80}")
        print(f"Start time: {datetime.now().isoformat()}")
        
        # Create multiple requests (cycle through test cases)
        requests = []
        for i in range(concurrency_level):
            test_case = test_cases[i % len(test_cases)]
            requests.append((i, test_case))
        
        # Submit all requests concurrently
        batch_start = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.submit_triage_request(session, req_id, test_case)
                for req_id, test_case in requests
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        batch_elapsed = time.time() - batch_start
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                self.results["failed"] += 1
                error_type = "exception"
                self.results["error_types"][error_type] = \
                    self.results["error_types"].get(error_type, 0) + 1
                self.results["raw_results"].append({
                    "status": "exception",
                    "error": str(result)[:100]
                })
            else:
                self.results["raw_results"].append(result)
                
                if result["status"] == "success":
                    self.results["successful"] += 1
                    self.results["inference_times"].append(result["inference_time"])
                else:
                    self.results["failed"] += 1
                    error_type = result["status"]
                    self.results["error_types"][error_type] = \
                        self.results["error_types"].get(error_type, 0) + 1
        
        # Calculate statistics
        self.results["success_rate"] = (
            self.results["successful"] / self.results["total_requests"] * 100
            if self.results["total_requests"] > 0
            else 0
        )
        
        # Print summary
        print(f"\n✅ RESULTS SUMMARY:")
        print(f"  Total Requests: {self.results['total_requests']}")
        print(f"  Successful: {self.results['successful']} ({self.results['success_rate']:.1f}%)")
        print(f"  Failed: {self.results['failed']} ({100-self.results['success_rate']:.1f}%)")
        print(f"  Total Batch Time: {batch_elapsed:.2f}s")
        
        if self.results["inference_times"]:
            print(f"\n⏱️  INFERENCE TIME STATISTICS (successful requests only):")
            times = sorted(self.results["inference_times"])
            print(f"  Min: {times[0]:.2f}s")
            print(f"  Max: {times[-1]:.2f}s")
            print(f"  Avg: {statistics.mean(times):.2f}s")
            print(f"  Median: {statistics.median(times):.2f}s")
            print(f"  P95: {times[int(len(times)*0.95)]:.2f}s")
            print(f"  P99: {times[int(len(times)*0.99)]:.2f}s" if len(times) > 10 else "  P99: N/A (< 10 samples)")
        
        if self.results["error_types"]:
            print(f"\n❌ ERROR BREAKDOWN:")
            for error_type, count in self.results["error_types"].items():
                pct = count / self.results["total_requests"] * 100
                print(f"  {error_type}: {count} ({pct:.1f}%)")
        
        print(f"\nEnd time: {datetime.now().isoformat()}\n")
        
        return self.results
    
    def save_results_csv(self, all_results: List[Dict]):
        """Save results to CSV for analysis"""
        import csv
        
        csv_file = f"concurrent_capacity_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'concurrency_level', 'total_requests', 'successful', 'failed',
                'success_rate', 'avg_inference_time', 'p95_inference_time',
                'error_json_parse_error', 'error_timeout', 'error_http_error'
            ])
            writer.writeheader()
            
            for result in all_results:
                avg_time = statistics.mean(result["inference_times"]) if result["inference_times"] else 0
                p95_time = (
                    sorted(result["inference_times"])[int(len(result["inference_times"])*0.95)]
                    if result["inference_times"] else 0
                )
                
                writer.writerow({
                    'concurrency_level': result['concurrency_level'],
                    'total_requests': result['total_requests'],
                    'successful': result['successful'],
                    'failed': result['failed'],
                    'success_rate': f"{result['success_rate']:.1f}%",
                    'avg_inference_time': f"{avg_time:.2f}s",
                    'p95_inference_time': f"{p95_time:.2f}s",
                    'error_json_parse_error': result['error_types'].get('json_parse_error', 0),
                    'error_timeout': result['error_types'].get('timeout', 0),
                    'error_http_error': result['error_types'].get('http_error', 0)
                })
        
        print(f"📊 Results saved to: {csv_file}")

def main():
    print("🧪 CONCURRENT CAPACITY TEST SUITE")
    print("=" * 80)
    print(f"Testing baseline capacity BEFORE queue implementation")
    print(f"Flask URL: {FLASK_URL}")
    print(f"Concurrency levels to test: {CONCURRENCY_LEVELS}")
    
    # Load test data
    try:
        with open(TEST_DATA_FILE, 'r') as f:
            test_data = json.load(f)
            test_cases = test_data['test_cases']
            print(f"✅ Loaded {len(test_cases)} test cases from {TEST_DATA_FILE}")
    except FileNotFoundError:
        print(f"❌ Test data file not found: {TEST_DATA_FILE}")
        sys.exit(1)
    
    # Run tests
    tester = ConcurrencyTester(FLASK_URL, TEST_DATA_FILE)
    all_results = []
    
    for concurrency_level in CONCURRENCY_LEVELS:
        # Delay between test runs to let server cool down
        if concurrency_level > 1:
            print(f"\n⏳ Waiting 10s before next concurrency level...")
            time.sleep(10)
        
        result = asyncio.run(tester.run_concurrent_test(concurrency_level, test_cases))
        all_results.append(result)
        
        # Stop testing if success rate drops below 30% (server is overwhelmed)
        if result["success_rate"] < 30 and concurrency_level > 2:
            print(f"\n⚠️  Success rate below 30% - server overwhelmed, stopping tests")
            break
    
    # Save results
    print(f"\n{'='*80}")
    print("📈 FINAL SUMMARY")
    print(f"{'='*80}")
    
    for result in all_results:
        print(f"Concurrency {result['concurrency_level']:2d}: "
              f"{result['successful']:2d}/{result['total_requests']:2d} "
              f"({result['success_rate']:5.1f}%) "
              f"Avg: {statistics.mean(result['inference_times']) if result['inference_times'] else 0:.1f}s")
    
    tester.save_results_csv(all_results)
    
    print(f"\n✅ Tests complete. Review CSV results for detailed analysis.")
    print(f"Use this baseline to measure improvement after implementing queue system.")

if __name__ == "__main__":
    main()
```

### Step 1.3: Run the Baseline Test

**Terminal Command:**
```bash
cd backend/ML
python test_concurrent_capacity.py
```

**Expected Output:**
```
================================================================================
CONCURRENCY TEST: 1 simultaneous requests
================================================================================
Start time: 2026-06-21T14:30:00

✅ RESULTS SUMMARY:
  Total Requests: 1
  Successful: 1 (100.0%)
  Failed: 0 (0.0%)
  Total Batch Time: 25.34s

⏱️  INFERENCE TIME STATISTICS (successful requests only):
  Min: 25.34s
  Max: 25.34s
  Avg: 25.34s
  Median: 25.34s
  P95: 25.34s

================================================================================
CONCURRENCY TEST: 2 simultaneous requests
================================================================================
...
  Total Requests: 2
  Successful: 2 (100.0%)
  Failed: 0 (0.0%)
...

================================================================================
CONCURRENCY TEST: 3 simultaneous requests
================================================================================
...
  Total Requests: 3
  Successful: 2 (66.7%)
  Failed: 1 (33.3%)

❌ ERROR BREAKDOWN:
  json_parse_error: 1 (33.3%)
...

📊 Results saved to: concurrent_capacity_results_20260621_143000.csv
```

**Key Metrics to Watch:**
- At what concurrency level does success rate drop below 90%? (this is the threshold)
- At what level does JSON parse errors start appearing?
- How much slower is inference at higher concurrency? (p95 time increase)

---

## PART 2: IMPLEMENT PHASE 1a - PYTHON BACKEND QUEUE

### Step 2.1: Create Queue Management Module

**File**: `backend/ML/llm_request_queue.py` (NEW FILE)

```python
"""
Request queue system for managing concurrent LLM triage requests.
Features:
- Max 2 concurrent workers (prevent CPU starvation)
- Priority queue (emergencies first)
- Request deduplication (avoid re-triaging same patient)
- Job status tracking (pending, processing, complete, failed)
- Request caching (30s TTL for identical requests)
"""

import asyncio
import uuid
import time
import json
import threading
from queue import PriorityQueue
from typing import Dict, Optional, Callable
from datetime import datetime, timedelta
import hashlib

class TriageRequest:
    """Single triage request in queue"""
    
    def __init__(self, job_id: str, patient_data: Dict, priority: int = 5):
        self.job_id = job_id
        self.patient_data = patient_data
        self.priority = priority  # 1=high (emergency), 5=normal, 10=low
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.status = 'pending'  # pending, processing, complete, failed
        self.result = None
        self.error = None
        self.retry_count = 0
        self.max_retries = 3
    
    def __lt__(self, other):
        """Priority queue comparison (lower priority value = higher priority)"""
        return self.priority < other.priority
    
    @property
    def queue_wait_time(self) -> float:
        """Time spent waiting in queue (seconds)"""
        if self.started_at:
            return (self.started_at - self.created_at).total_seconds()
        else:
            return (datetime.now() - self.created_at).total_seconds()
    
    @property
    def processing_time(self) -> Optional[float]:
        """Time spent processing (seconds)"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.now() - self.started_at).total_seconds()
        else:
            return None
    
    @property
    def total_time(self) -> float:
        """Total time from submission to completion"""
        if self.completed_at:
            return (self.completed_at - self.created_at).total_seconds()
        else:
            return (datetime.now() - self.created_at).total_seconds()


class RequestCache:
    """Simple request cache with TTL"""
    
    def __init__(self, ttl_seconds: int = 30):
        self.ttl_seconds = ttl_seconds
        self.cache = {}  # {request_hash: (result, expiry_time)}
        self.lock = threading.Lock()
    
    def get_request_hash(self, patient_data: Dict) -> str:
        """Create deterministic hash of request for deduplication"""
        # Include patient ID, vitals, symptoms (exclude timestamp)
        key_parts = [
            str(patient_data.get('id_number', '')),
            str(patient_data.get('vitals', {})),
            str(patient_data.get('symptoms', '')),
        ]
        key_str = '|'.join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, patient_data: Dict) -> Optional[Dict]:
        """Retrieve cached result if exists and not expired"""
        request_hash = self.get_request_hash(patient_data)
        
        with self.lock:
            if request_hash in self.cache:
                result, expiry_time = self.cache[request_hash]
                if datetime.now() < expiry_time:
                    return result
                else:
                    del self.cache[request_hash]
        
        return None
    
    def set(self, patient_data: Dict, result: Dict):
        """Store result in cache"""
        request_hash = self.get_request_hash(patient_data)
        expiry_time = datetime.now() + timedelta(seconds=self.ttl_seconds)
        
        with self.lock:
            self.cache[request_hash] = (result, expiry_time)
    
    def clear_expired(self):
        """Remove expired cache entries"""
        with self.lock:
            now = datetime.now()
            expired_keys = [
                k for k, (_, expiry) in self.cache.items()
                if now >= expiry
            ]
            for k in expired_keys:
                del self.cache[k]


class TriageRequestQueue:
    """Central queue manager for LLM triage requests"""
    
    def __init__(self, max_workers: int = 2, max_queue_size: int = 100):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.queue = PriorityQueue()
        self.jobs = {}  # {job_id: TriageRequest}
        self.lock = threading.Lock()
        self.cache = RequestCache(ttl_seconds=30)
        self.active_workers = 0
        self.worker_thread = None
        self.shutdown_event = threading.Event()
    
    def enqueue(self, patient_data: Dict, priority: int = 5, 
                job_id: Optional[str] = None) -> str:
        """
        Add request to queue.
        Returns: job_id (for polling status)
        """
        job_id = job_id or str(uuid.uuid4())
        
        # Check cache first
        cached_result = self.cache.get(patient_data)
        if cached_result:
            print(f'[queue] 💾 CACHE HIT for job {job_id} - returning cached result')
            request = TriageRequest(job_id, patient_data, priority)
            request.status = 'complete'
            request.result = cached_result
            with self.lock:
                self.jobs[job_id] = request
            return job_id
        
        # Check for duplicates already in queue
        with self.lock:
            for existing_id, existing_req in list(self.jobs.items()):
                if (existing_req.status == 'pending' or existing_req.status == 'processing') and \
                   self.cache.get_request_hash(existing_req.patient_data) == \
                   self.cache.get_request_hash(patient_data):
                    print(f'[queue] 🔁 DUPLICATE DETECTED - job {job_id} reuses {existing_id}')
                    return existing_id  # Return existing job ID
        
        # Create new request
        request = TriageRequest(job_id, patient_data, priority)
        
        # Check queue not full
        if self.queue.qsize() >= self.max_queue_size:
            raise RuntimeError(f'Queue full (max {self.max_queue_size})')
        
        # Add to queue
        with self.lock:
            self.queue.put(request)
            self.jobs[job_id] = request
        
        print(f'[queue] ✅ Job {job_id} enqueued (priority {priority}, position {self.queue.qsize()})')
        return job_id
    
    def get_status(self, job_id: str) -> Dict:
        """Get current status of a job"""
        with self.lock:
            if job_id not in self.jobs:
                return {'status': 'not_found'}
            
            request = self.jobs[job_id]
            
            if request.status == 'complete':
                return {
                    'status': 'complete',
                    'result': request.result,
                    'queue_wait_time': request.queue_wait_time,
                    'processing_time': request.processing_time,
                    'total_time': request.total_time
                }
            elif request.status == 'failed':
                return {
                    'status': 'failed',
                    'error': request.error,
                    'queue_wait_time': request.queue_wait_time,
                    'processing_time': request.processing_time,
                    'total_time': request.total_time,
                    'retry_count': request.retry_count
                }
            elif request.status == 'processing':
                return {
                    'status': 'processing',
                    'queue_wait_time': request.queue_wait_time,
                    'processing_time': request.processing_time
                }
            else:  # pending
                queue_position = sum(
                    1 for _, j in list(self.jobs.items())
                    if j.status == 'pending' and j.priority <= request.priority and j.created_at < request.created_at
                ) + 1
                
                return {
                    'status': 'pending',
                    'queue_position': queue_position,
                    'queue_size': self.queue.qsize(),
                    'queue_wait_time': request.queue_wait_time
                }
    
    def get_queue_stats(self) -> Dict:
        """Get overall queue statistics"""
        with self.lock:
            pending = sum(1 for j in self.jobs.values() if j.status == 'pending')
            processing = sum(1 for j in self.jobs.values() if j.status == 'processing')
            complete = sum(1 for j in self.jobs.values() if j.status == 'complete')
            failed = sum(1 for j in self.jobs.values() if j.status == 'failed')
            
            return {
                'pending': pending,
                'processing': processing,
                'complete': complete,
                'failed': failed,
                'total': len(self.jobs),
                'active_workers': self.active_workers,
                'max_workers': self.max_workers,
                'queue_size': self.queue.qsize()
            }
    
    def process_queue_worker(self, process_func: Callable):
        """
        Worker thread that processes jobs from queue.
        process_func: async function(request) -> result
        """
        print('[queue] 🚀 Queue worker started')
        
        while not self.shutdown_event.is_set():
            try:
                # Get next job (with timeout to allow shutdown)
                if self.queue.empty():
                    time.sleep(0.5)
                    continue
                
                request = self.queue.get(timeout=1)
                
                with self.lock:
                    if self.active_workers >= self.max_workers:
                        # Put it back if we have too many workers
                        self.queue.put(request)
                        time.sleep(0.5)
                        continue
                    
                    request.status = 'processing'
                    request.started_at = datetime.now()
                    self.active_workers += 1
                
                print(f'[queue] 🔄 Worker processing job {request.job_id} (retry {request.retry_count}/{request.max_retries})')
                
                try:
                    # Call the processing function
                    result = process_func(request)
                    
                    request.result = result
                    request.status = 'complete'
                    request.completed_at = datetime.now()
                    
                    # Store in cache
                    self.cache.set(request.patient_data, result)
                    
                    print(f'[queue] ✅ Job {request.job_id} complete ({request.total_time:.1f}s total)')
                
                except Exception as e:
                    print(f'[queue] ❌ Job {request.job_id} failed: {str(e)[:100]}')
                    
                    if request.retry_count < request.max_retries:
                        print(f'[queue] 🔁 Retrying job {request.job_id} ({request.retry_count+1}/{request.max_retries})')
                        request.retry_count += 1
                        request.status = 'pending'
                        request.started_at = None
                        self.queue.put(request)
                    else:
                        request.status = 'failed'
                        request.error = str(e)[:200]
                        request.completed_at = datetime.now()
                        print(f'[queue] 💥 Job {request.job_id} max retries exceeded')
                
                finally:
                    with self.lock:
                        self.active_workers -= 1
            
            except Exception as e:
                print(f'[queue] ⚠️  Worker error: {str(e)[:100]}')
                time.sleep(1)
    
    def start(self, process_func: Callable):
        """Start background worker thread"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.shutdown_event.clear()
            self.worker_thread = threading.Thread(
                target=self.process_queue_worker,
                args=(process_func,),
                daemon=True
            )
            self.worker_thread.start()
            print('[queue] ✅ Queue system started')
    
    def shutdown(self):
        """Stop queue processing"""
        self.shutdown_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        print('[queue] ✅ Queue system shut down')


# Global queue instance
_queue = None

def get_queue() -> TriageRequestQueue:
    """Get or create global queue"""
    global _queue
    if _queue is None:
        _queue = TriageRequestQueue(max_workers=2, max_queue_size=100)
    return _queue
```

### Step 2.2: Add Retry Logic Module

**File**: `backend/ML/llm_inference_retry.py` (NEW FILE)

```python
"""
Retry logic for LLM inference with dynamic max_tokens adjustment.
Handles JSON truncation recovery by reducing token budget on each retry.
"""

import json
import time
from typing import Tuple, Dict, Optional

def validate_json_completeness(json_str: str) -> bool:
    """
    Check if extracted JSON has all 14 required clinical fields
    and each field has actual content (not empty).
    """
    required_fields = [
        'esi_level', 'priority_code', 'specialty', 'urgency_level',
        'pathophysiology', 'clinical_discriminators', 'severity_assessment',
        'differential_diagnoses', 'recommended_specialists',
        'immediate_actions', 'pathognomonic_markers', 'diagnostic_workup',
        'clinical_summary', 'disposition_rationale'
    ]
    
    try:
        parsed = json.loads(json_str)
        if not isinstance(parsed, dict):
            return False
        
        for field in required_fields:
            if field not in parsed:
                print(f"[retry] ⚠️  Missing field: {field}")
                return False
            
            value = parsed[field]
            
            # Check for empty content
            if value is None:
                print(f"[retry] ⚠️  Field {field} is None")
                return False
            
            if isinstance(value, list) and len(value) == 0:
                print(f"[retry] ⚠️  Field {field} is empty array")
                return False
            
            if isinstance(value, str) and len(value.strip()) == 0:
                print(f"[retry] ⚠️  Field {field} is empty string")
                return False
        
        return True
    
    except json.JSONDecodeError as e:
        print(f"[retry] ⚠️  JSON parse error: {str(e)[:100]}")
        return False
    except Exception as e:
        print(f"[retry] ⚠️  Validation error: {str(e)[:100]}")
        return False


def extract_json_from_output(raw_output: str) -> str:
    """
    Extract valid JSON object from raw GGUF output.
    Handles markdown code fences and partial outputs.
    """
    cleaned = raw_output.strip()
    
    # Remove markdown code fences
    if cleaned.startswith('```json'):
        cleaned = cleaned[7:].strip()
    elif cleaned.startswith('```'):
        cleaned = cleaned[3:].strip()
    elif cleaned.startswith('`'):
        cleaned = cleaned[1:].strip()
    
    while cleaned.endswith('```') or cleaned.endswith('`'):
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3].strip()
        elif cleaned.endswith('`'):
            cleaned = cleaned[:-1].strip()
    
    # Find first { and matching }
    first_brace = cleaned.find('{')
    if first_brace == -1:
        raise ValueError("No JSON object found in output")
    
    # Smart extraction using brace depth tracking
    depth = 0
    in_string = False
    escape_next = False
    end_pos = -1
    
    for i in range(first_brace, len(cleaned)):
        char = cleaned[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\' and in_string:
            escape_next = True
            continue
        
        if char == '"':
            in_string = not in_string
            continue
        
        if not in_string:
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    end_pos = i
                    break
    
    if end_pos == -1:
        # Incomplete JSON - try to repair
        print(f"[retry] 🔧 Incomplete JSON detected (depth={depth}), attempting repair...")
        return repair_incomplete_json(cleaned[first_brace:], depth, in_string)
    
    return cleaned[first_brace:end_pos+1]


def repair_incomplete_json(raw_json: str, depth: int, in_string: bool) -> str:
    """
    Repair JSON that was truncated mid-field.
    Closes unclosed strings and braces.
    """
    repaired = raw_json
    
    # Close unclosed string
    if in_string:
        repaired = repaired + '"'
        print(f"[retry] 🔧 Added closing quote for unclosed string")
    
    # Close unclosed braces
    while depth > 0:
        repaired = repaired + '}'
        depth -= 1
    
    if depth < 0:
        print(f"[retry] ⚠️  Warning: more closing braces than opening (depth={depth})")
    else:
        print(f"[retry] 🔧 Added {depth} closing braces")
    
    return repaired


def gguf_generate_with_retry(
    gguf_generate_func,
    prompt: str,
    initial_max_tokens: int = 3100,
    max_retries: int = 3
) -> Tuple[Dict, bool]:
    """
    Call GGUF generation with automatic retry on incomplete JSON.
    
    Returns: (result_dict, success_bool)
    """
    max_token_budgets = [initial_max_tokens, 2500, 2000, 1500]
    last_output = None
    last_error = None
    
    for attempt in range(min(len(max_token_budgets), max_retries + 1)):
        max_tokens = max_token_budgets[attempt]
        
        print(f"\n[retry] 🔄 GGUF INFERENCE - Attempt {attempt+1}/{len(max_token_budgets)}")
        print(f"[retry]    max_tokens: {max_tokens}")
        
        try:
            # Call GGUF inference
            start_time = time.time()
            gguf_output = gguf_generate_func(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.0
            )
            elapsed = time.time() - start_time
            
            print(f"[retry] ⏱️  Inference completed in {elapsed:.1f}s ({len(gguf_output)//4} tokens)")
            
            # Extract JSON from output
            try:
                json_str = extract_json_from_output(gguf_output)
                last_output = json_str
            except ValueError as e:
                print(f"[retry] ⚠️  JSON extraction failed: {str(e)}")
                last_error = str(e)
                
                if attempt < len(max_token_budgets) - 1:
                    print(f"[retry] 🔄 Retrying with smaller max_tokens...")
                    time.sleep(1)  # Brief delay before retry
                    continue
                else:
                    return {}, False
            
            # Validate JSON completeness
            is_complete = validate_json_completeness(json_str)
            
            if is_complete:
                print(f"[retry] ✅ Complete JSON on attempt {attempt+1}")
                parsed = json.loads(json_str)
                return parsed, True
            else:
                print(f"[retry] ⚠️  Incomplete JSON on attempt {attempt+1}")
                last_error = "Incomplete JSON (missing or empty fields)"
                
                if attempt < len(max_token_budgets) - 1:
                    print(f"[retry] 🔄 Retrying with smaller max_tokens...")
                    time.sleep(1)  # Exponential backoff
                    continue
                else:
                    # Last attempt - use best effort
                    print(f"[retry] ⚠️  Max retries exhausted, using incomplete JSON")
                    parsed = json.loads(json_str)
                    return parsed, False
        
        except Exception as e:
            print(f"[retry] ❌ Inference error: {str(e)[:100]}")
            last_error = str(e)
            
            if attempt < len(max_token_budgets) - 1:
                print(f"[retry] 🔄 Retrying...")
                time.sleep(1)
                continue
            else:
                raise
    
    # All retries exhausted
    print(f"[retry] 💥 All retry attempts failed")
    return {}, False
```

---

## PART 3: INTEGRATE INTO ESI_Engine.py

### Step 3.1: Locate Current Flask Route Handler

**Current Code Location**: `backend/ML/ESI_Engine.py` line ~8200

**Search for**: `@app.route('/llm-triage', methods=['POST'])`

This is the function that currently:
1. Receives request
2. Validates vitals
3. Calls `gguf_generate()` directly
4. Returns response

**We will refactor this to**:
1. Receive request
2. Enqueue to queue
3. Return jobId immediately
4. Add new status endpoint

### Step 3.2: Key Changes to Make

**Step 3.2a**: Add imports at top of ESI_Engine.py

```python
# Add after existing imports (around line 1-50)
from llm_request_queue import get_queue
from llm_inference_retry import gguf_generate_with_retry, validate_json_completeness

# Initialize queue when module loads
_triage_queue = None

def initialize_queue():
    """Initialize the request queue (called at Flask startup)"""
    global _triage_queue
    _triage_queue = get_queue()
    
    # Start worker thread with process function
    _triage_queue.start(process_triage_request_worker)
    print('[ESI_Engine] ✅ Triage queue initialized')
```

**Step 3.2b**: Create worker function

Add this new function before the Flask route handlers (around line ~8100):

```python
def process_triage_request_worker(request_obj):
    """
    Worker function that processes a single triage request.
    Called by queue worker thread.
    
    This is the actual inference + analysis logic (moved from original endpoint)
    """
    from llm_request_queue import TriageRequest
    
    print(f'[queue-worker] 🎯 Processing request {request_obj.job_id}')
    
    # Extract patient data from request
    patient_data = request_obj.patient_data
    patient = patient_data.get('patient', {})
    vitals_raw = patient_data.get('vitals', {})
    
    # [COPY ALL THE VITALS VALIDATION, TEWS, SAFETY GATES, ETC FROM CURRENT ROUTE HERE]
    # This is the core triage logic - ~1000 lines from current /llm-triage endpoint
    
    # ============================================================
    # STEPS 1-5: Vitals, Safety Gates, Specialty, RAG (UNCHANGED)
    # ============================================================
    print(f'[queue-worker] 📊 Validating vitals...')
    vitals, validation_errors, validation_flags, override_specialty, force_red, min_priority_override = validate_and_enhance_vitals(vitals_raw, patient_text, patient)
    
    # Calculate TEWS
    tews_comp, tews_total = tews_from_vitals(vitals)
    tews_priority = tews_to_priority_code(tews_total)
    
    print(f'[queue-worker] ✅ TEWS Score: {tews_total}')
    
    # ... [Rest of logic from current endpoint] ...
    
    # ============================================================
    # STEP 6: CALL MEDGEMMA WITH RETRY LOGIC (NEW)
    # ============================================================
    print(f'[queue-worker] 🎯 Calling MedGemma with retry logic...')
    
    try:
        # Call with automatic retry on incomplete JSON
        gguf_parsed, is_complete = gguf_generate_with_retry(
            gguf_generate_func=gguf_generate,
            prompt=triage_prompt,
            initial_max_tokens=3100,
            max_retries=3
        )
        
        if not is_complete:
            print(f'[queue-worker] ⚠️  JSON incomplete after all retries')
            # Continue with what we have - partial results better than failure
    
    except Exception as e:
        print(f'[queue-worker] ❌ Inference failed: {str(e)[:100]}')
        # Fall back to TEWS + emergency detection
        gguf_parsed = None
    
    # ============================================================
    # STEPS 7+: Field extraction, validation, response (UNCHANGED)
    # ============================================================
    # ... [Rest of logic] ...
    
    # Build supervisor_output (all 14 fields)
    supervisor_output = { ... }
    
    # Return result
    return {
        'esi_level_from_llm': esi_from_llm,
        'priority_code': priority_code,
        'specialty': final_specialty,
        'urgency_level': urgency_level,
        'pathophysiology': pathophysiology,
        'clinical_discriminators': discriminators,
        'severity_assessment': severity_assessment,
        'differential_diagnoses': diagnoses,
        'recommended_specialists': recommended_specialists,
        'immediate_actions': immediate,
        'pathognomonic_markers': markers,
        'diagnostic_workup': workup,
        'clinical_summary': clinical_summary,
        'disposition_rationale': disposition_rationale,
        'total_time': time.time() - request_obj.created_at.timestamp()
    }
```

**Step 3.2c**: Replace `/llm-triage` endpoint

Find the current `/llm-triage` route (around line ~8200) and replace with:

```python
@app.route('/llm-triage/enqueue', methods=['POST'])
def enqueue_triage_request():
    """
    Enqueue a triage request and return immediately with jobId.
    Frontend polls /llm-triage/status/{jobId} for result.
    """
    try:
        data = request.get_json() or {}
        
        # Check for red flags to set priority
        patient_text = data.get('chiefComplaint', '') + ' ' + data.get('triageNotes', '')
        red_flags = detect_red_flag_keywords(patient_text, data.get('vitals', {}))
        priority = 1 if red_flags.get('has_critical') else 5  # 1=urgent, 5=normal
        
        # Enqueue the request
        job_id = _triage_queue.enqueue(
            patient_data=data,
            priority=priority
        )
        
        queue_stats = _triage_queue.get_queue_stats()
        
        return {
            'job_id': job_id,
            'status': 'queued',
            'message': 'Triage request queued for processing',
            'queue_position': queue_stats['pending'],
            'status_url': f'/llm-triage/status/{job_id}'
        }, 202  # 202 Accepted
    
    except Exception as e:
        print(f'[/llm-triage/enqueue] ❌ Error: {str(e)}')
        return {'error': str(e)}, 400


@app.route('/llm-triage/status/<job_id>', methods=['GET'])
def get_triage_status(job_id):
    """
    Poll the status of a queued triage request.
    Returns: {status, result?, error?, queue_position?}
    """
    try:
        status_info = _triage_queue.get_status(job_id)
        
        if status_info['status'] == 'not_found':
            return {'error': f'Job {job_id} not found'}, 404
        
        return status_info, 200
    
    except Exception as e:
        print(f'[/llm-triage/status] ❌ Error: {str(e)}')
        return {'error': str(e)}, 400


@app.route('/llm-triage/stats', methods=['GET'])
def get_queue_stats():
    """
    Get overall queue statistics (for monitoring).
    """
    return _triage_queue.get_queue_stats(), 200
```

---

## PART 4: UPDATE NODE.JS BACKEND

This will be detailed in next section...

