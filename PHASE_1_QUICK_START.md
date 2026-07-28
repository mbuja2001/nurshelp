# Phase 1 Implementation: Quick Start Guide

## 📋 What You'll Do

This Phase 1 implementation adds a **request queue** to prevent LLM inference failures under concurrent load.

**Expected Improvement:**
- **Before**: 50% success rate at 8+ concurrent requests (JSON truncation failures)
- **After**: 95%+ success rate at 8+ concurrent requests (with retry logic)

**Timeline**: ~2-3 days total
- Day 1: Measure baseline + implement Python queue
- Day 2: Test & validate queue
- Day 3: Deploy Node.js integration + final testing

---

## 🎯 Immediate Action Plan

### STEP 1: Measure Baseline Capacity (TODAY - 30 minutes)

**1a. Create test data file**
```bash
cd backend/ML
# Create file: test_data_concurrent.json (see PHASE_1_IMPLEMENTATION_GUIDE.md, Step 1.1)
# Contains 4 diverse test cases (chest pain, meningitis, stroke, abdominal pain)
```

**1b. Create test script**
```bash
# Create file: test_concurrent_capacity.py (see PHASE_1_IMPLEMENTATION_GUIDE.md, Step 1.2)
# This script submits N concurrent requests and measures:
#   - Success rate
#   - Inference time distribution
#   - Error types (JSON truncation, timeouts, etc)
```

**1c. Run baseline test**
```bash
# Ensure Flask backend is running:
cd backend
python ESI_Engine.py  # Start on port 5000

# In another terminal:
cd backend/ML
python test_concurrent_capacity.py

# Expected output:
# CONCURRENCY TEST: 1 requests -> 100% success, 25s avg
# CONCURRENCY TEST: 2 requests -> 100% success, 25s avg
# CONCURRENCY TEST: 3 requests -> 100% success, 26s avg
# CONCURRENCY TEST: 4 requests -> 75-90% success (JSON errors start)
# CONCURRENCY TEST: 5+ requests -> 50% success (heavy failures)
```

**Expected Output File**: `concurrent_capacity_results_20260621_HHMMSS.csv`
- Keep this for comparison after Phase 1

---

### STEP 2: Implement Python Queue System (TODAY/TOMORROW - 2-3 hours)

**2a. Create Queue Module**
```bash
cd backend/ML

# Create file: llm_request_queue.py (see PHASE_1_IMPLEMENTATION_GUIDE.md, Step 2.1)
# This file contains:
#   - TriageRequest class (job metadata)
#   - RequestCache class (30s TTL deduplication)
#   - TriageRequestQueue class (PriorityQueue with 2 workers)
#   - Worker loop that processes jobs from queue
```

**2b. Create Retry Module**
```bash
# Create file: llm_inference_retry.py (see PHASE_1_IMPLEMENTATION_GUIDE.md, Step 2.2)
# This file contains:
#   - validate_json_completeness() - check all 14 fields
#   - extract_json_from_output() - smart JSON extraction
#   - repair_incomplete_json() - close unclosed strings/braces
#   - gguf_generate_with_retry() - retry on incomplete JSON with dynamic max_tokens
```

**2c. Update ESI_Engine.py**
```bash
# Location: backend/ML/ESI_Engine.py

# At TOP of file (after imports):
from llm_request_queue import get_queue
from llm_inference_retry import gguf_generate_with_retry, validate_json_completeness

# At BOTTOM of file (before Flask routes):
# Add imports and queue initialization function
_triage_queue = None

def initialize_queue():
    global _triage_queue
    _triage_queue = get_queue()
    _triage_queue.start(process_triage_request_worker)
    print('[ESI_Engine] ✅ Triage queue initialized')

# Around line 8100 (before current /llm-triage route):
# Add process_triage_request_worker() function
# This is the core triage logic (currently in /llm-triage endpoint)
# Will be called by queue worker thread

# Around line 8200 (replace current /llm-triage route):
# Replace /llm-triage endpoint with:
#   @app.route('/llm-triage/enqueue', methods=['POST'])
#   @app.route('/llm-triage/status/<job_id>', methods=['GET'])
#   @app.route('/llm-triage/stats', methods=['GET'])

# In Flask app.run() or startup:
initialize_queue()  # Start queue worker thread
```

**Key Code Snippet for ESI_Engine.py**:
```python
# Add these imports at top
from llm_request_queue import get_queue
from llm_inference_retry import gguf_generate_with_retry

# Initialize queue when module loads
_triage_queue = None

@app.before_request
def init_queue():
    global _triage_queue
    if _triage_queue is None:
        _triage_queue = get_queue()
        _triage_queue.start(process_triage_request_worker)
```

---

### STEP 3: Update Node.js Backend (TOMORROW - 1-2 hours)

**3a. Create Triage Queue Service**
```bash
cd backend/src/services

# Create file: triageQueueService.js (see PHASE_1_NODE_JS_INTEGRATION.md, Step 4.1)
# Exports:
#   - enqueueTriage(data) -> {jobId, queuePosition}
#   - pollTriageStatus(jobId) -> {status, result}
#   - triageAndWait(data) -> complete result
```

**3b. Update Triage Controller**
```bash
# File: backend/src/controllers/triage.controller.js

# Update runTriage() function:
#   OLD: const pythonRes = await axios.post(PYTHON_URL, ...)
#   NEW: const {jobId} = await enqueueTriage(enginePayload)
#        return res.status(202).json({jobId, statusUrl: ...})

# Add new functions:
#   - getTriageStatus(req, res) - GET /api/triage/status/:jobId
#   - getQueueStats(req, res) - GET /api/triage/queue-stats
```

**3c. Update Routes**
```bash
# File: backend/src/routes/triage.route.js

# Add new routes:
router.get("/status/:jobId", getTriageStatus);
router.get("/queue-stats", getQueueStats);
```

---

### STEP 4: Test Queue System (TOMORROW - 1 hour)

**4a. Start Services**
```bash
# Terminal 1: Python backend
cd backend
python ESI_Engine.py  # Port 5000

# Terminal 2: Node.js backend  
npm start  # Port 3000

# Terminal 3: Frontend (if needed)
npm run dev  # Port 5173
```

**4b. Manual Test via curl**
```bash
# Submit a triage request
curl -X POST http://localhost:5000/llm-triage/enqueue \
  -H "Content-Type: application/json" \
  -d '{
    "patient": {"age": 45, "gender": "male", "symptoms": "Chest pain"},
    "vitals": {"temp": 37, "bp_systolic": 150, "hr": 100, "o2": 95}
  }'

# Response (202 Accepted):
# {"job_id": "abc-123", "status": "queued", "queue_position": 2, ...}

# Poll status
curl http://localhost:5000/llm-triage/status/abc-123

# Response (pending):
# {"status": "pending", "queue_position": 1, "queue_wait_time": 5.3}

# Later response (complete):
# {"status": "complete", "result": {...all 14 clinical fields...}}
```

**4c. Run Concurrent Capacity Test**
```bash
# Run same test script as baseline
cd backend/ML
python test_concurrent_capacity.py

# Expected improvement:
# BEFORE: Concurrency 4 -> 75% success
# AFTER:  Concurrency 4 -> 95%+ success
#
# BEFORE: Concurrency 5 -> 50% success  
# AFTER:  Concurrency 5 -> 95%+ success
```

---

## 📂 Files to Create

1. **backend/ML/llm_request_queue.py** (450 lines)
   - Queue management, job tracking, deduplication
   
2. **backend/ML/llm_inference_retry.py** (300 lines)
   - JSON parsing, truncation repair, retry logic

3. **backend/ML/test_data_concurrent.json** (100 lines)
   - Test cases for load testing

4. **backend/ML/test_concurrent_capacity.py** (400 lines)
   - Load testing script

5. **backend/src/services/triageQueueService.js** (200 lines)
   - Node.js async queue client

### Files to Modify

1. **backend/ML/ESI_Engine.py**
   - Add imports (2 lines)
   - Add queue initialization (5 lines)
   - Move core triage logic to worker function (~100 lines refactored)
   - Replace /llm-triage endpoint with new async endpoints (80 lines new)

2. **backend/src/controllers/triage.controller.js**
   - Update runTriage() to enqueue instead of direct call (20 lines changed)
   - Add getTriageStatus() function (50 lines new)
   - Add getQueueStats() function (20 lines new)

3. **backend/src/routes/triage.route.js**
   - Add 2 new routes (3 lines)

4. **frontend/src/services/triageService.js** (NEW)
   - Service for polling triage status (optional, for better UX)

---

## ⚡ Quick Implementation Checklist

### Day 1: Baseline & Python Queue
- [ ] Create test_data_concurrent.json
- [ ] Create test_concurrent_capacity.py
- [ ] Run baseline test (note results)
- [ ] Create llm_request_queue.py
- [ ] Create llm_inference_retry.py
- [ ] Update ESI_Engine.py with queue integration
- [ ] Manual curl tests of /llm-triage/enqueue and /llm-triage/status

### Day 2: Node.js Integration & Validation
- [ ] Create triageQueueService.js
- [ ] Update triage.controller.js
- [ ] Update triage.route.js
- [ ] Test via Node.js backend
- [ ] Run concurrent capacity test again
- [ ] Compare metrics (should show 95%+ success at 5+ concurrent)

### Day 3: Frontend & Deployment
- [ ] Create triageService.js in frontend
- [ ] Update triage submission form to use new service
- [ ] Show queue position UI
- [ ] Show processing progress UI
- [ ] Deploy to staging
- [ ] Run production load test

---

## 📊 Expected Metrics After Phase 1

### Before (Current)
| Metric | Value |
|--------|-------|
| Concurrency Level | 1 |
| Success Rate | 100% |
| **Concurrency Level** | **4** |
| **Success Rate** | **75-90%** |
| **Concurrency Level** | **5** |
| **Success Rate** | **50%** ❌ |
| **Concurrency Level** | **10** |
| **Success Rate** | **20%** ❌ |
| Avg Inference Time | 22-28s |
| P95 Inference Time | 30-50s |
| Failure Reason | JSON truncation mid-field |

### After (With Queue + Retry)
| Metric | Value |
|--------|-------|
| Concurrency Level | 1 |
| Success Rate | 100% |
| **Concurrency Level** | **4** |
| **Success Rate** | **95%+** ✅ |
| **Concurrency Level** | **5** |
| **Success Rate** | **95%+** ✅ |
| **Concurrency Level** | **10** |
| **Success Rate** | **90%+** ✅ |
| Avg Inference Time | 20-25s (faster due to retry) |
| P95 Inference Time | 35-50s (slightly slower, but complete) |
| Queue Wait Time | 5-30s depending on position |
| Cache Hit Rate | ~15% (skips expensive inference) |

---

## 🔧 Common Issues & Fixes

### Issue 1: "Queue module not found"
```python
# Ensure llm_request_queue.py is in same directory as ESI_Engine.py
# cd backend/ML
# ls llm_request_queue.py  # Should exist
```

### Issue 2: "gguf_generate is not defined in worker function"
```python
# Make sure gguf_generate() is imported at top of ESI_Engine.py
# It's likely defined later in the file
# Move import or define: gguf_generate = _llm.create_completion (or equivalent)
```

### Issue 3: "Queue never processes requests"
```python
# Check that initialize_queue() is called at Flask startup
# Add to app.run():
#   initialize_queue()  # Before serve
# Or use @app.before_first_request decorator
```

### Issue 4: "Node.js can't reach Python queue endpoints"
```bash
# Ensure Python Flask is running on port 5000
netstat -an | grep 5000  # Check port

# Ensure PYTHON_URL in Node.js is correct
# backend/src/controllers/triage.controller.js:
# const PYTHON_URL = process.env.PYTHON_URL || 'http://localhost:5000';

# Test connectivity:
curl http://localhost:5000/llm-triage/stats
```

---

## 📚 Documentation Files Created

1. **LLM_INFERENCE_ARCHITECTURE_ANALYSIS.md** (70KB)
   - Complete architecture overview
   - Problem diagnosis
   - Proposed solution design
   
2. **PHASE_1_IMPLEMENTATION_GUIDE.md** (100KB)
   - Detailed step-by-step for measurement & Python queue
   - Full source code for queue modules
   - Test script with async handling

3. **PHASE_1_NODE_JS_INTEGRATION.md** (40KB)
   - Node.js service and controller updates
   - Frontend usage examples
   - All async API endpoints

4. **PHASE_1_QUICK_START_GUIDE.md** (THIS FILE)
   - Executive summary
   - Timeline
   - Checklist
   - Expected metrics

---

## 🚀 Next Immediate Steps (RIGHT NOW)

1. **Read the architecture analysis** (10 min)
   - Open: LLM_INFERENCE_ARCHITECTURE_ANALYSIS.md
   - Understand the problem and solution design

2. **Create test files** (30 min)
   - Create: test_data_concurrent.json
   - Create: test_concurrent_capacity.py
   - Run baseline test

3. **Review baseline results** (10 min)
   - At what concurrency level does success drop?
   - What error types appear?
   - Save the CSV for comparison

4. **Start Python queue implementation** (2 hours)
   - Create: llm_request_queue.py
   - Create: llm_inference_retry.py
   - Update: ESI_Engine.py

5. **Test with curl** (30 min)
   - Manual enqueue test
   - Manual status check test
   - Verify Python queue works

**Total time investment for Phase 1: ~2-3 days**
**Expected ROI: 50% → 95%+ success rate under load** ✅

