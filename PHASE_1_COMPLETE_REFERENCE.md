# Phase 1 Implementation: Complete Reference Guide

## Overview: What This Achieves

**Current Problem:**
- Concurrent triage requests cause JSON truncation (50% failure rate at 5+ concurrent)
- No request queue = all requests hammer GGUF simultaneously
- max_tokens=3100 insufficient under concurrent load
- Mid-field truncation can't be repaired

**Phase 1 Solution:**
- ✅ Request queue (max 2 workers) prevents resource starvation
- ✅ Retry logic with dynamic max_tokens (3100→2500→2000→1500)
- ✅ JSON completeness validation gates
- ✅ Request deduplication (30s cache)
- ✅ Job status tracking (async API)
- ✅ Emergency prioritization (red-flag cases jump queue)

**Expected Result:**
- Success rate: 50% → 95%+
- Concurrent capacity: 5 → 10+ simultaneous requests
- Better UX: Queue position visibility, progress tracking

---

## Implementation Phases

### Phase 1a: Measure Baseline (1 day)
**Goal**: Establish performance metrics before queue implementation
**Files**: test_concurrent_capacity.py, test_data_concurrent.json
**Output**: concurrent_capacity_results_YYYYMMDD_HHMMSS.csv
**Key Metric**: At what concurrency does success rate drop?

### Phase 1b: Python Backend Queue (1 day)
**Goal**: Implement request queue + retry logic
**Files**: llm_request_queue.py, llm_inference_retry.py, ESI_Engine.py (modified)
**New Endpoints**: 
  - POST /llm-triage/enqueue (return jobId)
  - GET /llm-triage/status/{jobId} (poll status)
  - GET /llm-triage/stats (queue info)

### Phase 1c: Node.js Integration (1 day)
**Goal**: Update Node.js to use async queue API
**Files**: triageQueueService.js, triage.controller.js (modified), triage.route.js (modified)
**Changes**: runTriage() returns jobId immediately, polling for result

### Phase 1d: Frontend & Validation (0.5 day)
**Goal**: Test and validate complete queue system
**Tests**: Concurrent capacity test again (should show 95%+ success)
**Metrics**: Compare before/after results

---

## Detailed File Structure

```
backend/
├── ML/
│   ├── ESI_Engine.py (MODIFY)
│   │   ├── Add: from llm_request_queue import get_queue
│   │   ├── Add: from llm_inference_retry import gguf_generate_with_retry
│   │   ├── Add: initialize_queue() function
│   │   ├── Add: process_triage_request_worker() function
│   │   ├── Replace: @app.route('/llm-triage') → /llm-triage/enqueue
│   │   ├── Add: @app.route('/llm-triage/status/<job_id>')
│   │   └── Add: @app.route('/llm-triage/stats')
│   │
│   ├── llm_request_queue.py (NEW - 450 lines)
│   │   ├── TriageRequest class
│   │   ├── RequestCache class
│   │   ├── TriageRequestQueue class
│   │   └── Global get_queue() function
│   │
│   ├── llm_inference_retry.py (NEW - 300 lines)
│   │   ├── validate_json_completeness()
│   │   ├── extract_json_from_output()
│   │   ├── repair_incomplete_json()
│   │   └── gguf_generate_with_retry()
│   │
│   ├── test_data_concurrent.json (NEW - 100 lines)
│   │   └── 4 test cases (chest pain, meningitis, stroke, abd pain)
│   │
│   └── test_concurrent_capacity.py (NEW - 400 lines)
│       ├── ConcurrencyTester class
│       └── Test cases: concurrency 1, 2, 3, 4, 5, 10
│
├── src/
│   ├── controllers/
│   │   └── triage.controller.js (MODIFY)
│   │       ├── Update: runTriage() → enqueue instead of direct call
│   │       ├── Add: getTriageStatus() → GET /api/triage/status/:jobId
│   │       └── Add: getQueueStats() → GET /api/triage/queue-stats
│   │
│   ├── routes/
│   │   └── triage.route.js (MODIFY)
│   │       ├── Import: { getTriageStatus, getQueueStats }
│   │       ├── Add: router.get("/status/:jobId", getTriageStatus)
│   │       └── Add: router.get("/queue-stats", getQueueStats)
│   │
│   └── services/
│       └── triageQueueService.js (NEW - 200 lines)
│           ├── enqueueTriage(data) → jobId
│           ├── pollTriageStatus(jobId) → result
│           ├── getQueueStats() → stats
│           └── triageAndWait(data) → convenience wrapper
│
└── .env (UPDATE)
    └── PYTHON_URL=http://localhost:5000  # Ensure set
```

---

## Step-by-Step Implementation

### Step 1: Baseline Measurement (1 hour)

**A. Create Test Data**
```bash
# File: backend/ML/test_data_concurrent.json
{
  "test_cases": [
    {
      "id": "case_1",
      "patient": {"age": 45, "gender": "male", "symptoms": "Chest pain + SOB"},
      "vitals": {"temp": 36.8, "bp_systolic": 155, "hr": 105, "o2": 92}
    },
    ... 3 more test cases ...
  ]
}
```

**B. Create Test Script**
```bash
# File: backend/ML/test_concurrent_capacity.py
# Submits N concurrent requests (1, 2, 3, 4, 5, 10)
# Measures: success rate, inference time, error types
# Outputs: CSV with results
```

**C. Run Baseline Test**
```bash
# Terminal 1: Start Python backend
cd backend && python ESI_Engine.py

# Terminal 2: Run test
cd backend/ML && python test_concurrent_capacity.py

# Output: concurrent_capacity_results_20260621_HHMMSS.csv
# Note: At which concurrency level does success drop below 90%?
```

---

### Step 2: Python Queue Implementation (2-3 hours)

**A. Create Queue Module**
```bash
# File: backend/ML/llm_request_queue.py (450 lines)

# Classes:
# - TriageRequest: job metadata, timing, status
# - RequestCache: 30s TTL deduplication
# - TriageRequestQueue: PriorityQueue with worker thread

# Key Features:
# - Max 2 concurrent workers (prevent CPU starvation)
# - Priority queue (emergencies first)
# - Deduplication (same patient vitals = skip re-triage)
# - Job status tracking (pending/processing/complete/failed)
# - Request caching (30s TTL)
```

**B. Create Retry Module**
```bash
# File: backend/ML/llm_inference_retry.py (300 lines)

# Functions:
# - validate_json_completeness(json_str) → bool
#   Check all 14 clinical fields present and non-empty
# 
# - extract_json_from_output(raw_output) → str
#   Smart extraction using brace-depth tracking
#   Handles markdown fences, partial JSON
#
# - repair_incomplete_json(raw_json, depth, in_string) → str
#   Close unclosed strings and braces
#
# - gguf_generate_with_retry(func, prompt, max_tokens, max_retries) → (dict, bool)
#   Retry with dynamic max_tokens: 3100 → 2500 → 2000 → 1500
#   Returns (parsed_json, is_complete_bool)
```

**C. Modify ESI_Engine.py**
```python
# Add at top:
from llm_request_queue import get_queue
from llm_inference_retry import gguf_generate_with_retry

# Add before Flask routes (line ~8100):
_triage_queue = None

def initialize_queue():
    global _triage_queue
    _triage_queue = get_queue()
    _triage_queue.start(process_triage_request_worker)
    print('[ESI_Engine] ✅ Triage queue initialized')

def process_triage_request_worker(request_obj):
    """Worker function called by queue thread"""
    # MOVE all current /llm-triage logic here
    # Instead of returning HTTP response, return dict
    # Queue will store result and update job status
    
    # Key changes:
    # - Use gguf_generate_with_retry() instead of gguf_generate()
    # - Return complete result dict with all 14 clinical fields
    # - Let queue handle job status updates

# Replace /llm-triage endpoint (line ~8200):
@app.route('/llm-triage/enqueue', methods=['POST'])
def enqueue_triage_request():
    """Enqueue request and return jobId immediately"""
    data = request.get_json()
    # Determine priority (1=urgent if red flags, 5=normal)
    job_id = _triage_queue.enqueue(data, priority=...)
    return {
        'job_id': job_id,
        'status': 'queued',
        'status_url': f'/llm-triage/status/{job_id}'
    }, 202  # 202 Accepted

@app.route('/llm-triage/status/<job_id>', methods=['GET'])
def get_triage_status(job_id):
    """Poll job status"""
    status_info = _triage_queue.get_status(job_id)
    if status_info['status'] == 'complete':
        return status_info, 200
    elif status_info['status'] == 'failed':
        return status_info, 400
    else:  # pending or processing
        return status_info, 200

@app.route('/llm-triage/stats', methods=['GET'])
def get_queue_stats():
    """Get queue statistics"""
    return _triage_queue.get_queue_stats(), 200

# In Flask startup:
initialize_queue()  # Start queue worker before serving
```

---

### Step 3: Node.js Integration (1-2 hours)

**A. Create Queue Service**
```bash
# File: backend/src/services/triageQueueService.js (200 lines)

# Functions:
# - enqueueTriage(triageData) → {jobId, queuePosition}
# - pollTriageStatus(jobId, maxWaitMs, pollIntervalMs) → {status, result}
# - getQueueStats() → {pending, processing, complete, failed}
# - triageAndWait(triageData) → complete result (convenience)
```

**B. Modify Triage Controller**
```javascript
// File: backend/src/controllers/triage.controller.js

import { enqueueTriage, pollTriageStatus } from '../services/triageQueueService.js';

// Update existing runTriage():
export const runTriage = async (req, res) => {
    const payload = req.body;
    
    // OLD CODE (replaced):
    // const pythonRes = await axios.post(PYTHON_URL, enginePayload, {timeout: 0});
    // return res.json({encounter: ...});
    
    // NEW CODE:
    try {
        const {jobId, queuePosition} = await enqueueTriage(enginePayload);
        return res.status(202).json({
            jobId,
            queuePosition,
            statusUrl: `/api/triage/status/${jobId}`
        });
    } catch (error) {
        // Fallback to direct call if queue unavailable
        // ... existing fallback code ...
    }
};

// Add new function:
export const getTriageStatus = async (req, res) => {
    const {jobId} = req.params;
    const statusInfo = await axios.get(
        `http://localhost:5000/llm-triage/status/${jobId}`
    );
    
    if (statusInfo.data.status === 'complete') {
        // Create encounter from result
        const encounter = await Encounter.create({
            jobId,
            triage: statusInfo.data.result,
            ...
        });
        return res.json({status: 'complete', encounter});
    }
    return res.json(statusInfo.data);
};

// Add new function:
export const getQueueStats = async (req, res) => {
    const statsRes = await axios.get(
        'http://localhost:5000/llm-triage/stats'
    );
    return res.json(statsRes.data);
};
```

**C. Update Routes**
```javascript
// File: backend/src/routes/triage.route.js

import { getTriageStatus, getQueueStats } from '../controllers/triage.controller.js';

// Add routes:
router.get("/status/:jobId", getTriageStatus);
router.get("/queue-stats", getQueueStats);
```

---

### Step 4: Frontend (Optional - Better UX)

**A. Create Frontend Service**
```javascript
// File: frontend/src/services/triageService.js

export const submitTriageAndWait = async (triageData, onProgress) => {
    // Enqueue
    const submitRes = await axios.post('/api/triage/run', triageData);
    const {jobId} = submitRes.data;
    
    // Poll status
    const maxAttempts = 180;  // 3 min
    for (let i = 0; i < maxAttempts; i++) {
        await sleep(1000);
        const statusRes = await axios.get(`/api/triage/status/${jobId}`);
        
        onProgress?.(statusRes.data);  // Update UI
        
        if (statusRes.data.status === 'complete') {
            return statusRes.data.encounter;
        }
    }
    throw new Error(`Timeout waiting for job ${jobId}`);
};
```

**B. Update Component**
```jsx
// Show queue position while waiting
{progress.status === 'pending' && (
    <div>🔄 Waiting in queue (position {progress.queuePosition})</div>
)}

// Show processing progress
{progress.status === 'processing' && (
    <div>⏳ Processing triage ({progress.processingTime}s elapsed)</div>
)}

// Show result when complete
{progress.status === 'complete' && (
    <TriageResult encounter={progress.encounter} />
)}
```

---

### Step 5: Validation & Testing (1 hour)

**A. Manual curl Tests**
```bash
# Test enqueue
curl -X POST http://localhost:5000/llm-triage/enqueue \
  -H "Content-Type: application/json" \
  -d '{"patient":{"age":45},"vitals":{"temp":37}}'

# Response: {"job_id": "abc-123", "status": "queued"}

# Test status
curl http://localhost:5000/llm-triage/status/abc-123

# Response: {"status": "pending", "queue_position": 1}

# Later: {"status": "complete", "result": {...}}

# Test stats
curl http://localhost:5000/llm-triage/stats

# Response: {"pending": 1, "processing": 1, "complete": 5, ...}
```

**B. Run Concurrent Capacity Test Again**
```bash
cd backend/ML
python test_concurrent_capacity.py

# Expected output AFTER queue implementation:
# Concurrency 1: 100% success, 22-25s
# Concurrency 2: 100% success, 22-25s
# Concurrency 3: 100% success, 23-26s
# Concurrency 4: 95%+ success, 23-27s (vs 75% before)
# Concurrency 5: 95%+ success, 24-28s (vs 50% before)
# Concurrency 10: 90%+ success, 25-30s (vs 20% before)
```

**C. Compare Metrics**
```
Metric                  BEFORE      AFTER
Success @ Concurrency 1: 100%        100%
Success @ Concurrency 4: 75-90%      95%+ ✅
Success @ Concurrency 5: 50%         95%+ ✅
Success @ Concurrency 10: 20%        90%+ ✅
Avg Inference Time:      22-28s      20-25s ✅
P95 Inference Time:      30-50s      35-50s (OK, was stalling)
Queue Wait Time (p50):   N/A         2-5s
Queue Wait Time (p95):   N/A         10-15s
Cache Hit Rate:          0%          15-20% ✅
```

---

## Key Implementation Details

### Why Max 2 Workers?
- Hardware: 6 physical cores available
- Reserve: 1 core for OS, 1 core for other tasks
- GGUF needs: ~4-5 cores for parallel inference
- Result: 2 concurrent GGUF inferences = ~8-10 cores used safely

### Why 3100 → 2500 → 2000 → 1500?
- 3100 tokens: Full clinical output with all arrays (target)
- 2500 tokens: Compress arrays, reduce verbosity
- 2000 tokens: Minimal viable output
- 1500 tokens: Last resort, truncate gracefully

### Why 30s Cache TTL?
- Typical re-triage (user submitted twice): <10s
- Dashboard refresh cycle: ~5s
- Allows 3-6 duplicate submissions to be cached
- Saves ~15-20% of expensive GGUF calls

### Priority Queue Logic
- Red flag keywords (chest pain, meningitis, stroke, etc) → priority 1
- Emergency vitals (SBP>180, HR>120, SpO2<90) → priority 1
- Normal triage → priority 5
- FIFO within same priority level

---

## Troubleshooting Guide

### "Queue not processing requests"
**Symptom**: Jobs stuck in "pending" state
**Cause**: initialize_queue() not called, worker thread not started
**Fix**:
```python
# In ESI_Engine.py, ensure called at startup:
if __name__ == '__main__':
    initialize_queue()  # Start queue worker
    app.run(host='0.0.0.0', port=5000)
```

### "gguf_generate_with_retry not found"
**Symptom**: ImportError in process_triage_request_worker()
**Cause**: llm_inference_retry.py in wrong location
**Fix**:
```bash
# Ensure file exists:
ls backend/ML/llm_inference_retry.py

# Check import path:
from llm_inference_retry import gguf_generate_with_retry
```

### "Node.js can't reach queue endpoints"
**Symptom**: axios.post to /llm-triage/enqueue fails
**Cause**: Python not running, port 5000 blocked
**Fix**:
```bash
# Check Python is running
ps aux | grep ESI_Engine.py

# Check port available
netstat -an | grep 5000
netstat -an | grep LISTEN

# Test connectivity
curl http://localhost:5000/llm-triage/stats
```

### "JSON still truncates even with retry"
**Symptom**: Incomplete JSON even at max_tokens=1500
**Cause**: Prompt too large, fields too verbose
**Solution** (Phase 2):
- Reduce prompt size
- Compress prompt template
- Use shorter field names
- Reduce n_ctx from 1200 to 800

---

## Success Criteria

### Phase 1 is COMPLETE when:

✅ **Measurement Phase**
- [ ] test_concurrent_capacity.py runs successfully
- [ ] Baseline results saved to CSV
- [ ] At what concurrency does success drop?

✅ **Python Queue**
- [ ] llm_request_queue.py created and tested
- [ ] llm_inference_retry.py created and tested
- [ ] ESI_Engine.py modified with queue integration
- [ ] Manual curl tests pass
- [ ] Queue stats endpoint works

✅ **Node.js Integration**
- [ ] triageQueueService.js works
- [ ] triage.controller.js enqueues instead of direct call
- [ ] getTriageStatus() endpoint works
- [ ] Frontend receives jobId + can poll status

✅ **Validation**
- [ ] Concurrent capacity test shows 95%+ success at 5+ concurrent
- [ ] No JSON truncation errors
- [ ] Cache hits working (15%+ dedup rate)
- [ ] Queue stats show proper worker distribution

---

## What NOT to Do in Phase 1

❌ Don't modify prompt engineering (Phase 2)
❌ Don't add GPU support (Phase 2)
❌ Don't change max_tokens budget scaling algorithm (Phase 2)
❌ Don't implement distributed queue (Phase 2)
❌ Don't add request analytics/monitoring (Phase 2)

**Phase 1 ONLY adds**: Queue + Retry Logic ✅

---

## Timeline Summary

```
Day 1 (4 hours)
├─ Baseline testing (1h)
├─ Python queue modules (2h)
└─ Quick validation (1h)

Day 2 (3 hours)
├─ Node.js integration (1.5h)
├─ Route updates (0.5h)
└─ Testing (1h)

Day 3 (2 hours)
├─ Frontend enhancements (0.5h)
├─ Concurrent capacity test (1h)
└─ Metrics comparison (0.5h)

Total: ~2-3 days for Phase 1
Expected ROI: 50% → 95%+ success rate
```

---

## Next Steps (Right Now!)

1. **Read**: LLM_INFERENCE_ARCHITECTURE_ANALYSIS.md (understand problem)
2. **Read**: PHASE_1_QUICK_START.md (high-level plan)
3. **Read**: This file (implementation details)
4. **Create**: test_data_concurrent.json (30 min)
5. **Create**: test_concurrent_capacity.py (30 min)
6. **Run**: Baseline test (30 min) → Save results
7. **Create**: llm_request_queue.py (1 hour)
8. **Create**: llm_inference_retry.py (1 hour)
9. **Modify**: ESI_Engine.py (1 hour)
10. **Test**: Manual curl tests (30 min)

**Start now. Finish Phase 1 by end of week.** 🚀

