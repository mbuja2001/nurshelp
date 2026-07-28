# PART 4: UPDATE NODE.JS BACKEND FOR ASYNC QUEUE API

## Step 4.1: Create Async Triage Helper Module

**File**: `backend/src/services/triageQueueService.js` (NEW FILE)

```javascript
/**
 * Service to interact with Python backend's async queue API
 * Handles job enqueuing, polling, and result retrieval
 */

import axios from 'axios';

const PYTHON_URL = process.env.PYTHON_URL || 'http://localhost:5000';
const PYTHON_TRIAGE_ENQUEUE_URL = `${PYTHON_URL}/llm-triage/enqueue`;
const PYTHON_TRIAGE_STATUS_URL = (jobId) => `${PYTHON_URL}/llm-triage/status/${jobId}`;
const PYTHON_QUEUE_STATS_URL = `${PYTHON_URL}/llm-triage/stats`;

/**
 * Submit triage request to queue and get jobId
 * @param {Object} triageData - Patient vitals, symptoms, etc
 * @returns {Promise<string>} - jobId for polling
 */
export const enqueueTriage = async (triageData) => {
  try {
    console.log('[triageQueueService] 📨 Enqueuing triage request...');
    
    const response = await axios.post(
      PYTHON_TRIAGE_ENQUEUE_URL,
      triageData,
      {
        timeout: 5000  // Enqueue should be fast
      }
    );
    
    const { job_id, queue_position, status_url } = response.data;
    console.log(`[triageQueueService] ✅ Job ${job_id} enqueued (position ${queue_position})`);
    
    return {
      jobId: job_id,
      queuePosition: queue_position,
      statusUrl: status_url
    };
  } catch (error) {
    console.error('[triageQueueService] ❌ Enqueue error:', error.message);
    throw error;
  }
};

/**
 * Poll job status until complete or timeout
 * @param {string} jobId - Job ID from enqueue
 * @param {number} maxWaitMs - Max time to wait (default 120s)
 * @param {number} pollIntervalMs - Interval between polls (default 1s)
 * @returns {Promise<Object>} - Complete triage result
 */
export const pollTriageStatus = async (
  jobId,
  maxWaitMs = 120000,
  pollIntervalMs = 1000
) => {
  const startTime = Date.now();
  let lastQueuePosition = null;
  
  while (Date.now() - startTime < maxWaitMs) {
    try {
      const response = await axios.get(
        PYTHON_TRIAGE_STATUS_URL(jobId),
        { timeout: 5000 }
      );
      
      const { status, result, error, queue_position, processing_time } = response.data;
      
      // Log position changes
      if (queue_position && queue_position !== lastQueuePosition) {
        lastQueuePosition = queue_position;
        console.log(`[triageQueueService] 📍 Job ${jobId} position in queue: ${queue_position}`);
      }
      
      if (status === 'complete') {
        const elapsed = Date.now() - startTime;
        console.log(`[triageQueueService] ✅ Job ${jobId} complete (${elapsed}ms total)`);
        return {
          status: 'complete',
          result,
          totalTime: elapsed
        };
      }
      
      if (status === 'failed') {
        const elapsed = Date.now() - startTime;
        console.error(`[triageQueueService] ❌ Job ${jobId} failed: ${error}`);
        throw new Error(`Triage job failed: ${error}`);
      }
      
      if (status === 'processing') {
        console.log(`[triageQueueService] 🔄 Job ${jobId} processing (${processing_time?.toFixed(1)}s so far)...`);
      }
      
      // Sleep before next poll
      await new Promise(resolve => setTimeout(resolve, pollIntervalMs));
      
    } catch (error) {
      if (error.response?.status === 404) {
        throw new Error(`Job ${jobId} not found`);
      }
      console.error(`[triageQueueService] ⚠️  Poll error: ${error.message}`);
      // Continue polling on transient errors
      await new Promise(resolve => setTimeout(resolve, pollIntervalMs));
    }
  }
  
  // Timeout
  throw new Error(`Triage job ${jobId} exceeded max wait time (${maxWaitMs}ms)`);
};

/**
 * Get queue statistics
 * @returns {Promise<Object>} - Queue stats
 */
export const getQueueStats = async () => {
  try {
    const response = await axios.get(PYTHON_QUEUE_STATS_URL, { timeout: 5000 });
    return response.data;
  } catch (error) {
    console.error('[triageQueueService] ❌ Stats error:', error.message);
    return null;
  }
};

/**
 * Submit triage and wait for result (convenience function)
 * @param {Object} triageData - Patient data
 * @returns {Promise<Object>} - Complete result
 */
export const triageAndWait = async (triageData, options = {}) => {
  const { maxWaitMs = 120000, pollIntervalMs = 1000 } = options;
  
  try {
    // Enqueue
    const { jobId } = await enqueueTriage(triageData);
    
    // Poll until complete
    const { result } = await pollTriageStatus(jobId, maxWaitMs, pollIntervalMs);
    
    return result;
  } catch (error) {
    console.error('[triageQueueService] ❌ Triage failed:', error.message);
    throw error;
  }
};
```

## Step 4.2: Update Triage Controller

**File**: `backend/src/controllers/triage.controller.js`

**Location to modify**: The `runTriage()` function (around line ~210)

**Find this code**:
```javascript
export const runTriage = async (req, res) => {
  try {
    const startTime = Date.now();
    const payload = req.body;

    // ... validation code ...

    const enginePayload = {
      patient: payload.patient,
      vitals: payload.vitals,
      transcript: transcriptArray,
      computeMode,
      staffScheduling: payload.staffScheduling || {},
      encounterId: payload.encounterId
    };

    console.log(`[Triage] Processing with computeMode: ${computeMode}`);
    
    // ❌ OLD: Direct synchronous call
    const pythonRes = await axios.post(PYTHON_URL, enginePayload, {
      timeout: 0,  // No timeout
      headers: { "Content-Type": "application/json" }
    });
    
    triage = pythonRes.data;
    // ... create encounter ...
    return res.status(201).json({ ... });
```

**Replace with**:
```javascript
import { enqueueTriage, pollTriageStatus } from '../services/triageQueueService.js';

export const runTriage = async (req, res) => {
  try {
    const startTime = Date.now();
    const payload = req.body;

    if (!payload || !payload.patient || !payload.vitals) {
      return res.status(400).json({ message: "Invalid payload" });
    }

    const computeMode = payload.computeMode || 'HYBRID';
    const nurseId = req.user?._id || null;
    const transcriptArray = parseTranscript(payload.transcript);

    // Build payload for Python queue
    const enginePayload = {
      patient: payload.patient,
      vitals: payload.vitals,
      transcript: transcriptArray,
      computeMode,
      staffScheduling: payload.staffScheduling || {},
      encounterId: payload.encounterId,
      chiefComplaint: payload.chiefComplaint || payload.patient.symptoms || '',
      triageNotes: payload.triageNotes || '',
      medicalHistory: payload.medicalHistory || ''
    };

    console.log(`[Triage] 📨 Enqueuing triage request (${computeMode} mode)`);
    
    // ✅ NEW: Enqueue to async queue
    try {
      const { jobId, queuePosition } = await enqueueTriage(enginePayload);
      
      // Return immediately with jobId (202 Accepted)
      return res.status(202).json({
        message: "Triage queued for processing",
        jobId,
        queuePosition,
        statusUrl: `/api/triage/status/${jobId}`,
        estimatedWaitTime: queuePosition * 20  // Rough estimate: 20s per request
      });
    
    } catch (queueError) {
      console.error('[Triage] Queue enqueue failed:', queueError.message);
      
      // Fallback: try direct synchronous call (for backward compatibility)
      console.log('[Triage] ⚠️  Falling back to direct inference...');
      
      const pythonRes = await axios.post(PYTHON_URL, enginePayload, {
        timeout: 120000,  // 2 min timeout
        headers: { "Content-Type": "application/json" }
      });
      
      const triage = pythonRes.data;
      
      // Create encounter with result
      const encounter = await Encounter.create({
        nurse_id: nurseId,
        patient: payload.patient,
        transcript: transcriptArray,
        vitals: payload.vitals,
        triage: { 
          ...triage, 
          severity: triage.ESI,
          computeMode
        },
        status: nurseId ? "pending" : "unassigned"
      });
      
      return res.status(201).json({ 
        message: "Triage completed (fallback mode)",
        encounter
      });
    }
    
  } catch (error) {
    console.error('[Triage] Fatal error:', error.message);
    return res.status(500).json({ 
      error: error.message,
      message: "Triage processing failed"
    });
  }
};
```

## Step 4.3: Add New Status Check Endpoint

**File**: `backend/src/controllers/triage.controller.js`

**Add this new function** (before or after `runTriage`):

```javascript
/**
 * GET /api/triage/status/:jobId
 * Check status of a queued triage request
 * Frontend polls this endpoint to track progress
 */
export const getTriageStatus = async (req, res) => {
  try {
    const { jobId } = req.params;
    
    if (!jobId) {
      return res.status(400).json({ error: "jobId required" });
    }
    
    console.log(`[Triage Status] 📍 Checking status for job ${jobId}`);
    
    // Get status from Python backend
    const PYTHON_STATUS_URL = `${PYTHON_URL}/llm-triage/status/${jobId}`;
    
    const statusRes = await axios.get(PYTHON_STATUS_URL, { timeout: 5000 });
    const { status, result, error, queue_position, queue_wait_time, processing_time, total_time } = statusRes.data;
    
    if (status === 'not_found') {
      return res.status(404).json({ error: `Job ${jobId} not found` });
    }
    
    // If complete, create encounter record and return
    if (status === 'complete' && result) {
      console.log(`[Triage Status] ✅ Job ${jobId} complete - creating encounter`);
      
      const encounter = await Encounter.create({
        jobId,
        nurse_id: req.user?._id || null,
        patient: result.patient || {},
        triage: result,
        status: "pending",
        aiProcessingTime: total_time
      });
      
      return res.json({
        status: 'complete',
        encounter,
        metrics: {
          queueWaitTime: queue_wait_time,
          processingTime: processing_time,
          totalTime: total_time
        }
      });
    }
    
    if (status === 'failed') {
      console.error(`[Triage Status] ❌ Job ${jobId} failed: ${error}`);
      return res.status(400).json({
        status: 'failed',
        error
      });
    }
    
    // Still pending or processing
    return res.json({
      status,
      queuePosition: queue_position,
      queueWaitTime: queue_wait_time,
      processingTime: processing_time,
      message: status === 'pending'
        ? `Job queued (position ${queue_position})`
        : `Job processing (${processing_time?.toFixed(1)}s elapsed)`
    });
    
  } catch (error) {
    console.error('[Triage Status] Error:', error.message);
    
    if (error.response?.status === 404) {
      return res.status(404).json({ error: "Job not found" });
    }
    
    return res.status(500).json({ 
      error: error.message,
      jobId: req.params.jobId
    });
  }
};

/**
 * GET /api/triage/queue-stats
 * Get current queue statistics (for monitoring UI)
 */
export const getQueueStats = async (req, res) => {
  try {
    const PYTHON_STATS_URL = `${PYTHON_URL}/llm-triage/stats`;
    
    const statsRes = await axios.get(PYTHON_STATS_URL, { timeout: 5000 });
    
    return res.json(statsRes.data);
    
  } catch (error) {
    console.error('[Queue Stats] Error:', error.message);
    return res.status(500).json({ error: error.message });
  }
};
```

## Step 4.4: Update Routes

**File**: `backend/src/routes/triage.route.js`

**Add these new routes** (after existing routes):

```javascript
import { getTriageStatus, getQueueStats } from '../controllers/triage.controller.js';

// Existing routes remain unchanged...
router.post("/run", runTriage);
router.get("/staff", getStaffForTriage);
router.post("/", submitManualTriage);
// ... etc ...

// ✅ ADD THESE NEW ROUTES:

/**
 * GET /api/triage/status/:jobId
 * Check status of queued triage request
 */
router.get("/status/:jobId", getTriageStatus);

/**
 * GET /api/triage/queue-stats
 * Get queue statistics
 */
router.get("/queue-stats", getQueueStats);

export default router;
```

---

## Step 4.5: Update Frontend to Use New Async API

**File**: `frontend/src/services/triageService.js` (or equivalent)

```javascript
/**
 * Service for submitting triage and polling results
 */

import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000/api';

export const submitTriageAndWait = async (triageData, onProgress = null) => {
  try {
    // Step 1: Submit triage request
    console.log('[triageService] 📨 Submitting triage request...');
    
    const submitRes = await axios.post(
      `${API_URL}/triage/run`,
      triageData,
      { timeout: 5000 }
    );
    
    const { jobId, queuePosition, estimatedWaitTime } = submitRes.data;
    console.log(`[triageService] ✅ Job ${jobId} queued (position ${queuePosition})`);
    
    if (onProgress) {
      onProgress({
        status: 'queued',
        jobId,
        queuePosition,
        message: `Waiting in queue... Position ${queuePosition}`
      });
    }
    
    // Step 2: Poll for result
    const maxAttempts = 180;  // 3 minutes max (1s polling)
    let attempts = 0;
    
    while (attempts < maxAttempts) {
      await new Promise(resolve => setTimeout(resolve, 1000));  // Wait 1s
      attempts++;
      
      try {
        const statusRes = await axios.get(
          `${API_URL}/triage/status/${jobId}`,
          { timeout: 5000 }
        );
        
        const { status, encounter, error, queuePosition: currentPosition } = statusRes.data;
        
        if (status === 'complete') {
          console.log(`[triageService] ✅ Triage complete (${attempts}s)`);
          return {
            success: true,
            encounter
          };
        }
        
        if (status === 'failed') {
          console.error(`[triageService] ❌ Triage failed: ${error}`);
          throw new Error(error);
        }
        
        // Update progress
        if (onProgress) {
          onProgress({
            status,
            jobId,
            queuePosition: currentPosition,
            elapsedSeconds: attempts,
            message: status === 'pending'
              ? `Waiting in queue... Position ${currentPosition}`
              : `Processing triage (${attempts}s elapsed)...`
          });
        }
        
      } catch (pollError) {
        if (pollError.response?.status === 404) {
          // Job not found, might be race condition
          await new Promise(resolve => setTimeout(resolve, 1000));
          continue;
        }
        throw pollError;
      }
    }
    
    throw new Error(`Triage processing timeout (${maxAttempts}s)`);
    
  } catch (error) {
    console.error('[triageService] ❌ Triage submission error:', error.message);
    return {
      success: false,
      error: error.message
    };
  }
};

/**
 * Get current queue statistics
 */
export const getQueueStats = async () => {
  try {
    const res = await axios.get(`${API_URL}/triage/queue-stats`, { timeout: 5000 });
    return res.data;
  } catch (error) {
    console.error('[triageService] Error getting queue stats:', error.message);
    return null;
  }
};
```

---

## Step 4.6: Usage in Components

**Example React Component**:

```jsx
import { submitTriageAndWait, getQueueStats } from '../services/triageService.js';
import { useState } from 'react';

export function TriageForm() {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [progress, setProgress] = useState(null);
  
  const handleSubmit = async (formData) => {
    setIsSubmitting(true);
    setProgress({ status: 'queued', message: 'Submitting triage...' });
    
    try {
      const result = await submitTriageAndWait(formData, (progressUpdate) => {
        setProgress(progressUpdate);
      });
      
      if (result.success) {
        // Handle success
        console.log('Triage complete:', result.encounter);
        setProgress({ status: 'complete', message: 'Triage complete!' });
      } else {
        // Handle error
        setProgress({ status: 'error', message: result.error });
      }
    } finally {
      setIsSubmitting(false);
    }
  };
  
  return (
    <div>
      {/* Form inputs ... */}
      
      {isSubmitting && progress && (
        <div className="progress-indicator">
          {progress.status === 'queued' && (
            <div>
              🔄 {progress.message}
              {progress.queuePosition && ` (Position ${progress.queuePosition})`}
            </div>
          )}
          {progress.status === 'processing' && (
            <div>⏳ {progress.message}</div>
          )}
          {progress.status === 'complete' && (
            <div>✅ {progress.message}</div>
          )}
          {progress.status === 'error' && (
            <div>❌ {progress.message}</div>
          )}
        </div>
      )}
    </div>
  );
}
```

---

## Summary: What Changed

### BEFORE (Synchronous, No Queue)
```
Frontend POST /api/triage/run
    ↓ (waits indefinitely)
Node.js axios.post() to http://localhost:5000/llm-triage
    ↓ (blocking, locks connection)
Python GGUF inference (140-1000s)
    ↓ Direct call, no queue
Result returned to frontend
```

### AFTER (Async with Queue)
```
Frontend POST /api/triage/run
    ↓ Returns IMMEDIATELY (202 Accepted)
Node.js enqueues to Python queue
    ↓ Connection released
Frontend polls GET /api/triage/status/{jobId} (1s interval)
    ↓ Shows queue position, processing status
Python queue processes 1-2 requests at a time
    ↓ Prevents resource starvation
GGUF inference with retry logic (dynamic max_tokens)
    ↓ On JSON truncation, retry with smaller budget
Result stored in job
    ↓ Frontend receives on next poll
```

---

## Next Steps

1. ✅ Copy Python queue modules (llm_request_queue.py, llm_inference_retry.py)
2. ✅ Update ESI_Engine.py with new endpoints
3. ✅ Add Node.js service and controller updates
4. ✅ Test baseline capacity (Part 1)
5. 🔄 Run baseline concurrent capacity test
6. 🔄 Deploy queue system
7. 🔄 Run post-implementation capacity test
8. 🔄 Compare metrics (50% success → 95%+ success)

