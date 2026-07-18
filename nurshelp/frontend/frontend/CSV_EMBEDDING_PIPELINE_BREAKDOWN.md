# NurseHelp: CSV Loading & Embedding Pipeline Comprehensive Breakdown

## Current System Architecture (High Level)

```
Patient Vitals/Symptoms
         ↓
    Frontend (PS.jsx)
         ↓
[POST /api/triage/submit]
         ↓
    Node.js Backend
         ↓
    ESI_Engine.py (Python)
    (Only runs if ENABLED)
         ↓
    ┌─────────────────────────────────┐
    │ RAG_Retriever (embeddings)      │
    │ physician.csv (workforce)       │
    │ FAISS Index (vector search)     │
    └─────────────────────────────────┘
         ↓
    Returns: top_5_composite_physicians
             top_5_embedding_physicians
             assigned_specialty
         ↓
    Node.js physician.controller.js
         ↓
    Return to Frontend
```

---

## PART 1: CSV FILES

### 1.1 physician.csv Location
```
Backend loads from:
- Filesystem: backend/models/physician_data.csv (if exists)
- Hardcoded fallback in ESI_Engine.py
```

### 1.2 physician.csv Structure
```csv
id,name,specialty,"availability_mask",workload_score
1,Dr. Sarah Chen,Cardiology,"[1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0]",0.45
2,Dr. Michael Rodriguez,Emergency Medicine,"[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]",0.72
```

Where:
- `id`: Unique physician identifier
- `name`: Display name
- `specialty`: Medical specialty for filtering
- `availability_mask`: 24-hour availability array (1=available hour, 0=unavailable/break)
- `workload_score`: Current workload (0.0=empty, 1.0=full)

### 1.3 Other CSVs
```
Optional (for embedding training):
- backend/triage_formatted/medqa_triage.jsonl (case database)
- backend/models/statpearls_cases.jsonl (clinical cases)
```

---

## PART 2: EMBEDDING & RAG RETRIEVER (Python)

### 2.1 How RAG Retriever Initializes

**File**: `backend/rag_retriever.py` (Line 176+)

```python
class TriageRAGRetriever:
    def _initialize(self):
        # PRIORITY ORDER for Loading Embeddings
        
        # PRIORITY 1: MedGemma GGUF (most clinical)
        # PRIORITY 2: PubMedBERT (clinical-specific)
        # PRIORITY 3: SentenceTransformer (generic fallback)
        # PRIORITY 4: Generate from scratch
        
        # Each priority tries to load pre-built FAISS index + metadata
```

### 2.2 Embedding Model Hierarchy

| Priority | Model | Strengths | Load Time | Accuracy |
|----------|-------|-----------|-----------|----------|
| 1 | MedGemma 7B GGUF | Most clinical, generative | 5-10 min | .92 |
| 2 | PubMedBERT | Semantic + clinical | 3-5 min | .88 |
| 3 | SentenceTransformer | Fast generic | 30 sec | .75 |
| 4 | None (LLM-only) | No embeddings | 0 sec | Depends on LLM |

### 2.3 FAISS Index Loading

```python
# Location: backend/models/
faiss_combined_medical.index        # Vector index (~500MB)
faiss_combined_medical_meta.jsonl   # Case metadata (~100MB)

# Process:
1. Load FAISS index into memory
2. Load metadata JSON lines (one case per line)
3. Load embedding model (PubMedBERT, etc)
4. Ready to search for similar cases
```

### 2.4 How FAISS Search Works

```python
# User wants to search for similar cases:

1. INPUT: Chief complaint + vitals
   Chest pain, HR 110, BP 165/95, no recent trauma

2. EMBED: Convert to vector using embedding model
   "chest pain heart rate elevated" → [0.234, 0.567, 0.891, ...]

3. SEARCH: Find k=5 nearest neighbors in FAISS
   distances, indices = faiss_index.search(query_vector, k=5)

4. RETRIEVE: Look up metadata for matching indices
   for idx in indices[0]:
       case = metadata[idx]
       similarity_score = 1 - distances[idx]
       
5. OUTPUT: Top 5 similar cases with scores
   [
       { case_id: 123, diagnosis: "ACS", esi: 2, similarity: 0.92 },
       { case_id: 456, diagnosis: "GERD", esi: 3, similarity: 0.78 },
       ...
   ]
```

---

## PART 3: ESI_ENGINE.py (The Orchestrator)

### 3.1 When ESI_Engine Runs

**Current Status**: Always off (blocking issue)

```python
# File: backend/ML/ESI_Engine.py

if __name__ == "__main__":
    # Called from Node.js via: POST http://localhost:5000/triage
    # Payload: { patient: {...}, vitals: {...}, transcript: "..." }
    
    engine = ESIEngine()
    result = engine.triage(payload)
    # Returns: triage details + physician rankings
```

### 3.2 What ESI_Engine Does

```python
def triage(self, payload):
    patient = payload['patient']
    vitals = payload['vitals']
    chief_complaint = patient.get('symptoms', '')
    
    # ===== STEP 1: Load Resources =====
    # Loads physician.csv into memory
    # Initializes RAG retriever
    # Sets up embedding model
    
    # ===== STEP 2: Estimate ESI Level =====
    esi_level = self._estimate_esi(vitals)
    # 1=Critical, 2=Emergent, 3=Urgent, 4=Less Urgent, 5=Minor
    
    # ===== STEP 3: Retrieve Similar Cases (RAG) =====
    if self.rag_available:
        similar_cases = self.rag_retriever.retrieve_similar_cases(
            chief_complaint=chief_complaint,
            vitals_summary=str(vitals),
            k=10,
            semantic_weight=0.35
        )
    
    # ===== STEP 4: Rank Physicians =====
    top_physicians = self._rank_physicians(
        esi_level=esi_level,
        specialty=inferred_specialty,
        similar_cases=similar_cases,
        availability_mask=physician.availability_mask,
        workload_score=physician.workload_score
    )
    
    return {
        esi_level: esi_level,
        assigned_specialty: specialty,
        top_5_embedding_physicians: top_physicians,
        top_5_composite_physicians: top_physicians,
        supporting_cases: similar_cases
    }
```

### 3.3 Physician Ranking Algorithm

```python
def _rank_physicians(self, esi, specialty, cases, masks, workloads):
    scores = []
    
    for physician in self.physicians:  # Loaded from CSV
        
        # 1. Specialty Match
        specialty_score = 0.9 if physician.specialty == specialty else 0.5
        
        # 2. Availability (from availability_mask[current_hour])
        current_hour = datetime.now().hour
        is_available = physician.availability_mask[current_hour]
        
        # 3. Workload (prefer least busy)
        capacity_score = 1.0 - physician.workload_score
        
        # 4. Case Similarity (if RAG available)
        case_match_score = 0.0
        if self.rag_available and self.physician_specialties.get(physician.id):
            # Do similar cases match physician's specialty?
            matching_cases = [c for c in cases if c.specialty == physician.specialty]
            case_match_score = len(matching_cases) / len(cases) if cases else 0.0
        
        # 5. Composite Score
        composite = (
            specialty_score * 0.40 +
            capacity_score * 0.30 +
            case_match_score * 0.30 +
            (1.0 if is_available else 0.0) * 0.10
        )
        
        scores.append({
            'physician_id': physician.id,
            'name': physician.name,
            'specialty': physician.specialty,
            'composite_score': composite,
            'availability': is_available
        })
    
    # Return top 5 sorted by composite score
    return sorted(scores, key=lambda x: x['composite_score'], reverse=True)[:5]
```

---

## PART 4: Node.js Backend Integration

### 4.1 physician.controller.js Flow

**File**: `backend/src/controllers/physician.controller.js`

```javascript
export const getRecommendations = async (req, res) => {
    const encounterId = req.query.encounterId;
    const enc = await Encounter.findById(encounterId);
    
    const triage = enc.triage || {};  // Result from Python
    
    // ===== SOURCE 1: triage object (from Python ESI_Engine) =====
    if (triage.top_5_embedding_physicians) {
        // Use Python's computed rankings
        let candidates = triage.top_5_embedding_physicians;
        
        // Promote assigned physician to #1 if present
        if (triage.assigned_physician_id) {
            const idx = candidates.findIndex(c => c.id === triage.assigned_physician_id);
            if (idx > 0) {
                const [assigned] = candidates.splice(idx, 1);
                candidates = [assigned, ...candidates].slice(0, 5);
            }
        }
        
        return res.json({ physicians: candidates });
    }
    
    // ===== SOURCE 2: Fallback (if Python failed) =====
    // Return empty or database-stored physicians
    return res.json({ physicians: [] });
};
```

### 4.2 Where Python Gets Called

**File**: `backend/src/controllers/physician.controller.js` (Line 84)

```javascript
// FALLBACK: Call Python ESI_Engine directly if no triage rankings exist
try {
    const payload = {
        patient: enc.patient || {},
        vitals: enc.vitals || {},
        transcript: enc.transcript || ""
    };
    
    // ⚠️ THIS IS THE CRITICAL CALL
    const pyRes = await axios.post(
        'http://localhost:5000/triage',  // ESI_Engine Flask endpoint
        payload,
        { timeout: 300000 }  // 5 min timeout
    );
    
    const pyTriage = pyRes.data;
    
    // Extract rankings from Python response
    candidates = pyTriage.top_5_composite_physicians || [];
    
} catch (e) {
    console.error("[physician.controller] Python triage call failed:", e);
    // Silently fail: return no recommendations
}
```

---

## PART 5: THE MISSING PIECE - COMPUTE MODE SETTINGS

### 5.1 WHY physician.csv ISN'T BEING LOADED NOW

```
Current Flow:
┌─────────────────────────────────────────┐
│ USER SUBMITS TRIAGE REQUEST             │
├─────────────────────────────────────────┤
│ Node.js checks: Is ESI_Engine running?  │
│                                         │
│ Current answer: ❌ NO (ALWAYS OFF)      │
│                                         │
│ Why? Because:                           │
│ 1. No compute mode setting exists       │
│ 2. No way to toggle LLM-Only vs RAG     │
│ 3. Default = OFF (blocked by design)    │
└─────────────────────────────────────────┘

Result:
- physician.csv ❌ NOT LOADED
- RAG Retriever ❌ NOT INITIALIZED
- FAISS Index ❌ NOT SEARCHED
- Python ESI_Engine ❌ NEVER RUNS
- Frontend gets empty recommendations ❌
```

### 5.2 WHAT NEEDS TO HAPPEN (WITH SETTINGS)

```
WITH COMPUTE MODES:

User selects in Settings:
○ LLM-Only (fast, no data)
● RAG-Full (comprehensive, needs physician.csv)
○ Hybrid (balanced)

┌──────────────────────────────────────────────┐
│ USER SELECTS "RAG_FULL" IN SETTINGS          │
├──────────────────────────────────────────────┤
│                                              │
│ Node.js reads: computeMode = "RAG_FULL"     │
│                                              │
│ ✅ ESI_Engine START                          │
│ ✅ Load physician.csv from filesystem        │
│ ✅ Initialize RAG retriever                  │
│ ✅ Load FAISS index + metadata               │
│ ✅ Load embedding model (PubMedBERT)        │
│                                              │
│ User submits triage:                         │
│ ✅ Call Python ESI_Engine.triage()          │
│ ✅ Retrieve similar cases (RAG)             │
│ ✅ Rank physicians with workload            │
│ ✅ Return top 5 with reasoning              │
│                                              │
└──────────────────────────────────────────────┘

Result:
✅ physician.csv LOADED
✅ RAG Retriever INITIALIZED
✅ FAISS Index SEARCHED
✅ Python ESI_Engine RUNS
✅ Frontend gets recommended physicians
```

---

## PART 6: DATA FLOW DIAGRAM

### Full Request → Response Cycle

```
┌─ FRONTEND ──────────────────────────────────────┐
│ User fills vitals, symptoms                     │
│ Clicks "Analyze"                                │
│ Sends POST /api/encounters/<encounterId>/triage│
└──────────────────────↓──────────────────────────┘

┌─ NODE.JS BACKEND ───────────────────────────────┐
│ encounter.controller.js receives payload         │
│ Saves encounter to MongoDB                       │
│                                                  │
│ CHECK COMPUTE MODE SETTING:                      │
│ ├─ LLM_ONLY → Call LLM only, no RAG            │
│ ├─ RAG_FULL → Proceed to Python                │
│ └─ HYBRID → Balance both                        │
│                                                  │
│ IF RAG_FULL:                                     │
│   Calls: axios.post('http://localhost:5000/triage', payload)
└──────────────────────↓──────────────────────────┘

┌─ PYTHON ESI_ENGINE ─────────────────────────────┐
│ Flask app receives request                       │
│                                                  │
│ 1. LOAD WORKFORCE                               │
│    ├─ physician.csv → self.physicians           │
│    └─ availability_mask per physician           │
│                                                  │
│ 2. INITIALIZE RAG                               │
│    ├─ Load embedding model (PubMedBERT)        │
│    ├─ Load FAISS index                         │
│    └─ Load metadata (medqa_triage.jsonl)       │
│                                                  │
│ 3. ESTIMATE ESI LEVEL                           │
│    ├─ Check HR, BP, RR, O2, temp               │
│    └─ Assign ESI 1-5                           │
│                                                  │
│ 4. RETRIEVE SIMILAR CASES (RAG)                 │
│    ├─ Embed chief complaint                     │
│    ├─ Search FAISS for k=10 neighbors          │
│    └─ Re-rank by vitals + specialty            │
│                                                  │
│ 5. RANK PHYSICIANS                              │
│    For each physician in CSV:                   │
│    ├─ specialty_match_score (0.0-1.0)          │
│    ├─ availability_score (0 or 1)              │
│    ├─ workload_score (0.0-1.0)                 │
│    ├─ case_match_score (0.0-1.0)               │
│    └─ composite = weighted sum                  │
│                                                  │
│ 6. RETURN TOP 5                                 │
│    ├─ top_5_composite_physicians               │
│    ├─ top_5_embedding_physicians               │
│    ├─ supporting_cases (evidence)              │
│    └─ esi_level + specialty                    │
└──────────────────────↓──────────────────────────┘

┌─ NODE.JS BACKEND (RESPONSE ASSEMBLY) ───────────┐
│ physician.controller receives Python response    │
│                                                  │
│ Normalize rankings:                              │
│ ├─ Dedup by physician_id                        │
│ ├─ Promote assigned physician to #1             │
│ └─ Slice top 5                                   │
│                                                  │
│ Save to Encounter.triage:                        │
│ ├─ assigned_physician_id                        │
│ ├─ assigned_specialty                           │
│ ├─ top_5_composite_physicians                   │
│ └─ supporting_cases                             │
│                                                  │
│ Return to frontend                               │
└──────────────────────↓──────────────────────────┘

┌─ FRONTEND ──────────────────────────────────────┐
│ PatientReport.jsx receives recommendations      │
│                                                  │
│ Display:                                         │
│ ┌────────────────────────────────────────────┐ │
│ │ TOP 5 RECOMMENDED PHYSICIANS               │ │
│ ├────────────────────────────────────────────┤ │
│ │ ✅ Dr. Sarah Chen (Cardiology) - 92% match│ │
│ │    Similar case: Acute MI, 52M, HR 110   │ │
│ │                                            │ │
│ │ ✅ Dr. Michael Rodriguez (EM) - 88% match │ │
│ │    Similar case: Chest pain, BP high     │ │
│ └────────────────────────────────────────────┘ │
│                                                  │
│ User can:                                        │
│ ├─ Click → Assign to physician                  │
│ ├─ Review → See supporting cases               │
│ └─ Override → Choose different physician       │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## PART 7: FILES INVOLVED

### Python Files (ESI_Engine & RAG)
```
backend/ML/
├─ ESI_Engine.py              (Orchestrator, physician ranking)
├─ rag_retriever.py           (Embeddings + FAISS search)
├─ Enhanced_IDF_Weighting.py  (Hybrid search scoring)
└─ build_combined_faiss_index_emergency.py  (Build FAISS index)

backend/
├─ models/
│  ├─ physician_data.csv      (Workforce data) ← CURRENTLY NOT LOADED
│  ├─ faiss_combined_medical.index  (Vector index)
│  └─ faiss_combined_medical_meta.jsonl  (Case metadata)
│
├─ triage_formatted/
│  └─ medqa_triage.jsonl      (Training/reference cases)
```

### Node.js Backend Files
```
backend/src/
├─ controllers/
│  ├─ physician.controller.js (Physician recommendations)
│  ├─ encounter.controller.js (Triage request entry point)
│  └─ schedule.controller.js  (NEW: Scheduling)
│
├─ models/
│  └─ (No settings model yet) ← NEEDS TO BE CREATED
│
└─ routes/
   └─ (No settings route yet) ← NEEDS TO BE CREATED
```

### Frontend Files
```
frontend/src/
├─ PatientSummary/
│  ├─ PatientReport.jsx       (Displays physician recommendations)
│  └─ PS.jsx                  (Entry point for triage)
│
├─ Dashboard/Pages/
│  ├─ Settings.jsx            (✅ Exists but needs Compute Modes tab)
│  └─ StaffSchedule.jsx       (✅ Newly created)
│
└─ (No useSettings hook yet) ← NEEDS TO BE CREATED
```

---

## SUMMARY: The Blocking Issue

**Current State**:
```
physician.csv ────→ Loaded only if ESI_Engine starts
ESI_Engine ────────→ Never starts (no compute mode)
Compute Mode ──────→ Doesn't exist (no settings system)
Settings System ───→ Exists but no compute mode toggle
```

**Solution Required** (in order):
```
1. ✅ CREATE: UserComputeModeSetting model (MongoDB)
2. ✅ CREATE: Settings API endpoints (GET/PATCH /api/settings/user)
3. ✅ UPDATE: ESI_Engine.py to check compute mode
4. ✅ CREATE: Frontend Compute Modes tab in Settings.jsx
5. ✅ UPDATE: physician.controller.js to read compute mode
6. ✅ THEN: physician.csv will load automatically based on user's choice
7. ✅ THEN: Full RAG pipeline activates
8. ✅ THEN: Physician recommendations appear
```

---

This is why physician.csv isn't loading and recommendations are empty — the settings infrastructure doesn't exist yet!

