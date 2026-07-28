# DETAILED SECTION-BY-SECTION COMPARISON

## LOCAL FILE vs REFERENCE IMPLEMENTATION

---

## SECTION 1: INITIALIZATION & IMPORTS

### LOCAL (ESI_Engine.py Lines 1-100)
```python
"""Triage service (fixed TEWS logic + explicit SATS discriminators)"""

import json, re, ast, sys, uuid, threading, datetime
import pandas as pd
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# HuggingFace configuration
os.environ['TRANSFORMERS_CACHE'] = './model_cache/'
os.environ['HF_HOME'] = './model_cache/'
...
# Diagnostic criteria import
try:
    from diagnostic_criteria import get_pathognomonic_markers
except:
    def get_pathognomonic_markers(): return {}

# RAG/Scenario matcher
try:
    from scenario_matcher import retrieve_scenarios
    SCENARIO_MATCHER_AVAILABLE = True
except:
    SCENARIO_MATCHER_AVAILABLE = False
```

**Status**: ✅ Similar to reference
**Issues**: Still tries to import RAG modules (not removed)

### REFERENCE (Prompt)
```python
"""Triage service (fixed TEWS logic + explicit SATS discriminators)"""

# HuggingFace offline-first configuration
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_CACHE'] = './model_cache/'
...
# GPU detection with fallback
if torch.cuda.is_available():
    DEVICE = 'cuda'
    gpu_name = torch.cuda.get_device_name(0)
else:
    DEVICE = 'cpu'
    LLM_ONLY_MODE = True  # Force LLM-only if CPU
```

**Difference**: Reference adds offline-first flags explicitly

---

## SECTION 2: MODEL LOADING & CONFIGURATION

### LOCAL (Lines 485-570)
```python
def load_gguf_model():
    model_path = _find_gguf_model()
    return Llama(
        model_path=model_path,
        n_ctx=2048,  # FIXED
        n_batch=512,
        n_gpu_layers=-1,
        verbose=False
    )

# Single configuration - not adaptive
gguf_model = load_gguf_model()
```

**Problems**:
- Fixed n_ctx=2048 (not 2500/3000/4096)
- No adaptive tier system
- No hardware detection

### REFERENCE (Lines 360-475)
```python
def calculate_adaptive_gguf_config(resources=None):
    """Returns n_ctx/n_batch based on available RAM"""
    if not resources:
        resources = get_system_resources()
    
    ram_gb = resources['available_gb']
    
    # TIER 1: Low-memory (<4GB)
    if ram_gb < 4:
        return {'n_ctx': 2500, 'n_batch': 256, 'n_threads': 2}
    
    # TIER 2: Mid-range (4-12GB) ← User's requested optimization
    elif ram_gb <= 12:
        return {'n_ctx': 3000, 'n_batch': 512, 'n_threads': 4}
    
    # TIER 3: High-end (>12GB)
    else:
        return {'n_ctx': 4096, 'n_batch': 1024, 'n_threads': 8}

config = calculate_adaptive_gguf_config()
gguf_model = Llama(
    model_path=model_path,
    n_ctx=config['n_ctx'],
    n_batch=config['n_batch'],
    n_threads=config['n_threads'],
    n_gpu_layers=-1,
    verbose=False
)
```

**Advantages**:
- Adapts to available RAM
- Tier 2 = 3000 (matches user request)
- Better hardware utilization

---

## SECTION 3: CHIEF COMPLAINT EXTRACTION

### LOCAL (Lines ~1400-1410)
```python
# Try to find chief complaint by looking for common phrases
chief_complaint_match = re.search(
    r'(?:chief complaint|cc:|cc\s*:|presents?(?:\s+with)?|c/o):?\s*([^.]+\.?)',
    pt_lower
)
chief_complaint = chief_complaint_match.group(1).strip() if chief_complaint_match else patient_text[:50]
```

**Issues**:
- Regex-based extraction from transcript
- Fragile (depends on formatting)
- Doesn't check patient.symptoms field first
- Returns first 50 chars if no match

### REFERENCE (Lines 7837-7850)
```python
def extract_chief_complaint(data):
    """Priority-ordered field extraction"""
    patient = data.get('patient', {})
    
    # Priority order:
    chief_complaint = (
        patient.get('symptoms') or              # 1st: Frontend symptoms field
        patient.get('presentingComplaint') or  # 2nd: Curated chief complaint
        data.get('chiefComplaint') or          # 3rd: Fallback field
        "Unspecified presentation"
    )
    
    # Clean and normalize
    chief_complaint = clean_field(chief_complaint)
    return chief_complaint

def clean_field(value):
    """Strip whitespace, handle None/empty"""
    if not value:
        return ""
    return str(value).strip()
```

**Advantages**:
- Structured JSON input (not transcript parsing)
- Clear priority order
- Handles missing fields gracefully
- Full symptoms text preserved

**Test Case Validation**:
- **Local**: Would extract "Dizziness" from first mention
- **Reference**: Would extract "Dizziness, palpitations, jaw pain, back pain, sweating" (full symptom list)

---

## SECTION 4: TRIAGE PROMPT

### LOCAL (Lines 1440-1560)
```python
pass1_prompt = f"""You are a clinical diagnostic AI trained to generate 
SYNTHESIS DIAGNOSIS queries.

Goal: Identify rare "Zebra" diagnoses

PATIENT PRESENTATION: {chief_complaint}
VITALS: {vitals_str}

🎯 SYNTHESIS DIAGNOSIS FRAMEWORK
... (300+ lines of query optimization advice)

OUTPUT ONLY VALID JSON with:
- search_queries: array of synthesis-aware queries
- primary_hypothesis: working diagnosis
- synthesis_reasoning: ...
- confidence: 0.0-1.0
"""
```

**Purpose**: Optimize RAG queries to find rare diagnoses
**Output**: JSON with search_queries (not clinical decisions)
**Problem**: Focuses on diagnosis hypothesis, not on ESI classification or specific interventions

### REFERENCE (Lines 8084-8180)
```python
triage_prompt = f"""You are an emergency physician in the ED performing triage.

PATIENT INFO:
Name: {patient_name}
Age: {patient_age}
Gender: {patient_gender}
Medical History: {medical_history}
Chief Complaint: {chief_complaint}

VITAL SIGNS:
HR: {hr} bpm
RR: {rr} breaths/min
SBP/DBP: {sbp}/{dbp} mmHg
O2 Sat: {o2}%
Temp: {temp}°C

═══════════════════════════════════════════════════════════════
VITAL INTERPRETATION REFERENCE:
═══════════════════════════════════════════════════════════════
HR normal: 60-100 bpm (abnormal if >130 or <40)
RR normal: 12-20 breaths/min (abnormal if >30 or <8)
SBP normal: 90-140 mmHg (abnormal if <90 or >200)
O2 saturation: >95% (abnormal if <88%)
Temperature: 36-38°C (abnormal if <36 or >38.5)

═══════════════════════════════════════════════════════════════
ESI CLASSIFICATION RULES:
═══════════════════════════════════════════════════════════════
ESI-1: Requires immediate intervention
  - Intubation, emergency thoracotomy, emergency pacing

ESI-2: High-risk situation; STAT procedures needed
  - ACS (chest pain + risk factors), Sepsis, Acute stroke, GCS <9
  - Hypotension <90 SBP with concerning presentation

ESI-3: Moderate-risk; urgent labs/imaging/specialty
  - Possible AMI without ST elevation, stable COPD exacerbation

ESI-4: Low-risk stable; may need workup
  - Uncomplicated visit, stable vitals

ESI-5: Minor injury/illness
  - Self-limited condition

═══════════════════════════════════════════════════════════════
MAJOR PRESENTATION PROTOCOLS (WITH SPECIFIC MEASUREMENTS):
═══════════════════════════════════════════════════════════════

**ACUTE CORONARY SYNDROME (ACS):**
- 12-lead ECG within 10 minutes (STAT)
- Troponin (high-sensitivity) blood draw (STAT)
- IV access minimum 18-gauge cannula
- Aspirin 325mg oral (or 500mg if GI intolerance)
- Continuous cardiac monitoring
- Cardiology consultation (STAT if STEMI)
- Nitroglycerin 0.4mg sublingual PRN (if SBP >90)

**SEPSIS/SEPTIC SHOCK:**
- Blood cultures ×2 (before antibiotics) STAT
- Broad-spectrum antibiotics <1 hour
- IV fluid bolus: 30mL/kg initial
- Lactate level (STAT)
- Central venous line (if hypotensive after fluids)
- Vasopressors (norepinephrine/dopamine) if SBP <90

**SHOCK (Cardiogenic/Hypovolemic/Distributive):**
- IV access: ×2 sites, 18G+ preferred
- Fluid bolus: 500-1000mL initial
- Vasopressor if SBP <90 after fluids
- Arterial line for monitoring
- ICU consideration

**ACUTE NEURO EMERGENCY:**
- Blood glucose (STAT)
- CT head non-contrast (STAT)
- Neurology consultation (STAT)
- Keep NPO
- Continuous monitoring

═══════════════════════════════════════════════════════════════
ENFORCEMENT RULES:
═══════════════════════════════════════════════════════════════
⚠️  All interventions must specify:
  - DOSE (mg/mL/volume)
  - ROUTE (PO/IV/IM/SC/etc)
  - VOLUME (mL if applicable)
  - TIMEFRAME (STAT/within X minutes/etc)

GOOD EXAMPLES:
  - "ECG 12-lead within 10 minutes" ✅
  - "Aspirin 325mg oral immediately" ✅
  - "IV fluid bolus 30mL/kg over 30 minutes" ✅

BAD EXAMPLES:
  - "Monitoring" ❌ (no specifics)
  - "Cardiac intervention" ❌ (vague)
  - "Observation" ❌ (insufficient detail)

═══════════════════════════════════════════════════════════════
RESPONSE FORMAT:
═══════════════════════════════════════════════════════════════
Output ONLY valid JSON (no markdown, no explanations):

{{
  "esi_level": 1,
  "color": "RED",
  "interventions": [
    "specific intervention with dose/route/timeframe",
    "another specific intervention",
    ...
  ],
  "reasoning": "evidence-based explanation"
}}
"""
```

**Key Differences**:

| Aspect | LOCAL | REFERENCE |
|--------|-------|-----------|
| **Goal** | Find rare diagnoses | Classify ESI level + generate interventions |
| **Input** | Chief complaint + vitals string | Full patient context + vital values |
| **Output** | search_queries JSON | esi_level, color, interventions array |
| **Enforcement** | Query optimization | Intervention specificity |
| **Length** | ~300 tokens | ~1200 tokens |
| **Prescriptive** | Find what to search | Tell me what to do |

---

## SECTION 5: RED FLAG DETECTION

### LOCAL (Lines 1051-1200+)
```python
def detect_red_flag_keywords(patient_text: str, vitals: dict) -> dict:
    """Detects red flags using HARDCODED keyword lists"""
    
    pt_lower = patient_text.lower()
    
    # HARDCODED LISTS (Exactly what user said "don't hardcode")
    cardiac_keywords = ['chest pain', 'palpitation', 'dyspnea', 'diaphoresis', 
                       'tachycardia', 'syncope', 'presyncope', 'crushing', 
                       'pressure', 'radiating']
    
    respiratory_keywords = ['dyspnea', 'wheezing', 'stridor', 'cyanosis', 
                           'apnea', 'hypoxia', 'desaturation']
    
    neurologic_keywords = ['seizure', 'stroke', 'altered mental status', 'ams', 
                          'loss of consciousness', 'loc', 'syncope', 'unresponsive']
    
    sepsis_keywords = ['fever', 'chills', 'altered mental status', 'hypotension', 
                      'tachycardia', 'tachypnea']
    
    # ... more hardcoded lists
    
    # Check if any keyword matches
    has_cardiac_red_flag = any(kw in pt_lower for kw in cardiac_keywords)
    has_respiratory_red_flag = any(kw in pt_lower for kw in respiratory_keywords)
    # ... etc
```

**Problems**:
- User explicitly said: "dont hardcode stress test examples that's cheating"
- 200+ lines of hardcoded keywords
- Brittle (misses synonyms, variations)
- Scales poorly
- Violates clinical reasoning principle

### REFERENCE (No Such Function)
```python
# No hardcoded red flag detection
# LLM evaluates based on VITAL CONTEXT:

# Example: HR 118, BP 88/55, symptoms = palpitations + jaw pain
# LLM prompt says: "ESI-2: High-risk, STAT procedures needed... if hypotension with concerning presentation"
# LLM thinks: "OK, this is hypotensive (88/55) + tachycardic (118) + concerning symptoms → ESI-2 ACS"
# LLM generates: ECG STAT, troponin STAT, IV access, aspirin, monitoring
```

**Advantage**: Clinical reasoning, not keyword matching

---

## SECTION 6: OUTPUT STRUCTURE

### LOCAL
```python
# Output example:
{
  "search_queries": [
    "fever with rash",
    "fever and arthritis",
    "negative blood culture"
  ],
  "primary_hypothesis": "adult onset still's disease",
  "synthesis_reasoning": "...",
  "confidence": 0.78,
  "lab_findings": [],
  "esi_level": 2  # Added by supervisor, not main output
}
```

**Purpose**: Information for knowledge graph retrieval
**Focus**: Diagnosis hypothesis

### REFERENCE
```python
{
  "esi_level": 2,
  "color": "ORANGE",
  "interventions": [
    "12-lead ECG within 10 minutes STAT",
    "Troponin (high-sensitivity) blood draw STAT",
    "IV access 18-gauge minimum",
    "Aspirin 325mg oral immediately",
    "Continuous cardiac monitoring",
    "Cardiology consultation STAT if STEMI suspected"
  ],
  "reasoning": "Hypotensive (88/55) with tachycardia (118) and concerning symptoms (jaw pain, back pain, palpitations) in diabetic → ESI-2 ACS presentation. Requires STAT ECG, troponin, and cardiac monitoring. Fluid resuscitation with careful monitoring given hypotension."
}
```

**Purpose**: Actionable clinical decisions
**Focus**: ESI classification + specific interventions

---

## SECTION 7: FALLBACK MECHANISM

### LOCAL (Line ~7720)
```python
def _generate_default_interventions(triage_data, esi_level):
    """Fallback if LLM or RAG fails - generates from hardcoded patterns"""
    
    if esi_level == 1:
        return ["Emergency assessment", "Consider intubation", "Emergency consulting"]
    elif esi_level == 2:
        return ["Urgent assessment", "IV access", "Labs and imaging"]
    elif esi_level == 3:
        return ["Standard assessment", "Vital signs monitoring"]
    # ... etc
```

**Problem**: Too generic

### REFERENCE (Lines 7720-7800)
```python
def _generate_default_interventions(triage_data, esi_level):
    """Generates context-aware fallback interventions when LLM fails"""
    
    vitals = triage_data.get('vitals', {})
    chief_complaint = triage_data.get('chief_complaint', '')
    
    if esi_level == 1:
        # STAT interventions (emergency airway/circulation)
        interventions = [
            "Rapid assessment with senior physician",
            "Continuous cardiopulmonary monitoring",
            "Prepare for emergency procedures"
        ]
        if vitals.get('sbp', 200) < 90:
            interventions.append("IV access ×2 sites, fluid bolus 500mL over 15 minutes")
        if vitals.get('o2', 100) < 88:
            interventions.append("Supplemental oxygen, prepare for intubation")
    
    elif esi_level == 2:
        # STAT labs/imaging (high-risk)
        interventions = [
            "Immediate physician assessment",
            "IV access",
            "Continuous monitoring"
        ]
        # Context-aware additions
        if 'chest' in chief_complaint.lower() or vitals.get('hr', 70) > 110:
            interventions.append("12-lead ECG STAT, troponin STAT")
        if vitals.get('temp', 37) > 38.5 or vitals.get('hr', 70) > 110:
            interventions.append("Blood cultures, lactate STAT")
    
    # ... more logic
    return interventions
```

**Advantages**:
- Context-aware (uses vitals + symptoms)
- Not purely hardcoded (makes decisions)
- Matches patient presentation

---

## SECTION 8: ENDPOINT STRUCTURE

### LOCAL
Multiple endpoints for different purposes:
```python
@app.route('/synthesis-diagnosis', methods=['POST'])
def synthesis_diagnosis():
    # Pass 1: Generate queries
    
@app.route('/supervisor', methods=['POST'])
def rag_supervisor():
    # Pass 2: RAG + LLM reasoning
    
@app.route('/specialty-decision', methods=['POST'])
def specialty_decision():
    # Pass 3: Final classification
```

**Architecture**: Multi-pass pipeline

### REFERENCE
Single unified endpoint:
```python
@app.route('/llm-triage', methods=['POST'])
def llm_triage():
    """Single-pass triage decision"""
    data = request.json
    
    patient = data['patient']
    vitals = data['vitals']
    
    # Extract + Process
    chief_complaint = extract_chief_complaint(data)
    tews_score = calculate_tews(vitals)
    
    # LLM Inference
    with gguf_semaphore:  # Thread-safe
        output = gguf_generate(prompt, max_tokens=1024, temperature=0.0)
    
    # Parse + Return
    response = parse_triage_json(output)
    if not response:
        response = _generate_default_interventions(data, tews_to_esi(tews_score))
    
    return jsonify(response)
```

**Architecture**: Single deterministic pass

---

## SECTION 9: TOKEN MATH

### LOCAL
```
n_ctx = 2048 tokens (fixed)
Chief complaint: ~50 tokens
Vitals: ~20 tokens
Prompt (Pass 1): ~300 tokens
Total input: ~370 tokens
Max output: 500 tokens
Required: 870 tokens ≤ 2048 ✅ Fits
```

### REFERENCE
```
n_ctx TIERS:
- Tier 1: 2500 tokens (serves <4GB systems)
- Tier 2: 3000 tokens (serves 4-12GB systems) ← User's request
- Tier 3: 4096 tokens (serves >12GB systems)

Chief complaint: ~50 tokens
Vitals: ~20 tokens
Prompt (comprehensive): ~1200 tokens (includes protocols + rules)
Total input: ~1270 tokens
Max output: 1024 tokens (clinical decision + reasoning)
Required: 2294 tokens ≤ 2500 ✅ Fits Tier 1
Required: 2294 tokens ≤ 3000 ✅ Fits Tier 2
```

**Implication**: Reference can handle comprehensive prompt across all hardware tiers

---

## SUMMARY TABLE

| Feature | LOCAL | REFERENCE | WINNER |
|---------|-------|-----------|--------|
| **Architecture** | RAG multi-pass | LLM single-pass | Reference (simpler) |
| **Input Format** | Free-form transcript | Structured JSON | Reference (cleaner) |
| **Chief Complaint** | Regex extraction | Priority field ordering | Reference (robust) |
| **Red Flags** | Hardcoded keywords ❌ | LLM reasoning | Reference (user request) |
| **Interventions** | Generic + RAG cases | Specific + doses/routes | Reference (user requirement) |
| **Config** | Fixed n_ctx=2048 | Adaptive 2500/3000/4096 | Reference (flexible) |
| **Offline** | RAG requires network | Fully offline ✅ | Reference (HIPAA) |
| **Performance** | 3 passes (slow) | 1 pass (fast) | Reference |
| **Maintenance** | Complex multi-pass | Simple single-pass | Reference |
| **Prompt** | 300 tokens (queries) | 1200 tokens (clinical) | Reference (comprehensive) |
| **Output** | Hypothesis + confidence | ESI + interventions | Reference (actionable) |
| **Fallback** | Generic | Context-aware | Reference |
| **Reproducibility** | Multi-pass variance | Deterministic | Reference |

---

## CONCLUSION

**The local implementation and reference are INCOMPATIBLE ARCHITECTURES.**

- **Local**: Knowledge graph + multi-pass + hardcoding (medical research tool)
- **Reference**: GGUF LLM + single-pass + clinical specificity (ED triage tool)

**User's Requirements Alignment**:
- ✅ Reference: "specific units of measurements of pharmaceutical resources"
- ✅ Reference: "don't hardcode"
- ✅ Reference: "life saving interventions"
- ❌ Local: Has extensive hardcoding
- ❌ Local: Generic interventions
- ❌ Local: Requires external knowledge graph

**Recommendation**: Migrate to Reference implementation for production ED triage.
