# CODE SNIPPET COMPARISON: LOCAL vs REFERENCE

## Example 1: RED FLAG DETECTION

### LOCAL (9405-line file)
**Location**: Lines 1051-1200
```python
def detect_red_flag_keywords(patient_text: str, vitals: dict) -> dict:
    """Detects red flags using HARDCODED keyword lists"""
    
    pt_lower = patient_text.lower()
    
    # HARDCODED KEYWORDS (User said "don't do this")
    cardiac_keywords = [
        'chest pain', 'chest discomfort', 'palpitation', 'palpitations',
        'dyspnea', 'dyspnoea', 'shortness of breath', 'sob',
        'diaphoresis', 'diaphoretic', 'sweating', 'perspiration',
        'tachycardia', 'tachycardic', 'rapid heart rate',
        'syncope', 'loss of consciousness', 'blackout',
        'presyncope', 'dizziness', 'vertigo', 'lightheadedness',
        'crushing', 'pressure', 'heaviness', 'constriction',
        'radiating', 'radiation', 'arm pain', 'jaw pain', 'back pain'
    ]
    
    respiratory_keywords = [
        'dyspnea', 'dyspnoea', 'shortness of breath', 'sob',
        'wheezing', 'wheeze', 'stridor', 'stertor',
        'cyanosis', 'cyanotic', 'bluish', 'blue lips',
        'apnea', 'apnoeic', 'apneic', 'no breath',
        'hypoxia', 'hypoxic', 'low oxygen',
        'desaturation', 'O2 sat', 'oxygen saturation'
    ]
    
    neurologic_keywords = [
        'seizure', 'seizing', 'seizures', 'convulsion', 'convulsive',
        'stroke', 'CVA', 'cerebrovascular accident', 'TIA', 'ministroke',
        'altered mental status', 'ams', 'confusion', 'confused',
        'loss of consciousness', 'loc', 'unconscious',
        'unresponsive', 'coma', 'comatose',
        'syncope', 'syncope syncope', 'fainting',
        'weakness', 'weak', 'paralysis', 'paralyzed',
        'numbness', 'numb', 'tingling', 'paresthesia'
    ]
    
    sepsis_keywords = [
        'fever', 'febrile', 'elevated temperature', 'high temperature',
        'chills', 'shivering', 'rigors', 'rigor',
        'altered mental status', 'ams', 'confusion',
        'hypotension', 'low blood pressure', 'low bp',
        'tachycardia', 'rapid heart rate', 'fast heart rate',
        'tachypnea', 'rapid breathing', 'fast breathing'
    ]
    
    # ... more hardcoded lists (200+ keywords total)
    
    # Check if any keyword matches
    has_cardiac_red_flag = any(kw in pt_lower for kw in cardiac_keywords)
    has_respiratory_red_flag = any(kw in pt_lower for kw in respiratory_keywords)
    has_neurologic_red_flag = any(kw in pt_lower for kw in neurologic_keywords)
    has_sepsis_red_flag = any(kw in pt_lower for kw in sepsis_keywords)
    
    return {
        'cardiac': has_cardiac_red_flag,
        'respiratory': has_respiratory_red_flag,
        'neurologic': has_neurologic_red_flag,
        'sepsis': has_sepsis_red_flag
    }

# USAGE IN MAIN LOGIC
red_flags = detect_red_flag_keywords(patient_text, vitals)
if red_flags['cardiac']:
    esi_level = 2
elif red_flags['sepsis']:
    esi_level = 2
elif red_flags['respiratory']:
    esi_level = 2
```

**Problems**:
- ❌ Exactly what user said "don't hardcode"
- ❌ Brittle (misses synonyms, variations, context)
- ❌ 200+ lines of keyword lists
- ❌ Scales poorly
- ❌ Doesn't use LLM reasoning

---

### REFERENCE (From Prompt Specification)
```python
# NO detect_red_flag_keywords() function at all
# Instead: LLM reads vital context and makes decision

# The prompt includes:
triage_prompt = f"""
PATIENT INFO:
- Age: {patient_age}
- Chief Complaint: {chief_complaint}
- Medical History: {medical_history}

VITAL SIGNS:
- HR: {hr} bpm
- RR: {rr} breaths/min
- SBP/DBP: {sbp}/{dbp} mmHg
- O2 Sat: {o2}%
- Temp: {temp}°C

VITAL INTERPRETATION REFERENCE:
- HR abnormal if >130 or <40
- RR abnormal if >30 or <8
- SBP abnormal if <90 or >200  ← KEY: This triggers ESI-2 logic
- O2 abnormal if <88%
- Temp abnormal if <36 or >38.5

ESI CLASSIFICATION RULES:
- ESI-2: High-risk situation; STAT procedures needed
  Examples: ACS, Sepsis, Acute stroke, Hypotension with concerning presentation
"""

# LLM OUTPUT: No hardcoding, just clinical reasoning
# Example: "HR 118, BP 88/55 (abnormal: <90), symptoms = palpitations + jaw pain"
# LLM thinks: "Hypotensive + tachycardic + ACS symptoms → ESI-2"
# Output: {"esi_level": 2, "interventions": ["ECG STAT", ...]}
```

**Advantages**:
- ✅ Clinical reasoning, not keyword matching
- ✅ Considers vital context
- ✅ Scales to any presentation
- ✅ LLM makes the decision
- ✅ No hardcoding

---

## Example 2: CHIEF COMPLAINT EXTRACTION

### LOCAL
**Location**: Lines 1400-1410
```python
# Try to find chief complaint by looking for common phrases
chief_complaint_match = re.search(
    r'(?:chief complaint|cc:|cc\s*:|presents?(?:\s+with)?|c/o):?\s*([^.]+\.?)',
    pt_lower
)
chief_complaint = (
    chief_complaint_match.group(1).strip() 
    if chief_complaint_match 
    else patient_text[:50]  # Fallback to first 50 chars
)

# Example processing:
# Input:  "Patient is a 72 year old male with dizziness, palpitations, jaw pain..."
# Output: "patient is a 72 year old male with dizz..."  ← TRUNCATED!
```

**Problems**:
- ❌ Truncates to first 50 characters
- ❌ Loses important symptoms (palpitations, jaw pain, back pain, sweating)
- ❌ Fragile regex (depends on formatting)
- ❌ Unstructured input parsing

---

### REFERENCE
```python
def extract_chief_complaint(data):
    """Priority-ordered field extraction from structured JSON"""
    
    patient = data.get('patient', {})
    
    # Priority order:
    # 1st: Frontend symptoms field (detailed list)
    # 2nd: Curated chief complaint (summary)
    # 3rd: Fallback field
    chief_complaint = (
        patient.get('symptoms') or 
        patient.get('presentingComplaint') or 
        data.get('chiefComplaint') or 
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

# Example processing:
# Input JSON:
# {
#   "patient": {
#     "symptoms": "Dizziness, palpitations, jaw pain, back pain, sweating"
#   }
# }
# 
# Output: "Dizziness, palpitations, jaw pain, back pain, sweating"
#         ✅ COMPLETE symptom list preserved
```

**Advantages**:
- ✅ Preserves all symptoms (no truncation)
- ✅ Robust (structured JSON input)
- ✅ Clear priority order
- ✅ Handles missing fields gracefully
- ✅ No regex fragility

---

## Example 3: PROMPT STRUCTURE

### LOCAL
**Location**: Lines 1440-1560
```python
pass1_prompt = f"""You are a clinical diagnostic AI trained to generate 
SYNTHESIS DIAGNOSIS queries.

Your goal is to identify rare "Zebra" diagnoses by understanding how 
multiple clinical features combine.

PATIENT PRESENTATION:
{chief_complaint}

VITALS: {vitals_str}

═══════════════════════════════════════════════════════════════
🎯 SYNTHESIS DIAGNOSIS FRAMEWORK
═══════════════════════════════════════════════════════════════

Most medical students learn the TARGET diagnosis + common DISTRACTORS side-by-side:
  • SALMON-PINK RASH + FEVER + ARTHRITIS → Think AOSD, NOT Amoxicillin rash
  • PETECHIAE + NEGATIVE CULTURE → Think Meningococcemia or TTP
  • HIGH FERRITIN + FEVER + RASH → Think HLH or MAS

⚠️  THE KEYWORD OVERLAP TRAP:
In medical datasets, common distractors often share keywords with zebra diagnoses.
Example: If you search "fever rash amoxicillin," RAG returns 1000 drug reaction cases
and misses the 2 AOSD cases.
SOLUTION: Generate queries that COMBINE multiple features to isolate the zebra.

...

OUTPUT ONLY VALID JSON with:
  "search_queries": array of 3 synthesis-aware clinical queries
  "lab_findings": array (EMPTY [] if no labs - do NOT invent)
  "primary_hypothesis": Working diagnosis
  "synthesis_reasoning": Explain clinical features
  "confidence": 0.0-1.0
"""
```

**Purpose**: Optimize RAG queries for rare disease discovery
**Output**: search_queries (not clinical actions)
**Length**: ~300 tokens

---

### REFERENCE
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
ESI-1: Requires immediate intervention (intubation, emergency pacing)
ESI-2: High-risk, STAT procedures needed (ACS, Sepsis, Stroke, Hypotension)
ESI-3: Moderate risk, urgent labs/imaging/specialty
ESI-4: Low-risk stable
ESI-5: Minor injury/illness

═══════════════════════════════════════════════════════════════
MAJOR PRESENTATION PROTOCOLS:
═══════════════════════════════════════════════════════════════

**ACUTE CORONARY SYNDROME (ACS):**
- 12-lead ECG within 10 minutes STAT
- Troponin (high-sensitivity) blood draw STAT
- IV access minimum 18-gauge cannula
- Aspirin 325mg oral (or 500mg if GI intolerance)
- Continuous cardiac monitoring
- Cardiology consultation STAT if STEMI

**SEPSIS/SEPTIC SHOCK:**
- Blood cultures ×2 (before antibiotics) STAT
- Broad-spectrum antibiotics <1 hour
- IV fluid bolus: 30mL/kg initial
- Lactate level STAT
- Vasopressors if SBP <90 after fluids

**SHOCK (Cardiogenic/Hypovolemic/Distributive):**
- IV access: ×2 sites, 18G+ preferred
- Fluid bolus: 500-1000mL initial
- Vasopressor if SBP <90 after fluids

**ACUTE NEURO EMERGENCY:**
- Blood glucose STAT
- CT head non-contrast STAT
- Neurology consultation STAT
- Keep NPO

═══════════════════════════════════════════════════════════════
ENFORCEMENT RULES:
═══════════════════════════════════════════════════════════════
⚠️  All interventions must specify:
  - DOSE (mg/mL)
  - ROUTE (PO/IV/IM/SC)
  - VOLUME (mL)
  - TIMEFRAME (STAT/within X minutes)

GOOD: "ECG 12-lead within 10 minutes" ✅
BAD: "obtain ECG" ❌

═══════════════════════════════════════════════════════════════
RESPONSE FORMAT:
═══════════════════════════════════════════════════════════════
Output ONLY valid JSON (no markdown):
{{
  "esi_level": 1-5,
  "color": "RED|ORANGE|YELLOW|GREEN|BLUE",
  "interventions": ["specific intervention 1", "specific intervention 2", ...],
  "reasoning": "evidence-based clinical reasoning"
}}
"""
```

**Purpose**: Generate ESI classification + specific interventions
**Output**: esi_level, color, interventions[] (actionable)
**Length**: ~1200 tokens

---

## Example 4: OUTPUT FORMAT

### LOCAL
```python
# Example response:
{
    "search_queries": [
        "fever with rash",
        "fever and arthritis", 
        "negative blood culture"
    ],
    "primary_hypothesis": "adult onset still disease",
    "synthesis_reasoning": "Salmon-pink rash + fever + arthritis + negative culture suggests AOSD, not drug reaction",
    "confidence": 0.78,
    "lab_findings": []
}

# Purpose: Information FOR KNOWLEDGE GRAPH RETRIEVAL
# User action: Search the knowledge graph with these queries
# NOT direct clinical decision
```

---

### REFERENCE
```python
# Example response:
{
    "esi_level": 2,
    "color": "ORANGE",
    "interventions": [
        "12-lead ECG within 10 minutes STAT",
        "Troponin (high-sensitivity) blood draw STAT",
        "IV access minimum 18-gauge cannula",
        "Aspirin 325mg oral immediately",
        "Continuous cardiac monitoring",
        "Cardiology consultation STAT if STEMI suspected",
        "IV fluid bolus if hypotension develops"
    ],
    "reasoning": "Hypotensive (88/55) with tachycardia (118), palpitations, and jaw/back pain in 72yo diabetic patient indicates high-probability ACS. Requires STAT ECG, troponin assessment, and immediate cardiology evaluation. Fluid resuscitation cautious given hypotension without volume overload findings."
}

# Purpose: DIRECT CLINICAL DECISION
# Nurse action: Start ECG, draw blood, establish IV, give aspirin, call cardiology
# Actionable and specific
```

---

## Example 5: CONFIGURATION

### LOCAL
```python
def load_gguf_model():
    model_path = _find_gguf_model()
    return Llama(
        model_path=model_path,
        n_ctx=2048,           # FIXED VALUE (not adaptive)
        n_batch=512,          # FIXED VALUE
        n_gpu_layers=-1,
        verbose=False
    )

# Problem: Works for mid-range systems
# Problem: May fail on low-memory systems (n_ctx too large)
# Problem: Doesn't leverage high-memory systems
```

---

### REFERENCE
```python
def get_system_resources():
    """Detects available RAM"""
    import psutil
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3)
    }

def calculate_adaptive_gguf_config(resources=None):
    """Returns n_ctx/n_batch based on available RAM"""
    
    if not resources:
        resources = get_system_resources()
    
    ram_gb = resources['available_gb']
    
    # TIER 1: Low-memory (<4GB)
    if ram_gb < 4:
        return {
            'n_ctx': 2500,
            'n_batch': 256,
            'n_threads': 2
        }
    
    # TIER 2: Mid-range (4-12GB) ← User requested: "mid range should be 3000 ncontx"
    elif ram_gb <= 12:
        return {
            'n_ctx': 3000,      # Optimized per user request
            'n_batch': 512,
            'n_threads': 4
        }
    
    # TIER 3: High-end (>12GB)
    else:
        return {
            'n_ctx': 4096,
            'n_batch': 1024,
            'n_threads': 8
        }

# Usage:
config = calculate_adaptive_gguf_config()
gguf_model = Llama(
    model_path=model_path,
    n_ctx=config['n_ctx'],       # Adaptive
    n_batch=config['n_batch'],   # Adaptive
    n_threads=config['n_threads'],
    n_gpu_layers=-1,
    verbose=False
)

# Advantages:
# ✅ Works on low-memory systems (n_ctx=2500)
# ✅ Works on mid-range systems (n_ctx=3000 per user request)
# ✅ Works on high-memory systems (n_ctx=4096)
# ✅ Matches user's preference: "mid range should be 3000 ncontx"
```

---

## SUMMARY TABLE

| Aspect | LOCAL | REFERENCE | Winner |
|--------|-------|-----------|--------|
| Red Flag Detection | Hardcoded keywords (200+) | LLM reasoning | REFERENCE |
| Symptom Extraction | Truncated (50 chars) | Complete (all symptoms) | REFERENCE |
| Prompt Focus | RAG query optimization | Clinical decision + interventions | REFERENCE |
| Output Purpose | Information retrieval | Clinical actions | REFERENCE |
| Configuration | Fixed (n_ctx=2048) | Adaptive (2500/3000/4096) | REFERENCE |
| User Alignment | ❌ Conflicts | ✅ Matches all | REFERENCE |

---

The reference implementation is architecturally superior and aligns with all explicit user requirements.
