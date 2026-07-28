# ESI_Engine.py: Local vs Reference Code Comparison

## Summary
The **local ESI_Engine.py (current implementation)** and the **reference implementation in your prompt** are **fundamentally different systems** with conflicting architectural approaches.

---

## 1. ARCHITECTURAL DIFFERENCES

### Local Implementation (Current)
- **RAG-First Architecture**: Relies on knowledge graph + Langchain RAG retrieval
- **Multi-Pass Prompting**: Pass 1 (synthesis diagnosis), Pass 2 (LLM reasoning), Pass 3 (integration)
- **Keyword-Based Red Flags**: Heavy use of hardcoded red flag detection
- **Complex Filtering**: Sex-based filtering, obstetric exclusions, drug reaction detection
- **Emphasis**: Finding rare diagnoses ("zebra cases") through synthesis queries
- **Max output tokens**: 150-500 tokens per pass
- **File size**: 9,405 lines

### Reference Implementation (Prompt)
- **LLM-Only Architecture**: Single direct call to MedGemma GGUF
- **Single-Pass Prompting**: One comprehensive prompt with all clinical context
- **Evidence-Based Protocols**: Specific ACS/Sepsis/Shock/Neuro protocols with doses
- **Minimal Filtering**: No hardcoded keyword matching
- **Emphasis**: Clinical specificity with exact measurements (doses, routes, volumes, timeframes)
- **Max output tokens**: 1,024 tokens
- **Adaptive n_ctx**: 2500/3000/4096 based on hardware tier
- **Offline-first**: Designed for disconnected operation (HIPAA-compliant)

---

## 2. MAJOR FUNCTIONAL DIFFERENCES

| Feature | Local Implementation | Reference Implementation |
|---------|---------------------|--------------------------|
| **Primary Method** | RAG retrieval + LLM synthesis | Direct LLM clinical reasoning |
| **Chief Complaint Extraction** | Regex pattern from transcript | Priority: patient.symptoms → presentingComplaint → chiefComplaint |
| **Vital Scoring** | TEWS/SATS framework | TEWS baseline + LLM interpretation |
| **Red Flag Detection** | 200+ hardcoded keyword lists | None (LLM decides based on vitals + context) |
| **ESI Classification** | LLM decides based on vitals + red flags | LLM decides with explicit ESI criteria rules |
| **Intervention Generation** | RAG-based case matching | LLM with enforced specificity (doses, routes, volumes) |
| **Output Format** | Free-form JSON with search_queries | Structured JSON: esi_level, color, interventions array |
| **Context Window (n_ctx)** | Not configurable | Adaptive: 2500/3000/4096 tokens |
| **Model Loading** | Single fixed configuration | Adaptive tier system based on available RAM |
| **Fallback Mechanism** | Synthetic query generation | _generate_default_interventions() with TEWS-based recommendations |
| **Request Deduplication** | Simple cache with 30-sec TTL | Same mechanism but integrated differently |
| **Thread Safety** | Flask default | Explicit semaphore lock on GGUF inference |
| **Endpoint Architecture** | Multiple endpoints (synthesis, supervisor, specialty) | Single /llm-triage endpoint |
| **Data Input** | Free-form patient transcript | Structured JSON with demographics + vitals + symptoms |

---

## 3. HARDCODING COMPARISON

### Local Implementation - EXTENSIVE Hardcoding
```python
# Line ~1051: detect_red_flag_keywords() function
cardiac_keywords = ['chest pain', 'palpitation', 'diaphoresis', ...]
respiratory_keywords = ['dyspnea', 'wheezing', 'stridor', ...]
neurologic_keywords = ['seizure', 'stroke', 'altered mental status', ...]
sepsis_keywords = ['fever', 'chills', 'altered mental status', ...]
# ... and many more hardcoded lists
```

**Problem**: User explicitly said "dont hardcode stress test examples that's cheating"

### Reference Implementation - ZERO Hardcoding
```python
# No keyword lists anywhere
# LLM evaluates context: "If HR 118, BP 88/55, symptoms are palpitations + chest pain + jaw pain → ACS"
# Clinical reasoning, not pattern matching
```

---

## 4. PROMPT ENGINEERING DIFFERENCES

### Local Implementation
- Multiple specialized prompts for different passes
- Pass 1: Focused on synthesis diagnosis queries (lines 1440-1560)
- Pass 2: RAG context + diagnostic reasoning
- Pass 3: Integration and final decision
- **Problem**: Complex multi-pass system vulnerable to compounding errors

### Reference Implementation
```
SINGLE comprehensive prompt with:
1. Vital Interpretation Reference (HR/RR/BP/O2/Temp normal ranges)
2. ESI Classification Rules (1=Emergent, 5=Minor)
3. Major Presentation Protocols (ACS, Sepsis, Shock, Neuro)
   - ACS: "ECG 12-lead within 10 min, troponin STAT, IV 18G+, aspirin 325mg PO"
   - Sepsis: "Blood cultures STAT ×2, antibiotics <1 hour, IV bolus 30mL/kg"
   - Shock: "IV access ×2, fluid bolus 500-1000mL, vasopressor if SBP <90"
   - Neuro: "Blood glucose STAT, CT head STAT, neuro consult, NPO"
4. Enforcement Rule: "All interventions must specify dose, route, volume, timeframe"
5. Response Format: JSON with {esi_level, color, interventions, reasoning}

Single pass → better consistency
```

---

## 5. DATA FLOW DIFFERENCES

### Local Data Flow
```
Patient Transcript (free text)
    ↓
Regex extraction (age, sex, chief complaint)
    ↓
[Pass 1] Synthesis diagnosis queries via LLM
    ↓
RAG retrieval from knowledge graph
    ↓
[Pass 2] LLM agentic supervisor with RAG context
    ↓
Red flag keyword detection (hardcoded lists)
    ↓
[Pass 3] Specialty decision + final triage
    ↓
Output: JSON with diagnoses + RAG citations
```

### Reference Data Flow
```
Structured JSON Input:
  - patient: {id, name, age, gender, medicalHistory, symptoms}
  - vitals: {temperature, heartRate, respiratoryRate, bloodPressure, oxygenSaturation}
    ↓
Extract chief complaint (priority: symptoms → presentingComplaint → chiefComplaint)
    ↓
Calculate TEWS baseline
    ↓
[Single Pass] Direct LLM call with comprehensive prompt
    - Vital interpretation rules
    - ESI classification criteria
    - Evidence-based protocols
    - Enforcement rules
    ↓
Parse JSON response: {esi_level, color, interventions, reasoning}
    ↓
Fallback (if LLM fails): _generate_default_interventions() based on TEWS + ESI level
    ↓
Return structured output
```

---

## 6. CRITICAL ARCHITECTURAL CONFLICTS

### Conflict 1: RAG vs. LLM-Only
- **Local**: Depends on knowledge graph retrieval (slow, requires data, network calls possible)
- **Reference**: Direct LLM inference (fast, 100% offline, HIPAA compliant)
- **Winner**: Reference (offline-first is requirement)

### Conflict 2: Keyword Patterns vs. Clinical Reasoning
- **Local**: Hardcoded 200+ keywords to detect red flags
- **Reference**: LLM reads vitals + context and makes clinical judgment
- **Winner**: Reference (user's explicit requirement: "don't hardcode")

### Conflict 3: Unstructured vs. Structured Input
- **Local**: Expects free-form transcript text
- **Reference**: Expects structured JSON with explicit fields
- **Winner**: Reference (frontend likely sends JSON, not transcripts)

### Conflict 4: Multi-Pass vs. Single-Pass
- **Local**: 3 passes (synthesis → supervisor → specialty) = slower, more error-prone
- **Reference**: 1 pass = deterministic, faster, easier to debug
- **Winner**: Reference (clinical precision > RAG sophistication)

### Conflict 5: Generic Recommendations vs. Specific Interventions
- **Local**: RAG returns case studies; LLM generates generic advice
- **Reference**: LLM enforces "Every intervention must specify dose, route, volume, timeframe"
- **Winner**: Reference (user requirement: "specific units of measurements")

---

## 7. KEY CODE SECTIONS MISSING IN LOCAL

The **reference implementation** includes these critical sections **NOT in local file**:

### 1. Structured Input Handling
```python
# Reference has explicit patient/vitals JSON parsing:
@app.route('/llm-triage', methods=['POST'])
def llm_triage():
    data = request.json
    patient = data['patient']  # {id, name, age, gender, medicalHistory, symptoms}
    vitals = data['vitals']     # {temp, hr, rr, bp, o2}
```

**Local equivalent**: Free-form text parsing with regex

### 2. Chief Complaint Priority Extraction
```python
# Reference (correct priority order):
chief_complaint = (
    patient.get('symptoms') or 
    patient.get('presentingComplaint') or 
    data.get('chiefComplaint')
)
```

**Local equivalent**: Regex pattern matching on transcript

### 3. Adaptive GGUF Configuration
```python
# Reference has calculate_adaptive_gguf_config():
if ram_gb < 4:
    return {'n_ctx': 2500, 'n_batch': 256}
elif ram_gb <= 12:
    return {'n_ctx': 3000, 'n_batch': 512}  # Mid-range (user's preference)
else:
    return {'n_ctx': 4096, 'n_batch': 1024}
```

**Local equivalent**: Fixed configuration, not adaptive

### 4. Enforcement of Specific Interventions
```python
# Reference prompt includes:
"All interventions must specify: dose (mg/mL), route (PO/IV/IM/SC), 
volume (mL), timeframe (STAT/within X minutes)"

"Example GOOD: 'ECG 12-lead within 10 minutes'"
"Example BAD: 'obtain ECG'"
```

**Local equivalent**: No such enforcement; generates generic recommendations

### 5. Evidence-Based Protocol Library
```python
# Reference includes specific protocols:
ACS: "ECG 12-lead within 10 min, troponin STAT, IV 18G+, aspirin 325mg PO, ..."
Sepsis: "Blood cultures STAT ×2, antibiotics <1 hour, IV bolus 30mL/kg, ..."
Shock: "IV access ×2, fluid bolus 500-1000mL, vasopressor if SBP <90, ..."
Neuro: "Blood glucose STAT, CT head STAT, neuro consult, NPO, ..."
```

**Local equivalent**: RAG-based case retrieval (indirect, slow)

### 6. Fallback Function
```python
# Reference has _generate_default_interventions(triage_data, esi_level):
# ESI-1: STAT interventions (airway/circulation management)
# ESI-2: Urgent labs/imaging/monitoring
# ESI-3: Standard procedures
# ESI-4/5: Basic care
```

**Local equivalent**: Synthetic query generation (different purpose)

### 7. Offline-First Configuration
```python
# Reference sets environment variables:
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_CACHE'] = './model_cache/'
```

**Local equivalent**: No offline-first optimization

### 8. Thread-Safe GGUF Access
```python
# Reference uses semaphore:
gguf_semaphore = threading.Semaphore(1)

def llm_triage():
    with gguf_semaphore:
        output = gguf_model.generate(prompt, max_tokens=1024)
```

**Local equivalent**: Flask default (concurrent requests may cause issues)

---

## 8. PROMPT COMPARISON

### Local Prompt Structure (Pass 1)
```
Focus: Generating synthesis diagnosis queries to find rare diseases
Purpose: Query optimization for RAG retrieval
Length: ~300-400 tokens
Output: JSON with {search_queries, primary_hypothesis, confidence}
```

### Reference Prompt Structure
```
1. Vital Interpretation Reference (100 tokens)
   - Normal ranges for HR, RR, BP, O2, Temp
   - When abnormal, what it indicates

2. ESI Classification Rules (150 tokens)
   - ESI-1: Requires immediate intervention
   - ESI-2: High-risk, STAT procedures
   - ESI-3: Moderate, urgent workup
   - ESI-4: Low-risk stable
   - ESI-5: Minor

3. Major Presentation Protocols (700 tokens)
   - ACS: Specific ECG, troponin, IV access, medications
   - Sepsis: Blood cultures, antibiotics, fluid bolus
   - Shock: IV lines, fluid bolus, vasopressors
   - Neuro: Blood glucose, CT, neuro consult, NPO

4. Enforcement Rules (150 tokens)
   - All interventions must specify dose, route, volume, timeframe
   - No generic recommendations
   - Clinical specificity required

5. Response Format (100 tokens)
   - JSON only
   - Fields: esi_level, color, interventions, reasoning

Total: ~1,200 tokens (fits in 2500-token context with 1300-token buffer for output)
```

---

## 9. WHICH VERSION IS CORRECT?

### Local Implementation Is Good For:
- Finding rare diagnoses through RAG synthesis
- Complex multi-system differential diagnosis
- Medical education/research
- When network/knowledge graph is available

### Reference Implementation Is Good For:
- **Acute emergency triage in hospital ED**
- Deterministic, reproducible clinical decisions
- Offline operation (HIPAA compliance)
- Speed (single pass vs. multi-pass)
- Hardware-constrained systems
- Specific intervention generation with exact measurements
- What the user explicitly requested

### User's Expressed Preference (from conversation):
> "the clinical interventions should be extremely clinical and life saving specific medical decisions with specific units of measurements of pharmaceutical resources, tools, medical physical techniques"

> "dont hardcode stress test examples that's cheating. remove hardcoded stress test triggers in prompt"

> "lets make sure we are extremely prioritizing life saving interventions"

**→ This clearly indicates the REFERENCE implementation is the intended direction.**

---

## 10. RECOMMENDED ACTION

The local file and reference prompt are **fundamentally incompatible**. You must choose:

### Option A: Keep Local (RAG-Based)
- Accept the hardcoding
- Maintain multi-pass approach
- Rely on knowledge graph availability
- Complex prompt engineering required

### Option B: Replace with Reference (LLM-Only)
- Single pass, deterministic
- Offline-first operation
- Specific interventions with exact measurements
- Simpler, more maintainable
- Aligns with all user requests

### Option C: Hybrid (Not Recommended)
- Mixing architectures creates confusion and technical debt
- Would require substantial refactoring anyway

**Based on conversation history, Option B (Reference implementation) is clearly the intended direction.**

---

## 11. CRITICAL MISSING IMPLEMENTATION DETAILS

If you want to implement the **Reference architecture**, these are the must-have components:

### Priority 1: Structure Change
- [ ] Replace free-form transcript parsing with JSON structure (patient + vitals)
- [ ] Implement chief complaint priority extraction
- [ ] Add explicit ESI level classification logic

### Priority 2: Prompt Engineering  
- [ ] Replace multi-pass prompts with single comprehensive prompt
- [ ] Add vital interpretation reference section
- [ ] Add evidence-based protocol section
- [ ] Add enforcement rules for specific interventions
- [ ] Remove all RAG references from prompt

### Priority 3: Configuration
- [ ] Implement adaptive GGUF configuration (calculate_adaptive_gguf_config)
- [ ] Set offline-first environment variables
- [ ] Add thread-safe semaphore for concurrent requests

### Priority 4: Fallback Logic
- [ ] Implement _generate_default_interventions(triage_data, esi_level)
- [ ] Remove hardcoded red flag detection
- [ ] Use LLM-based reasoning instead of keyword matching

### Priority 5: Testing
- [ ] Test ACS case (72yo diabetic: palpitations, BP 88/55, HR 118)
- [ ] Test Sepsis case (fever, altered mental status, lactate)
- [ ] Test Shock case (SBP 78, HR 145, weak perfusion)
- [ ] Test Neuro case (facial droop, arm weakness, speech difficulty)
- [ ] Validate interventions include exact doses, routes, volumes, timeframes

---

## Bottom Line

The **local ESI_Engine.py is an older, RAG-focused implementation**. The **reference in your prompt is the newer, LLM-only specification** aligned with your explicit requirements for clinical specificity, offline operation, and removal of hardcoding.

**To move forward, you need to either:**
1. **Merge**: Keep local RAG architecture but remove hardcoding (would still conflict with reference)
2. **Migrate**: Replace local with reference implementation (recommended)
3. **Validate**: Confirm the local RAG approach actually meets your requirements (it doesn't based on conversation history)
