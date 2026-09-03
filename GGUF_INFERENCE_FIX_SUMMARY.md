# GGUF Inference Fix Summary

## Problem Statement

GGUF model was returning the last instruction line from the prompt instead of generating JSON:

```
[llm-triage] ❌ CRITICAL: No opening brace { found in GGUF output
[llm-triage] 📄 FULL OUTPUT (81 chars):
[llm-triage] - If no life threats are ruled out, return [] in the life_threats_excluded array.
```

This caused the JSON parser to fail and fall back to TEWS-only scoring, preventing LLM reasoning.

## Root Causes Identified

1. **Aggressive Stop Sequence**: `stop=["\n\n"]` was terminating output after first line break
   - JSON output starts with `{` followed by first newline
   - Stop sequence would cut off generation before any JSON content
   
2. **Non-standard Prompt Format**: Prompt started with `[SYSTEM]` tag
   - MedGemma (instruction-tuned model) doesn't recognize this format
   - Model was confused about instruction boundaries
   
3. **Missing Chat Template**: Raw prompt sent to Llama without wrapping
   - No explicit context about JSON output requirement
   - Model interpreted prompt as instructions to echo back

## Fixes Applied

### 1. Fixed Stop Sequence (Critical)

**File**: `ESI_Engine.py` line ~615

**Before**:
```python
stop=["<end_of_turn>", "END", "\n\n"]  # Stop at natural boundaries
```

**After**:
```python
stop=["<end_of_turn>", "END"]  # Only stop at explicit boundaries, NOT newlines
```

**Rationale**: Remove `"\n\n"` which was cutting off JSON generation. Only use explicit end markers.

### 2. Improved Prompt Format (Critical)

**File**: `ESI_Engine.py` line ~8390

**Before**:
```
[SYSTEM] YOU MUST OUTPUT PURE JSON ONLY. NO PREAMBLE.

=== PATIENT DATA ===
...
=== CRITICAL RULES ===
...
```

**After**:
```
You are a clinical triage expert. Analyze the patient case and provide ONLY a valid JSON response with no preamble.

PATIENT DATA:
...
OUTPUT INSTRUCTIONS:
Generate valid JSON object with EXACTLY these fields. Start immediately with the opening brace.

{...}

Generate the JSON response now:
```

**Rationale**: 
- Use natural language instruction ("You are...") instead of `[SYSTEM]` tag
- Add explicit JSON schema in prompt
- End with "Generate the JSON response now:" to trigger output
- Simpler formatting that matches instruction-tuned model expectations

### 3. Added Pre-Send Debugging

**File**: `ESI_Engine.py` line ~8513

**Added**:
```python
# DEBUG: Log prompt before sending to GGUF
prompt_length = len(triage_prompt)
prompt_tokens_est = prompt_length // 4
print(f'[llm-triage] 📝 PROMPT ANALYSIS:')
print(f'[llm-triage]    Length: {prompt_length} chars')
print(f'[llm-triage]    Est. tokens: {prompt_tokens_est}')
print(f'[llm-triage]    First 150 chars: {triage_prompt[:150]}...')
print(f'[llm-triage]    Last 150 chars: ...{triage_prompt[-150:]}')
```

**Rationale**: Visibility into what's actually being sent to GGUF model.

## Token Budget Impact

The improved prompt is **slightly shorter** (~3420 chars vs ~3450 chars):
- Removed repetitive section headers (===SECTION===)
- Simplified bullet point format
- More concise instruction text

**Estimated tokens**: ~855 tokens (well under 1536 safe limit)

## Expected Outcome

1. ✅ GGUF generates complete JSON with all required fields
2. ✅ JSON includes `referenced_historical_encounters` and `ignored_historical_encounters` arrays
3. ✅ Clinical discriminators and differential diagnoses populated
4. ✅ No fallback to TEWS-only scoring
5. ✅ Better hospital-specific clinical reasoning

## Testing

Run test script to verify fix:
```bash
cd c:\Users\mbule\Videos\nursehelpdesk
python test_gguf_triage_improved.py
```

Expected output:
- HTTP 200 response
- ESI Level (1-5)
- SATS Level (red/orange/yellow/green)
- Differential diagnoses (2+ items)
- Clinical discriminators (3+ items)
- Referenced and ignored encounter arrays

## Safety Notes

- Fallback heuristic still active if GGUF fails
- No changes to safety-critical vital sign handling
- TEWS baseline used if JSON parsing fails
- All clinical validations preserved

## Files Modified

1. `nurshelp/backend/ML/ESI_Engine.py`
   - Line 615: Stop sequence fix
   - Line 8390: Prompt format improvement
   - Line 8513: Debug logging addition

2. Test files created:
   - `test_gguf_triage_improved.py` - Integration test

---

**Status**: Ready for testing with seeded patient data
**Recommended**: Run test after backend restart to allow GGUF model to load
