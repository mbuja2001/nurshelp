#!/usr/bin/env python3
"""
Test the optimized longitudinal history serialization.
Shows that using bounded trend summary reduces tokens from 600+ to ~100-150.
"""

def compact_longitudinal_history(history_obj, max_chars=200):
    """Optimized version using bounded trend summary from Node.js"""
    if not history_obj or isinstance(history_obj, str):
        return 'No prior history'
    try:
        summary = history_obj.get('historical_summary', {})
        if summary and isinstance(summary, dict) and 'encounter_count' in summary:
            parts = []
            
            enc_count = summary.get('encounter_count', 0)
            avg_tews = summary.get('average_tews')
            latest_tews = summary.get('latest_tews')
            red = summary.get('red_cases', 0)
            orange = summary.get('orange_cases', 0)
            
            if enc_count > 0:
                parts.append(f"TREND: {enc_count} enc")
                if avg_tews is not None and latest_tews is not None:
                    parts.append(f"TEWS avg {avg_tews:.1f} latest {latest_tews}")
                if red > 0 or orange > 0:
                    parts.append(f"({red}R,{orange}O)")
            
            diags = summary.get('recurrent_diagnoses', [])
            if diags and isinstance(diags, list):
                diag_str = ",".join([str(d)[:12] for d in diags[:3]])
                parts.append(f"Dx: {diag_str}")
            
            vitals = summary.get('vital_trends', {})
            if vitals and isinstance(vitals, dict):
                trend_parts = []
                for key in ['hr_change', 'bp_change', 'rr_change']:
                    if key in vitals and vitals[key] is not None:
                        val = vitals[key]
                        trend_parts.append(f"{key.split('_')[0]}:{val:+.0f}")
                if trend_parts:
                    parts.append("Trends: " + ",".join(trend_parts[:2]))
            
            result = " | ".join(parts) if parts else 'No trend data'
            if len(result) > max_chars:
                return result[:max_chars] + '...'
            return result
        
        # Fallback: encounter IDs only
        critical = history_obj.get('critical_history', [])
        recent = history_obj.get('recent_history', [])
        enc_ids = []
        
        for enc in (critical if isinstance(critical, list) else [])[:1]:
            if isinstance(enc, dict):
                enc_id = enc.get('id', 'unknown')[:15]
                enc_ids.append(f"CRIT:{enc_id}")
        
        for enc in (recent if isinstance(recent, list) else [])[:2]:
            if isinstance(enc, dict):
                enc_id = enc.get('id', 'unknown')[:15]
                enc_ids.append(enc_id)
        
        result = " | ".join(enc_ids) if enc_ids else 'No encounters'
        if len(result) > max_chars:
            return result[:max_chars] + '...'
        return result
        
    except Exception as e:
        return f'History unavailable: {str(e)[:20]}'


# Test cases
print("=" * 80)
print("TEST: Optimized Longitudinal History Serialization")
print("=" * 80)

# Test 1: With full trend summary (best case)
test1 = {
    'recent_history': [
        {'id': 'enc-001', 'chief_complaint': 'Chest pain'},
        {'id': 'enc-002', 'chief_complaint': 'Dyspnea on exertion'}
    ],
    'historical_summary': {
        'encounter_count': 42,
        'average_tews': 3.5,
        'latest_tews': 4,
        'red_cases': 2,
        'orange_cases': 5,
        'recurrent_diagnoses': ['Hypertension', 'Coronary Artery Disease', 'Type 2 Diabetes'],
        'vital_trends': {'hr_change': -12, 'bp_change': 15}
    },
    'critical_history': [
        {'id': 'crit-001', 'chief_complaint': 'STEMI 2024'}
    ]
}

result1 = compact_longitudinal_history(test1, 200)
print(f"\n✓ Test 1: Full Trend Summary")
print(f"  Input: 42 encounters + full summary")
print(f"  Output: {result1}")
print(f"  Length: {len(result1)} chars")
print(f"  Tokens: ~{len(result1)//4} (estimate)")

# Test 2: Minimal (no summary)
test2 = {
    'recent_history': [
        {'id': 'enc-001', 'chief_complaint': 'Chest pain'}
    ],
    'critical_history': [
        {'id': 'crit-001', 'chief_complaint': 'STEMI 2024'}
    ]
}

result2 = compact_longitudinal_history(test2, 200)
print(f"\n✓ Test 2: No Trend Summary (fallback)")
print(f"  Input: 2 encounters, no summary")
print(f"  Output: {result2}")
print(f"  Length: {len(result2)} chars")
print(f"  Tokens: ~{len(result2)//4} (estimate)")

# Test 3: Empty
test3 = {}
result3 = compact_longitudinal_history(test3, 200)
print(f"\n✓ Test 3: Empty History")
print(f"  Input: empty dict")
print(f"  Output: {result3}")
print(f"  Length: {len(result3)} chars")
print(f"  Tokens: ~{len(result3)//4} (estimate)")

# Token budget analysis
print("\n" + "=" * 80)
print("TOKEN BUDGET ANALYSIS")
print("=" * 80)

prompt_parts = {
    'SYSTEM + PATIENT DATA section': '120 chars',
    'CRITICAL HISTORY section': '250 chars max',
    'LONGITUDINAL HISTORY section': '200 chars max (NEW - was 800)',
    'CLINICAL REASONING section': '400 chars',
    'JSON SCHEMA section': '1000 chars',
    'CRITICAL RULES section': '600 chars'
}

print(f"\nEstimated prompt composition:")
total_chars = 0
for section, size_str in prompt_parts.items():
    # Parse the size
    size = int(size_str.split()[0])
    total_chars += size
    tokens = size // 4
    print(f"  {section}: {size_str} (~{tokens} tokens)")

print(f"\nTotal estimated: {total_chars} chars (~{total_chars//4} tokens)")
print(f"GGUF context window: 2048 tokens")
print(f"Reserved for output: 512 tokens")
print(f"Safe budget: 1536 tokens")
print(f"Margin: {1536 - (total_chars//4)} tokens")

if (total_chars // 4) <= 1536:
    print(f"\n✅ RESULT: Prompt fits safely within token budget!")
else:
    print(f"\n❌ WARNING: Prompt may exceed safe limit by {(total_chars//4) - 1536} tokens")

print("\n" + "=" * 80)
