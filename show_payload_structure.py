#!/usr/bin/env python3
"""
Show what data structure we're receiving from buildInferenceHistoryPayload
"""

# Example payload structure from Node's buildInferenceHistoryPayload
example = {
    'recent_history': [
        {'id': 'rec-1', 'chief_complaint': 'Chest pain'},
        {'id': 'rec-2', 'chief_complaint': 'SOB'}
    ],
    'historical_summary': {
        'encounter_count': 42,
        'average_tews': 3.5,
        'latest_tews': 4,
        'red_cases': 2,
        'orange_cases': 5,
        'vital_trends': {'hr_change': -15, 'bp_change': '+20/+10'},
        'recurrent_diagnoses': ['HTN', 'CAD', 'DM'],
        'repeated_interventions': ['IV', 'O2'],
        'dispositions': ['ICU', 'Wards', 'Discharge']
    },
    'critical_history': [
        {'id': 'crit-1', 'chief_complaint': 'STEMI 2024', 'date': '2024-03-15'}
    ],
    'relevant_history': []
}

print('✓ Payload structure from buildInferenceHistoryPayload:')
print(f'  - recent_history: {len(example["recent_history"])} encounters')
print(f'  - historical_summary (TREND): {len(example["historical_summary"])} fields')
print(f'  - critical_history: {len(example["critical_history"])} encounters')
print()
print('✓ Trend summary (NOT raw encounters - already bounded):')
for k, v in example['historical_summary'].items():
    print(f'  {k}: {v}')
print()
print('✓ What to serialize for prompt:')
print('  OPTION 1: Use only trend summary + critical IDs')
print('    → 100-150 chars total')
print('    → Includes: encounter count, TEWS trends, vital changes, diagnoses')
print('    → LOSES: individual encounter details')
print()
print('  OPTION 2: Include recent encounter IDs + trend summary')
print('    → 200-300 chars total')
print('    → Includes: last 2-3 encounter IDs + trend summary')
print('    → LOSES: full encounter details but keeps references')
print()
print('✓ Current problem:')
print('  compact_longitudinal_history() is listing ALL encounters')
print('  Should use trend summary + critical encounter IDs instead')
print()
print('✓ Solution:')
print('  Change compact_longitudinal_history to format as:')
print('  "TREND: 42 encounters, avg TEWS 3.5, latest 4 (2 red, 5 orange)')
print('   Diags: HTN, CAD, DM | Recent: rec-1, rec-2 | Critical: crit-1"')
