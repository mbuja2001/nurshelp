#!/usr/bin/env python3
"""
Test the improved GGUF triage prompt with instruction-tuned format.
This script triggers a real triage request to verify the fixes work.
"""

import requests
import json
import sys
from datetime import datetime

# Configuration
BACKEND_URL = "http://localhost:5001"
TRIAGE_ENDPOINT = f"{BACKEND_URL}/api/llm-triage"

# Test patient with historical encounters
test_patient = {
    "patient": {
        "id_number": "SYNTH-HISTORY-000011",
        "firstName": "Jane",
        "lastName": "Smith",
        "age": 28,
        "sex": "F",
        "painLevel": "Reports moderate pain rated 7/10",
        "criticalHistoryStatus": "established",
        "criticalHistory": {
            "allergies": ["Penicillin (anaphylaxis)"],
            "medications": ["Lisinopril 10mg daily", "Metoprolol 50mg BID"],
            "conditions": ["Hypertension", "Type 2 Diabetes"],
            "procedures": ["C-section 2023"]
        }
    },
    "vitals": {
        "temperature": 37.2,
        "hr": 92,
        "bp_systolic": 142,
        "bp_diastolic": 88,
        "resp": 16,
        "o2": 97,
        "avpu": "Alert"
    },
    "patientText": "Acute onset chest discomfort with mild dyspnea. Denies palpitations. No recent illness. Wife concerned about cardiac risk.",
    "computeMode": "llm_only",
    "ragEnabled": False,
    "triageNurseId": "NURSE-TEST-001",
    "hospitalId": "HOSP-TEST"
}

print("=" * 80)
print("TEST: Improved GGUF Triage Prompt Format")
print("=" * 80)
print(f"\nTest Time: {datetime.now().isoformat()}")
print(f"Patient: {test_patient['patient']['firstName']} {test_patient['patient']['lastName']} ({test_patient['patient']['age']}yo {test_patient['patient']['sex']})")
print(f"Chief Complaint: {test_patient['patientText'][:80]}...")
print(f"Vitals: {test_patient['vitals']['bp_systolic']}/{test_patient['vitals']['bp_diastolic']} | HR {test_patient['vitals']['hr']} | RR {test_patient['vitals']['resp']} | O2 {test_patient['vitals']['o2']}%")

print("\n" + "=" * 80)
print("SENDING TRIAGE REQUEST")
print("=" * 80)

try:
    response = requests.post(
        TRIAGE_ENDPOINT,
        json=test_patient,
        timeout=180  # 3 minute timeout for GGUF inference
    )
    
    print(f"\nResponse Status: {response.status_code}")
    print(f"Response Time: {datetime.now().isoformat()}")
    
    if response.status_code == 200:
        result = response.json()
        
        # Extract key information
        ai_report = result.get('data', {}).get('aiTriageReport', {})
        esi_level = ai_report.get('urgencyScore')
        sats_level = ai_report.get('satsLevel')
        specialty = ai_report.get('specialty')
        confidence = ai_report.get('confidence')
        clinical_summary = ai_report.get('clinical_summary')
        
        print(f"\n✅ TRIAGE RESULT:")
        print(f"  ESI Level: {esi_level}")
        print(f"  SATS Level: {sats_level}")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Specialty: {specialty}")
        print(f"  Clinical Summary: {clinical_summary[:200]}...")
        
        # Check for JSON output quality
        differential = ai_report.get('differential_diagnosis', [])
        discriminators = ai_report.get('clinical_discriminators', [])
        recommendations = ai_report.get('recommendations', [])
        referenced_enc = ai_report.get('referenced_historical_encounters', [])
        ignored_enc = ai_report.get('ignored_historical_encounters', [])
        
        print(f"\n📊 QUALITY METRICS:")
        print(f"  Differential Diagnoses: {len(differential)} items")
        if differential:
            for i, dx in enumerate(differential[:2], 1):
                print(f"    {i}. {dx.get('dx', dx) if isinstance(dx, dict) else dx}")
        
        print(f"  Clinical Discriminators: {len(discriminators)} items")
        if discriminators:
            for i, d in enumerate(discriminators[:2], 1):
                print(f"    {i}. {d}")
        
        print(f"  Recommendations: {len(recommendations)} items")
        if recommendations:
            for i, r in enumerate(recommendations[:2], 1):
                print(f"    {i}. {r}")
        
        print(f"\n📚 HISTORICAL CONTEXT:")
        print(f"  Referenced Encounters: {len(referenced_enc)} items")
        if referenced_enc:
            print(f"    {referenced_enc}")
        print(f"  Ignored Encounters: {len(ignored_enc)} items")
        if ignored_enc:
            print(f"    {ignored_enc}")
        
        # Determine success
        if esi_level and sats_level and differential and len(discriminators) > 0:
            print(f"\n✅ SUCCESS: GGUF generated complete JSON with all required fields")
            sys.exit(0)
        else:
            print(f"\n⚠️  PARTIAL SUCCESS: Some fields missing")
            print(f"   ESI: {bool(esi_level)}, SATS: {bool(sats_level)}, Diff: {bool(differential)}, Disc: {len(discriminators) > 0}")
            sys.exit(1)
    else:
        print(f"\n❌ ERROR: HTTP {response.status_code}")
        print(f"Response: {response.text[:500]}")
        sys.exit(1)

except requests.exceptions.Timeout:
    print(f"\n⏱️  TIMEOUT: GGUF inference took too long (>180s)")
    sys.exit(2)

except Exception as e:
    print(f"\n❌ EXCEPTION: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(3)

print("\n" + "=" * 80)
