# ST. MERCY HOSPITAL SCENARIO - QUICK REFERENCE SHEET

**Use this sheet to manually register the scenario**

---

## HOSPITAL DETAILS

```
Hospital Name: St. Mercy Hospital
Country: South Africa  
City: Cape Town
State: Western Cape
Zip: 8000
Phone: +27 21 555 0100
Email: admin@stmercy.hospital
```

---

## ADMIN MANAGER

```
Name: Alice
Surname: Johnson
Email: alice.johnson@stmercy.hospital
Password: alice_secure_123
Role: ADMIN
Department: All
```

---

## STAFF MEMBERS (4 people)

### Staff #1: Sarah (Triage Nurse)
```
First Name: Sarah
Last Name: Triage
Email: sarah.triage@stmercy.hospital
Password: sarah_secure_123
Role: NURSE
Department: ED (Emergency Department)
Specialty: Emergency Triage
License #: LIC-TRIAGE-001

CONTEXT IN SCENARIO: TRIAGE_NURSE
→ Processes patient queue
→ Performs triage assessments
→ Assigns clinical team
```

### Staff #2: Bob (Clinical Nurse)
```
First Name: Bob
Last Name: Martinez
Email: bob.martinez@stmercy.hospital
Password: bob_secure_123
Role: NURSE
Department: ED
Specialty: Clinical Nursing
License #: LIC-CLINICAL-001

CONTEXT IN SCENARIO: CLINICAL_NURSE_ER
→ Manages observation patients
→ Logs vitals
→ Escalates to physician
```

### Staff #3: Dr. Chen (Emergency Physician)
```
First Name: Dr. Chen
Last Name: MD
Email: dr.chen@stmercy.hospital
Password: chen_secure_123
Role: PHYSICIAN
Department: ED
Specialty: Emergency Medicine
License #: MD-CHEN-001

CONTEXT IN SCENARIO: PHYSICIAN_ER
→ Reviews all cases
→ Makes disposition decisions (DISCHARGE/ADMIT)
→ Detects triage discrepancies
```

### Staff #4: Ward Nurse
```
First Name: Ward
Last Name: Nurse
Email: ward.nurse@stmercy.hospital
Password: ward_secure_123
Role: NURSE
Department: Ward
Specialty: General Nursing
License #: LIC-WARD-001

CONTEXT IN SCENARIO: CLINICAL_NURSE_WARD
→ Manages hospitalized patients
→ Updates bed status
```

---

## SHIFT DETAILS

```
Shift Name: Tuesday ED Shift - April 20th
Date: 2026-04-20
Department: ED
Start Time: 07:00
End Time: 15:00
Duration: 8 hours

Assigned Staff:
✓ Sarah (Triage)
✓ Bob (Clinical)
✓ Dr. Chen (Physician)
✓ Ward Nurse (Ward)
```

---

## PATIENTS (4 people)

### Patient 1: John Smith
```
First Name: John
Last Name: Smith
Date of Birth: 03/15/1965
Age: 59
Gender: Male
Arrival Time: 07:15 AM

Chief Complaint: Acute chest pain

Vitals:
- Temperature: 37.2°C
- Heart Rate: 102 bpm (↑ elevated)
- Respiratory Rate: 20 breaths/min (↑ elevated)
- BP (Systolic/Diastolic): 158/92 mmHg (↑ elevated)
- Oxygen Saturation: 96%
- Pain Level: 8/10

Medical History:
- Hypertension (10 years)
- Smoking history

Current Medications:
- Lisinopril 10mg daily
- Aspirin 81mg daily

Allergies: NKDA (No Known Drug Allergies)

TRIAGE ASSESSMENT (Sarah):
- Priority: RED (high acuity)
- Reasoning: "Acute onset chest pain radiating to left arm, elevated vitals suggest possible cardiac event"
- Needs Bed? YES ✓
- Bed Reason: "Likely needs cardiac workup, probable admission for troponin levels and monitoring"

PHYSICIAN DECISION (Dr. Chen):
- Disposition: ADMIT to ICU/CCU
- Notes: "ACS protocol - likely NSTEMI. Needs continuous cardiac monitoring"
- Bed Assigned: ICU_05

OUTCOME: 
- Match: YES (Triage said needs bed → Physician admitted)
- Accuracy: TRUE POSITIVE ✓
- Metrics: Decision to Admit = 9m 30s
```

---

### Patient 2: Sarah Wong
```
First Name: Sarah
Last Name: Wong
Date of Birth: 07/22/1992
Age: 34
Gender: Female
Arrival Time: 07:35 AM

Chief Complaint: Ankle sprain from fall at home

Vitals:
- Temperature: 36.8°C (normal)
- Heart Rate: 78 bpm (normal)
- Respiratory Rate: 16 breaths/min (normal)
- BP: 120/80 mmHg (normal)
- Oxygen Saturation: 98% (normal)
- Pain Level: 5/10

Medical History: None significant
Current Medications: None
Allergies: NKDA

TRIAGE ASSESSMENT (Sarah):
- Priority: GREEN (low acuity)
- Reasoning: "Ankle sprain, vital signs normal, no signs of serious injury. Can be managed in observation."
- Needs Bed? NO
- Bed Reason: "Can be managed in observation, likely discharged same day after imaging"

PHYSICIAN DECISION (Dr. Chen):
- Disposition: DISCHARGE
- Notes: "Ankle sprain, X-rays show no fracture. Safe to discharge with follow-up care."

OUTCOME:
- Match: YES (Triage said no bed → Physician discharged)
- Accuracy: TRUE NEGATIVE ✓
- Patient outcome: Discharged to home
```

---

### Patient 3: Michael Chen
```
First Name: Michael
Last Name: Chen
Date of Birth: 11/05/1978
Age: 48
Gender: Male
Arrival Time: 08:15 AM

Chief Complaint: Mild fever and body aches

Vitals:
- Temperature: 38.2°C (↑ elevated - fever)
- Heart Rate: 85 bpm (normal)
- Respiratory Rate: 16 breaths/min (normal)
- BP: 125/81 mmHg (normal)
- Oxygen Saturation: 97% (normal)

Medical History: N/A
Current Medications: N/A
Allergies: NKDA

TRIAGE ASSESSMENT (Sarah):
- Priority: YELLOW (moderate - infection, needs evaluation)
- Reasoning: "Fever with body aches suggests viral infection. Vitals stable. Needs observation."
- Needs Bed? NO
- Bed Reason: "Appears to be viral infection, vitals stable, can go home with rest after observation"

CLINICAL OBSERVATION (Bob):
- Observation period: 08:15 AM - 09:30 AM (1 hour 15 min)
- Patient rests, vitals remain stable
- No deterioration observed
- Fever remains manageable

PHYSICIAN DECISION (Dr. Chen):
- Disposition: DISCHARGE
- Notes: "Stable vital signs, likely viral infection. Safe for discharge with home care instructions."
- Time: 09:35 AM

OUTCOME:
- Match: YES (Triage said no bed → Physician discharged)
- Accuracy: TRUE NEGATIVE ✓
- Patient outcome: Discharged to home
```

---

### Patient 4: Maria Rodriguez (THE TRICKY CASE)
```
First Name: Maria
Last Name: Rodriguez
Date of Birth: 08/12/1955
Age: 71
Gender: Female
Arrival Time: 11:30 AM

Chief Complaint: Mild headache and confusion for 2 hours

Vitals (at arrival):
- Temperature: 37.5°C (slightly elevated)
- Heart Rate: 88 bpm (normal)
- Respiratory Rate: 16 breaths/min (normal)
- BP: 138/85 mmHg (normal)
- Oxygen Saturation: 96% (normal)

Medical History:
- On warfarin (anticoagulation therapy) - CRITICAL
- Previous stroke (controlled)

Current Medications:
- Warfarin 5mg daily
- Metoprolol (beta blocker)

Allergies: NKDA

Additional Info:
- Recent fall yesterday (head strike noted by family)
- Slight bruising on left temporal area

TRIAGE ASSESSMENT (Sarah):
- Time: 11:33 AM
- Priority: YELLOW (moderate - needs evaluation)
- Reasoning: "Mild symptoms, vitals stable. Possibly dehydration and headache."
- Needs Bed? NO
- Bed Reason: "Mild symptoms, vitals stable, can observe and discharge"

⚠️ CRITICAL DISCOVERY (Bob - 11:45 AM):
- During observation, Bob more carefully reviews medications
- Discovers: ON WARFARIN (blood thinner)
- Discovers: Recent fall with head impact yesterday
- Notices: Patient confusion IS WORSENING ("feeling more confused now")
- Clinical concern: Potential INTRACRANIAL BLEED
- Bob escalates: "Physician review urgent"

PHYSICIAN DECISION (Dr. Chen):
- Time: 11:50 AM (17 minutes after triage)
- Assessment: "Warfarin + head trauma + worsening confusion = high risk for intracranial bleed"
- Disposition: ADMIT to ICU (neurology hold, CT head ordered)
- Notes: "Cannot safely discharge. Needs neuro monitoring and imaging."
- Bed Assigned: ICU_06

⚠️ DISCREPANCY DETECTED!
- Triage Assessment: NO BED NEEDED
- Physician Decision: ADMIT to ICU
- Type: FALSE NEGATIVE
- Root Cause: Medication interaction + neuro signs not caught in initial triage
- Cost Impact: $3,200 (potential re-admission cost if released too early)

QA ACTION ITEMS:
1. Flag Alice for coaching on medication interaction screening
2. Review triage protocol: warfarin + head trauma → auto-escalate
3. Training: Enhance neuro assessment competency
4. Add rule: Confusion + anticoagulation + recent trauma → RED priority

OUTCOME:
- Match: NO (Triage said no bed → Physician admitted)
- Accuracy: FALSE NEGATIVE ✗
- Learning: Triage accuracy system catches this discrepancy
- Improvement: Team huddle to discuss and prevent future false negatives
```

---

## SHIFT TIMELINE

```
7:00 AM  - Sarah, Bob, Dr. Chen, Ward Nurse CLOCK IN
7:15 AM  - John arrives (RED - chest pain)
7:18 AM  - Sarah triages John (needsBed: true)
7:20 AM  - Bob receives John in clinical mode
7:22 AM  - Bob escalates John to physician
7:25 AM  - Dr. Chen reviews and ADMITs John to ICU
         └─ TRUE POSITIVE ✓ (Triage match)

7:35 AM  - Sarah Wong arrives (GREEN - ankle)
7:38 AM  - Sarah triages Sarah Wong (needsBed: false)
7:40 AM  - Bob receives Sarah Wong in clinical mode
9:30 AM  - Dr. Chen DISCHARGEs Sarah Wong
         └─ TRUE NEGATIVE ✓ (Triage match)

8:15 AM  - Michael Chen arrives (YELLOW - fever)
8:18 AM  - Sarah triages Michael (needsBed: false)
9:35 AM  - Dr. Chen DISCHARGEs Michael
         └─ TRUE NEGATIVE ✓ (Triage match)

11:30 AM - Maria Rodriguez arrives (YELLOW - confusion)
11:33 AM - Sarah triages Maria (needsBed: false)
11:45 AM - Bob discovers: warfarin + head trauma + worsening confusion
11:50 AM - Dr. Chen reviews and ADMITs Maria to ICU
         └─ FALSE NEGATIVE ✗ (Triage mismatch - LEARNING OPPORTUNITY)

3:00 PM  - END OF SHIFT
         └─ Metrics: 75% accuracy (3/4), 1 false negative caught by system
```

---

## METRICS SUMMARY (End of Shift)

```
Total Patients Triaged: 4
Total Outcomes:
├─ Admitted: 2 (John → ICU, Maria → ICU)
├─ Discharged: 2 (Sarah Wong, Michael)
└─ Admission Rate: 50%

TRIAGE ACCURACY:
├─ TRUE POSITIVES: 1 (John)
├─ TRUE NEGATIVES: 2 (Sarah Wong, Michael)
├─ FALSE NEGATIVES: 1 (Maria)
├─ FALSE POSITIVES: 0
│
└─ Overall Accuracy: 75% (3 correct predictions out of 4)

DECISION LATENCY:
├─ John: 9m 30s (from triage to admission)
├─ Sarah Wong: 1h 52m (from arrival to discharge)
├─ Michael: 1h 17m (from arrival to discharge)
├─ Maria: 17m (from triage to admission)
│
└─ Average: 7m 23s (very efficient!)

PHYSICIAN RESPONSE TIME:
├─ John (RED): 4m 30s ✓ COMPLIANT (<5 min target)
├─ Sarah Wong (GREEN): Routine ✓
├─ Michael (YELLOW): Routine ✓
├─ Maria (escalated): 5m ✓ COMPLIANT
│
└─ All cases COMPLIANT

COST ANALYSIS:
├─ ICU admissions: 2 × $5,000/day = $10,000
├─ ED observation: 4 patients
├─ Prevented false discharge (Maria): Saved ~$3,200 readmission
│
└─ Value of triage system: Detection of 1 high-risk case

QA FLAGS:
├─ 1 False Negative (Maria)
├─ Root cause: Medication interaction not assessed
├─ Action: Coaching for Alice on neuro screening
└─ System learning: Improves model for future cases
```

---

## HOW TO USE THIS SHEET

### Option A: Manual Frontend Entry
1. Go to register page: `http://localhost:5181/register`
2. Enter hospital details (copy from HOSPITAL DETAILS section)
3. Enter admin manager (copy from ADMIN MANAGER section)
4. Add staff members one by one (copy from STAFF MEMBERS section)
5. Create shift (copy from SHIFT DETAILS section)
6. Each staff member logs in and clocks in
7. Add patients manually via patient intake form
8. Test triage workflow

### Option B: Create Seed Script
We can create a JavaScript seed file that:
1. Creates hospital
2. Creates all staff
3. Creates shift
4. Creates all patients
5. Runs with: `node backend/src/seeds/seedHospitalScenario.js`

This is 10x faster for testing!

---

## NEXT STEP

Ready to register this scenario? 

**Choose one**:
1. **Manual Registration** (realistic testing of UI flow)
2. **Seed Script** (rapid testing - 1 minute setup)

Let me know which you prefer! 🚀
