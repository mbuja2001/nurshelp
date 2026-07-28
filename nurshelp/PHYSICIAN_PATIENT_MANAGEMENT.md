# Physician Patient Management Guide

## ✅ Seeded Patient Data Ready

**8 realistic patients** have been created and assigned to hospital beds:

### Emergency Room Patients (7)
| Patient | Bed | Age | ESI | Primary Complaint |
|---------|-----|-----|-----|---|
| John Miller | ED-Ward-A-02 | 65 | 2 | Chest pain, SOB |
| Maria Garcia | ED-Ward-A-03 | 42 | 2 | Severe abdominal pain |
| **Robert Chen** | ED-Ward-A-04 | 55 | **1** | **Acute stroke symptoms** |
| Sarah Johnson | ED-Ward-A-05 | 38 | 3 | Asthma exacerbation |
| David Williams | ED-Ward-A-06 | 72 | 2 | Fall with hip pain |
| Lisa Thompson | ED-Ward-A-07 | 48 | 3 | Acute migraine |
| Jennifer Martinez | ICU-Ward... | 29 | 3 | Laceration |

### Intensive Care Patients (1)
| Patient | Bed | Age | ESI | Primary Complaint |
|---------|-----|-----|-----|---|
| Michael Anderson | ICU-Ward-A-01 | 51 | 2 | Delirium, fever, COPD |

---

## 🏥 How to Use as a Physician

### 1. **Login as Physician**
```
Email: [Your physician email from registration]
Role: Physician (assigned via StaffScheduleManager)
```

### 2. **View Patient Dashboard**
When clocked in with Physician role, you'll see:
- **Physician Dashboard** with 3 lanes:
  - Active ER (patients in emergency room)
  - Observation (patients under monitoring)
  - Escalations (critical or red-flag patients)

### 3. **Patient Information Available**
For each patient, access:
- **Demographics**: Name, age, gender, MRN
- **Current Vitals**: Temperature, HR, BP, O2 sat, Respiratory rate
- **ESI Level**: 1-5 (1=most critical, 5=least critical)
- **Acuity Score**: 0-100 (higher = more acute)
- **Bed Location**: Current room/bed assignment
- **Clinical History**: Past medical history, current symptoms

### 4. **Make Disposition Decision**
For each patient, decide:
- **ADMITTED**: Patient needs inpatient bed → Escalate to ward
- **DISCHARGED**: Patient safe to go home
- **TRANSFERRED**: Patient needs specialty care → Transfer to facility

### 5. **Key Patients to Monitor**
⚠️ **Robert Chen (ESI 1)** - Acute stroke symptoms
- **Action Required**: STAT assessment
- **Vitals**: BP 165/105 (elevated), HR 98 (elevated for stroke)
- **Decision**: ADMITTED → ICU or Stroke Unit

⚠️ **Maria Garcia (ESI 2)** - Acute surgical abdomen
- **Action Required**: Rapid exam + labs
- **Vitals**: Temp 38.5°C (fever), HR 105 (tachycardic)
- **Decision**: ADMITTED → Surgical evaluations

---

## 📊 Data Fields in EHR

### Patient Demographics
- Name, surname, age, gender
- Medical Record Number (MRN)
- Encounter ID

### Clinical Assessment
- **ESI Level**: Emergency Severity Index (1-5 scale)
- **Acuity Score**: 0-100 clinical severity
- **Current Department**: ER, ICU, Ward
- **Admission Time**: When patient arrived

### Current Vitals (Most Recent)
- Temperature (°C)
- Heart Rate (bpm)
- Blood Pressure (systolic/diastolic)
- Respiratory Rate (breaths/min)
- O2 Saturation (%)
- Time recorded

### Vital Signs Thresholds (Alert if outside range)
- **Heart Rate**: 60-100 bpm (default)
- **BP Systolic**: 90-160 mmHg (default)
- **Respiratory Rate**: 12-20 breaths/min (default)
- **Customizable** per patient protocol

### Clinical Orders (if any)
- Medication orders
- Procedure orders
- Imaging orders
- Lab orders
- High-risk flags

---

## 🔧 Physician Workflow

**Phase 1: Review**
- View patient list in dashboard
- Click patient → See full EHR record

**Phase 2: Assess**
- Review vitals and alert flags
- Check ESI level and acuity score
- Review history and current symptoms

**Phase 3: Decide**
- Document clinical notes
- Make disposition decision (ADMITTED/DISCHARGED/TRANSFERRED)
- Record bed assignment or discharge instructions

**Phase 4: Monitor**
- Track patient progression
- Update vitals as reported
- Escalate if red flags emerge

---

## 🎯 Next Steps

1. **Login to Frontend** (http://localhost:5173)
2. **Clock In** as Physician
3. **View Physician Dashboard** → See all 8 seeded patients
4. **Make Disposition Decisions** for each
5. **Track Clinical Outcomes** (triage accuracy scoring)

---

## 📝 Notes

- Seeded data includes realistic vital signs
- ESI levels range 1-5 (mixed acuity)
- Patients are assigned to actual beds
- EHR records track admission time & current location
- Physician decisions automatically calculated for accuracy metrics

---

**Ready to manage patients! 🚑**
