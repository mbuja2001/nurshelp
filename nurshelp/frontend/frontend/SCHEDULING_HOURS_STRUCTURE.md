# Scheduling Hours & CSV Structure Analysis

## Current Implementation

### How Hours Are Currently Handled

```javascript
function generateAvailabilityMask(shiftStart, shiftEnd) {
  const mask = new Array(24).fill(0);  // 24-hour array
  
  if (!shiftStart || !shiftEnd) {
    return mask;
  }
  
  try {
    const [startHour] = shiftStart.split(":").map(Number);  // Extract HOUR only
    const [endHour] = shiftEnd.split(":").map(Number);
    
    for (let h = startHour; h < endHour && h < 24; h++) {
      mask[h] = 1;  // Mark each hour as available (1) or unavailable (0)
    }
  } catch (e) {
    // Invalid format, return empty mask
  }
  
  return mask;
}
```

**Current Approach:**
- Input: `shiftStart: "08:00"`, `shiftEnd: "16:00"`
- Output: `[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]`
- **Hours 8-15 are marked as 1 (available)**
- Hours 0-7 and 16-23 are marked as 0 (unavailable)

**Limitations:**
- ❌ Only works at HOUR level (ignores minutes)
- ❌ "08:30" start and "08:45" end both treated as hour 8
- ❌ Cannot handle breaks or split shifts
- ❌ Cannot track off-peak vs on-peak availability
- ❌ Cannot distinguish ON_DUTY (full availability) vs ON_CALL (reduced workload)

---

## Proposed Alternative CSV Structures

### Option 1: Current Simple Structure (KEEP THIS - Simple & Works)
```csv
physicianId,name,specialty,date,status,shiftStart,shiftEnd,maxPatients,notes
PHY001,John,Smith,Cardiology,2026-04-10,ON_DUTY,08:00,16:00,20,"Standard shift"
PHY002,Sarah,Johnson,ER,2026-04-10,ON_CALL,14:00,22:00,15,"Limited availability"
PHY003,Michael,Williams,Neurology,2026-04-10,OFF,,,0,""
```

**Pros:** Simple, human-readable, works for standard shifts
**Cons:** Cannot handle complex schedules

---

### Option 2: Enhanced with Break Times (RECOMMENDED FOR ESI)
```csv
physicianId,name,specialty,date,status,shiftStart,shiftEnd,breakStart,breakEnd,maxPatients,notes
PHY001,John,Smith,Cardiology,2026-04-10,ON_DUTY,08:00,16:00,12:00,13:00,20,"Lunch 12-1pm"
PHY002,Sarah,Johnson,ER,2026-04-10,ON_DUTY,08:30,17:00,12:15,12:45,22,"30min lunch"
PHY003,Michael,Williams,Neurology,2026-04-10,OFF,,,,,0,""
```

**Processing:**
```javascript
// Would mark: 08, 09, 10, 11, 13, 14, 15 (skip 12 for break)
mask[08]=1; mask[09]=1; mask[10]=1; mask[11]=1;  // 8am-12pm
mask[12]=0;  // BREAK
mask[13]=1; mask[14]=1; mask[15]=1;  // 1pm-4pm
```

**Pros:** Realistic healthcare scheduling
**Cons:** Slightly more complex

---

### Option 3: Granular 30-min Slots (MOST FLEXIBLE)
```csv
physicianId,name,specialty,date,status,availabilitySlots,maxPatients,notes
PHY001,John,Smith,Cardiology,2026-04-10,ON_DUTY,"08:00-12:00,13:00-16:00",20,"Break 12-1pm"
PHY002,Sarah,Johnson,ER,2026-04-10,ON_DUTY,"08:30-12:15,12:45-17:00",22,"30min lunch"
PHY003,Michael,Williams,Neurology,2026-04-10,ON_CALL,"18:00-22:00",10,"Night on-call only"
```

**Processing:**
```javascript
// Parse "08:00-12:00,13:00-16:00" into multiple blocks
// Each block can be as granular as needed (30min, 15min, etc.)
```

**Pros:** Handles any schedule complexity
**Cons:** More complex to parse and store

---

### Option 4: ESI-Aware with Workload Considerations (BEST FOR ESI ENGINE)
```csv
physicianId,name,specialty,date,status,shiftStart,shiftEnd,peakHours,offPeakHours,breakStart,breakEnd,onDutyCapacity,onCallCapacity,notes
PHY001,John,Smith,Cardiology,2026-04-10,ON_DUTY,08:00,16:00,"08:00,09:00,10:00,14:00,15:00","11:00,12:00,13:00",12:00,13:00,20,5,"Peak morning/afternoon"
PHY002,Sarah,Johnson,ER,2026-04-10,ON_CALL,22:00,07:00,,,,,10,3,"Night shift on-call"
```

**Features:**
- `peakHours`: When physician can take complex cases
- `offPeakHours`: When capacity is reduced
- `onDutyCapacity` vs `onCallCapacity`: Different max patients by status
- Integrates with ESI severity weighting

**Pros:** Optimizes for ESI triage routing
**Cons:** Most complex

---

## Recommendation for NurseHelp

### **Use Option 2: Enhanced with Break Times**

**CSV Format:**
```csv
physicianId,name,surname,email,specialty,date,status,shiftStart,shiftEnd,breakStart,breakEnd,maxPatients,notes
nurseId,name,surname,email,licenseNumber,date,status,shiftStart,shiftEnd,breakStart,breakEnd,maxBeds,esiLevel,notes
```

**Why:**
1. ✅ Realistic healthcare scheduling (breaks are mandatory)
2. ✅ Compatible with ESI engine (respects actual availability)
3. ✅ Simple to implement (just extend current code)
4. ✅ Scales well for manual triage bypass workflow
5. ✅ Can encode "peak hours" in notes if needed

---

## Required Code Changes

### Update Model Schema
Add break time fields:
```javascript
// physicianSchedule.model.js & nurseSchedule.model.js
breakStart: { type: String }, // e.g., "12:00"
breakEnd: { type: String },   // e.g., "13:00"
```

### Update Availability Mask Generator
```javascript
function generateAvailabilityMask(shiftStart, shiftEnd, breakStart, breakEnd) {
  const mask = new Array(24).fill(0);
  
  if (!shiftStart || !shiftEnd) return mask;
  
  try {
    const [startHour] = shiftStart.split(":").map(Number);
    const [endHour] = shiftEnd.split(":").map(Number);
    const [breakHourStart] = breakStart?.split(":").map(Number) ?? [null];
    const [breakHourEnd] = breakEnd?.split(":").map(Number) ?? [null];
    
    // Mark shift hours as available
    for (let h = startHour; h < endHour && h < 24; h++) {
      // Skip break hours
      if (breakHourStart !== null && h >= breakHourStart && h < breakHourEnd) {
        mask[h] = 0;
      } else {
        mask[h] = 1;
      }
    }
  } catch (e) {
    // Invalid format, return empty mask
  }
  
  return mask;
}
```

### Example Output with Break
Input:
- `shiftStart: "08:00"`, `shiftEnd: "16:00"`, `breakStart: "12:00"`, `breakEnd: "13:00"`

Output:
```
[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23
                              (BREAK)
```

**Available hours: 8am-12pm, 1pm-4pm** ✅

---

## CSV Import Example (Option 2 - Recommended)

### Physician CSV
```csv
physicianId,name,surname,email,specialty,date,status,shiftStart,shiftEnd,breakStart,breakEnd,maxPatients,notes
PHY001,John,Smith,john@hospital.com,Cardiology,2026-04-10,ON_DUTY,08:00,16:00,12:00,13:00,20,"Standard day shift"
PHY002,Sarah,Johnson,sarah@hospital.com,ER,2026-04-10,ON_DUTY,08:30,17:00,12:15,12:45,22,"30min lunch"
PHY003,Michael,Williams,michael@hospital.com,Neurology,2026-04-10,ON_CALL,22:00,07:00,01:00,02:00,10,"Night on-call with break"
PHY004,Lisa,Brown,lisa@hospital.com,Trauma,2026-04-10,OFF,,,,,0,""
```

### Nurse CSV
```csv
nurseId,name,surname,email,licenseNumber,date,status,shiftStart,shiftEnd,breakStart,breakEnd,maxBeds,esiLevel,certifications,notes
NUR001,Jane,Doe,jane@hospital.com,RN123456,2026-04-10,ON_DUTY,07:00,15:00,11:30,12:30,5,2,"ACLS,PALS,BLS","Standard day"
NUR002,Bob,Smith,bob@hospital.com,RN789012,2026-04-10,ON_DUTY,15:00,23:00,18:00,19:00,4,3,"BLS","Afternoon shift"
NUR003,Alice,Brown,alice@hospital.com,RN345678,2026-04-10,OFF,,,,,0,3,"","Call-out"
```

---

## Summary

| Aspect | Current | Recommended (Option 2) |
|--------|---------|------------------------|
| Granularity | Hourly | Hourly with breaks |
| Break handling | ❌ Not supported | ✅ Full support |
| ESI compatibility | ⚠️ Partial | ✅ Excellent |
| Complexity | Low | Low-Medium |
| Scalability | Medium | High |
| Patient bypass routing | ⚠️ Will work but inaccurate during breaks | ✅ Accurate assignment |
| Implementation cost | Zero | ~50 lines of code |

**Recommendation:** Implement Option 2 now, then add Option 3 (granular slots) in a future sprint if needed.
