/**
 * CLINICAL OPERATIONS ENDPOINTS - TESTING & EXAMPLES
 * 
 * These endpoints support core clinical workflows:
 * 1. Real-time staff management (clocked-in staff)
 * 2. Bed assignment and hospitalization
 * 3. Staff assignment to encounters (physicians, nurses)
 * 4. Clinical handover tracking (edit history)
 * 
 * All endpoints require JWT authentication token
 */

// ============================================
// 1. GET CLOCKED-IN STAFF
// ============================================

/**
 * Get all clocked-in staff for a hospital
 * 
 * Endpoint: GET /api/clinical/staff/clocked-in/:hospitalId
 * 
 * Query Params (optional):
 *   - role: "PHYSICIAN" | "NURSE" | "TRIAGE_NURSE"
 * 
 * Response:
 *   {
 *     success: true,
 *     count: 5,
 *     hospital_id: "hospital123",
 *     staff: [
 *       {
 *         id: "staff123",
 *         name: "Dr. Chen",
 *         email: "chen@hospital.com",
 *         specialty: "Cardiology",
 *         role: "PHYSICIAN",
 *         department: "Cardiology",
 *         clock_status: "CLOCKED_IN",
 *         clock_in_time: "2024-01-15T08:00:00Z",
 *         last_break_end: "2024-01-15T12:30:00Z"
 *       }
 *     ]
 *   }
 * 
 * Examples:
 */

// Get all physicians clocked in
// GET /api/clinical/staff/clocked-in/hospital123?role=PHYSICIAN
// Headers: Authorization: Bearer <token>

// Get all nurses clocked in
// GET /api/clinical/staff/clocked-in/hospital123?role=NURSE

// Get all clocked-in staff (all roles)
// GET /api/clinical/staff/clocked-in/hospital123


// ============================================
// 2. GET AVAILABLE BEDS
// ============================================

/**
 * Get available beds for hospitalization
 * 
 * Endpoint: GET /api/clinical/beds/available/:hospitalId
 * 
 * Query Params (optional):
 *   - ward: "ICU" | "GENERAL" | "PEDIATRIC" | "MATERNITY"
 *   - status: "AVAILABLE" | "CLEAN" | "MAINTENANCE" (default: AVAILABLE, CLEAN)
 * 
 * Response:
 *   {
 *     success: true,
 *     count: 8,
 *     hospital_id: "hospital123",
 *     beds: [
 *       {
 *         _id: "bed123",
 *         bed_number: "ICU-01",
 *         ward: "ICU",
 *         bed_type: "Standard Hospital Bed",
 *         status: "AVAILABLE",
 *         is_occupied: false
 *       },
 *       {
 *         _id: "bed124",
 *         bed_number: "ICU-02",
 *         ward: "ICU",
 *         bed_type: "Standard Hospital Bed",
 *         status: "AVAILABLE",
 *         is_occupied: false
 *       }
 *     ]
 *   }
 * 
 * Examples:
 */

// Get all available beds in ICU
// GET /api/clinical/beds/available/hospital123?ward=ICU

// Get all available clean beds
// GET /api/clinical/beds/available/hospital123?status=CLEAN

// Get all available beds (any ward)
// GET /api/clinical/beds/available/hospital123


// ============================================
// 3. ASSIGN STAFF TO ENCOUNTER
// ============================================

/**
 * Assign physician or nurse to an encounter
 * 
 * Endpoint: POST /api/clinical/encounters/:encounterId/assign-staff
 * 
 * Request Body:
 *   {
 *     staffId: "staff123",
 *     role: "PHYSICIAN" | "NURSE"
 *   }
 * 
 * Response:
 *   {
 *     success: true,
 *     message: "PHYSICIAN assigned successfully",
 *     encounter_id: "encounter123",
 *     assigned_staff: {
 *       id: "staff123",
 *       name: "Dr. Chen",
 *       role: "PHYSICIAN"
 *     }
 *   }
 * 
 * Updates:
 *   - Sets encounter.assignedPhysicianId or encounter.assignedNurseId
 *   - Updates encounter.triage.assigned_physician or encounter.triage.assigned_nurse
 *   - Adds entry to encounter.assignment_edits tracking
 * 
 * Examples:
 */

// Assign physician to encounter
// POST /api/clinical/encounters/encounter123/assign-staff
// Headers: Authorization: Bearer <token>
// Body:
// {
//   "staffId": "staff123",
//   "role": "PHYSICIAN"
// }

// Assign nurse to encounter
// POST /api/clinical/encounters/encounter123/assign-staff
// Body:
// {
//   "staffId": "staff456",
//   "role": "NURSE"
// }


// ============================================
// 4. ASSIGN BED TO ENCOUNTER (Hospitalization/Discharge)
// ============================================

/**
 * Assign bed to encounter for hospitalization OR discharge patient
 * 
 * Endpoint: POST /api/clinical/encounters/:encounterId/assign-bed
 * 
 * Request Body for Hospitalization:
 *   {
 *     bedId: "bed123",
 *     hospitalize: true
 *   }
 * 
 * Request Body for Discharge:
 *   {
 *     bedId: null,
 *     hospitalize: false
 *   }
 * 
 * Response for Hospitalization:
 *   {
 *     success: true,
 *     message: "Patient hospitalized",
 *     encounter_id: "encounter123",
 *     hospitalized: true,
 *     bed_id: "bed123"
 *   }
 * 
 * Response for Discharge:
 *   {
 *     success: true,
 *     message: "Patient discharged",
 *     encounter_id: "encounter123",
 *     hospitalized: false,
 *     bed_id: null
 *   }
 * 
 * Side Effects:
 *   - Hospitalize:
 *     * Sets encounter.currentBedId = bedId
 *     * Sets encounter.isHospitalized = true
 *     * Sets encounter.hospitalization_date = now
 *     * Updates bed: is_occupied = true, current_patient_id = patient ID
 *     * Tracks in encounter.assignment_edits
 * 
 *   - Discharge:
 *     * Clears encounter.currentBedId
 *     * Sets encounter.isHospitalized = false
 *     * Sets encounter.discharge_date = now
 *     * Updates bed: is_occupied = false, current_patient_id = null
 *     * Tracks in encounter.assignment_edits
 * 
 * Examples:
 */

// Admit patient to bed
// POST /api/clinical/encounters/encounter123/assign-bed
// Headers: Authorization: Bearer <token>
// Body:
// {
//   "bedId": "bed123",
//   "hospitalize": true
// }

// Discharge patient from hospital
// POST /api/clinical/encounters/encounter123/assign-bed
// Body:
// {
//   "bedId": null,
//   "hospitalize": false
// }


// ============================================
// DATAFLOW EXAMPLE
// ============================================

/**
 * COMPLETE CLINICAL WORKFLOW WITH NEW ENDPOINTS
 * 
 * Scenario: Triage nurse sees that Dr. Chen is clocked in and available.
 *           Nurse wants to:
 *           1. Check available physicians
 *           2. Assign Dr. Chen to patient encounter
 *           3. Get available beds
 *           4. Admit patient to ICU bed
 * 
 * Step 1: Check available physicians
 * -----
 * GET /api/clinical/staff/clocked-in/hospital123?role=PHYSICIAN
 * Response includes: Dr. Chen (staff123) clocked in since 8:00 AM
 * 
 * Step 2: Assign Dr. Chen to patient encounter
 * -----
 * POST /api/clinical/encounters/encounter456/assign-staff
 * Body: { staffId: "staff123", role: "PHYSICIAN" }
 * Response: Encounter now has assignedPhysicianId = staff123
 *           assignment_edits logs this action with timestamp
 * 
 * Step 3: Get available ICU beds
 * -----
 * GET /api/clinical/beds/available/hospital123?ward=ICU
 * Response includes: ICU-01 (bed123) - AVAILABLE, not occupied
 * 
 * Step 4: Admit patient to ICU-01
 * -----
 * POST /api/clinical/encounters/encounter456/assign-bed
 * Body: { bedId: "bed123", hospitalize: true }
 * Response: Encounter now has:
 *   - currentBedId = bed123
 *   - isHospitalized = true
 *   - hospitalization_date = now
 *   - assignment_edits logs this admission
 * 
 * Bed123 now shows:
 *   - is_occupied = true
 *   - current_patient_id = patient ID
 * 
 * 
 * Later: Patient is ready for discharge
 * -----
 * POST /api/clinical/encounters/encounter456/assign-bed
 * Body: { bedId: null, hospitalize: false }
 * Response: Encounter now has:
 *   - currentBedId = null
 *   - isHospitalized = false
 *   - discharge_date = now
 *   - assignment_edits logs this discharge
 * 
 * Bed123 now shows:
 *   - is_occupied = false
 *   - current_patient_id = null
 * 
 * And bed is available again for next patient
 */


// ============================================
// CURL EXAMPLES - FOR TESTING
// ============================================

/**
 * 1. Get clocked-in physicians
 */
// curl -X GET \
//   "http://localhost:5000/api/clinical/staff/clocked-in/hospital123?role=PHYSICIAN" \
//   -H "Authorization: Bearer <your-jwt-token>" \
//   -H "Content-Type: application/json"

/**
 * 2. Get available ICU beds
 */
// curl -X GET \
//   "http://localhost:5000/api/clinical/beds/available/hospital123?ward=ICU" \
//   -H "Authorization: Bearer <your-jwt-token>" \
//   -H "Content-Type: application/json"

/**
 * 3. Assign physician to encounter
 */
// curl -X POST \
//   "http://localhost:5000/api/clinical/encounters/encounter123/assign-staff" \
//   -H "Authorization: Bearer <your-jwt-token>" \
//   -H "Content-Type: application/json" \
//   -d '{
//     "staffId": "staff123",
//     "role": "PHYSICIAN"
//   }'

/**
 * 4. Admit patient to bed
 */
// curl -X POST \
//   "http://localhost:5000/api/clinical/encounters/encounter123/assign-bed" \
//   -H "Authorization: Bearer <your-jwt-token>" \
//   -H "Content-Type: application/json" \
//   -d '{
//     "bedId": "bed123",
//     "hospitalize": true
//   }'

/**
 * 5. Discharge patient
 */
// curl -X POST \
//   "http://localhost:5000/api/clinical/encounters/encounter123/assign-bed" \
//   -H "Authorization: Bearer <your-jwt-token>" \
//   -H "Content-Type: application/json" \
//   -d '{
//     "bedId": null,
//     "hospitalize": false
//   }'


// ============================================
// JAVASCRIPT/FETCH EXAMPLES
// ============================================

// Helper: Get JWT token
async function getToken() {
  const response = await fetch("http://localhost:5000/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      email: "chen@hospital.com",
      password: "password123"
    })
  });
  const data = await response.json();
  return data.token;
}

// Get clocked-in physicians
async function getClockedInPhysicians(hospitalId) {
  const token = await getToken();
  const response = await fetch(
    `http://localhost:5000/api/clinical/staff/clocked-in/${hospitalId}?role=PHYSICIAN`,
    {
      headers: { "Authorization": `Bearer ${token}` }
    }
  );
  return response.json();
}

// Get available beds
async function getAvailableBeds(hospitalId, ward) {
  const token = await getToken();
  const response = await fetch(
    `http://localhost:5000/api/clinical/beds/available/${hospitalId}?ward=${ward}`,
    {
      headers: { "Authorization": `Bearer ${token}` }
    }
  );
  return response.json();
}

// Assign staff to encounter
async function assignStaffToEncounter(encounterId, staffId, role) {
  const token = await getToken();
  const response = await fetch(
    `http://localhost:5000/api/clinical/encounters/${encounterId}/assign-staff`,
    {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${token}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ staffId, role })
    }
  );
  return response.json();
}

// Admit patient to bed
async function admitPatientToBed(encounterId, bedId) {
  const token = await getToken();
  const response = await fetch(
    `http://localhost:5000/api/clinical/encounters/${encounterId}/assign-bed`,
    {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${token}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ bedId, hospitalize: true })
    }
  );
  return response.json();
}

// Discharge patient
async function dischargePatient(encounterId) {
  const token = await getToken();
  const response = await fetch(
    `http://localhost:5000/api/clinical/encounters/${encounterId}/assign-bed`,
    {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${token}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ bedId: null, hospitalize: false })
    }
  );
  return response.json();
}


// Workflow example: Complete hospital scenario
async function completeWorkflow() {
  const hospitalId = "hospital123";
  
  // 1. Get available physicians
  console.log("1. Fetching clocked-in physicians...");
  const physicians = await getClockedInPhysicians(hospitalId);
  console.log("Available physicians:", physicians.staff);
  
  // 2. Get available beds
  console.log("\n2. Fetching available ICU beds...");
  const beds = await getAvailableBeds(hospitalId, "ICU");
  console.log("Available beds:", beds.beds);
  
  // 3. Assign first physician
  if (physicians.staff.length > 0 && beds.beds.length > 0) {
    const physician = physicians.staff[0];
    const bed = beds.beds[0];
    const encounterId = "encounter123";
    
    // Assign physician
    console.log(`\n3. Assigning ${physician.name} to encounter...`);
    const assigned = await assignStaffToEncounter(encounterId, physician.id, "PHYSICIAN");
    console.log("Assignment result:", assigned);
    
    // Admit to bed
    console.log(`\n4. Admitting patient to bed ${bed.bed_number}...`);
    const admitted = await admitPatientToBed(encounterId, bed._id);
    console.log("Admission result:", admitted);
    
    // Discharge
    console.log("\n5. Discharging patient...");
    const discharged = await dischargePatient(encounterId);
    console.log("Discharge result:", discharged);
  }
}

// Run workflow
// completeWorkflow().catch(console.error);
