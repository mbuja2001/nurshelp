import mongoose from "mongoose";

const transcriptSchema = new mongoose.Schema({
  id: { type: Number },
  type: { type: String },
  text: { type: String }
}, { _id: false }); // avoid nested _id

const encounterSchema = new mongoose.Schema({
  hospitalId: { type: mongoose.Schema.Types.ObjectId, ref: "Hospital", default: null }, // Multi-tenant isolation (optional for backward compatibility)
  nurse_id: { type: mongoose.Schema.Types.ObjectId, ref: "Nurse", default: null },
  assignedNurseId: { type: mongoose.Schema.Types.ObjectId, ref: "Nurse", default: null },
  assignedPhysicianId: { type: String, default: null }, // Store as string since physician system may vary
  assignedNurseName: { type: String, default: null }, // Store display name directly
  assignedPhysicianName: { type: String, default: null }, // Store display name directly
  assigned_clinicians: [
    {
      id: mongoose.Schema.Types.ObjectId,
      name: String,
      role: { type: String, enum: ['PHYSICIAN', 'NURSE', 'SPECIALIST'] },
      specialty: String,
      score: Number,
      assigned_at: { type: Date, default: Date.now }
    }
  ],
  manually_selected_clinicians: [
    {
      id: mongoose.Schema.Types.ObjectId,
      name: String,
      role: { type: String, enum: ['PHYSICIAN', 'NURSE', 'SPECIALIST'] },
      specialty: String,
      score: Number,
      clock_in_time: Date,
      clock_out_time: Date,
      shift_start: Date,
      shift_end: Date,
      selected_at: { type: Date, default: Date.now }
    }
  ],
  assigned_ward: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Ward",
    default: null
  },
  assigned_bed: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "BedSchedule",
    default: null
  },
  ward_display_info: {
    _id: mongoose.Schema.Types.ObjectId,
    wardId: String,
    wardName: String,
    department: String,
    name: String,
    totalBeds: Number,
    occupiedBeds: Number,
    availableBeds: Number
  },
  bed_display_info: {
    _id: mongoose.Schema.Types.ObjectId,
    wardId: mongoose.Schema.Types.ObjectId,
    bedId: String,
    bedNumber: String,
    bed_number: String,
    roomNumber: String,
    department: String,
    bedType: String,
    status: String
  },
  patient: {
    id_number: String,
    name: String,
    surname: String,
    gender: String,
    age: Number,
    height_cm: Number,
    symptoms: String,
    duration: String,
    painLevel: String,
    history: String
  },
  transcript: { type: [transcriptSchema], default: [] }, // important: normalized array
  vitals: {
    temp: Number,
    bp: String,
    bp_left_systolic: Number,
    bp_left_diastolic: Number,
    bp_right_systolic: Number,
    bp_right_diastolic: Number,
    hr: Number,
    o2: Number,
    resp: Number,
    bp_systolic: Number,
    bp_diastolic: Number,
    avpu: { type: String }
  },
  // ===== TRIAGE REPORTS (Two-way system: AI-generated + Manual overrides) =====
  ai_triage_report: { type: Object, default: {} }, // Original AI-generated report (unchanged audit trail)
  manual_triage_report: {
    // Nurse's manual overrides/confirmations of AI fields
    ai_summary: String,
    assigned_specialty: String,
    assigned_physician: String,
    discriminators: [String],
    sats_points: Number,
    sats_color: String,
    recommendations: [String],
    notes: String,
    overridden_at: Date,
    overridden_by: mongoose.Schema.Types.ObjectId, // Reference to nurse who made override
    field_changes: { type: Object, default: {} } // Tracks which fields were changed: { field_name: { original, override } }
  },

  // Backward compatibility: Keep legacy triage field (maps to ai_triage_report)
  triage: { type: Object, default: {} }, // Legacy - will store full AI output for compatibility
  
  recommendations: { type: [String], default: [] }, // clinical recommendations
  nurseNotes: { type: String, default: "" },

  // ===== PHYSICIAN DIAGNOSIS & DISPOSITION =====
  diagnosis: { type: String, default: "" }, // Physician-documented diagnosis
  treatment_plan: { type: String, default: "" }, // Treatment plan & recommendations
  physician_notes: { type: String, default: "" }, // Additional clinical notes from physician
  assigned_physician: {
    // Physician who documented the diagnosis
    id: mongoose.Schema.Types.ObjectId,
    name: String,
    specialty: String,
    diagnosed_at: Date
  },

  // ===== TREATMENT PROGRESSION (Clinical trial audit trail) =====
  treatments: [
    {
      intervention_type: { type: String, enum: ['MEDICATION', 'PROCEDURE', 'THERAPY', 'MONITORING', 'IMAGING', 'LAB'], default: 'MEDICATION' },
      name: String, // e.g. "Ibuprofen", "Chest X-ray", "Blood draw"
      dosage: String, // e.g. "500mg"
      frequency: String, // e.g. "Every 4-6 hours"
      route: String, // e.g. "Oral", "IV", "IM"
      indication: String, // Why this treatment
      prescribed_by: String, // Physician name
      prescribed_at: { type: Date, default: Date.now },
      administered_at: Date,
      completed_at: Date,
      status: { type: String, enum: ['PRESCRIBED', 'ADMINISTERED', 'COMPLETED', 'HOLD', 'DISCONTINUED'], default: 'PRESCRIBED' },
      outcome: String, // Patient response
      notes: String
    }
  ],

  // ===== RECOMMENDATIONS & ACTIONS =====
  recommendations: [
    {
      recommendation_type: { type: String, enum: ['ADMISSION', 'TRANSFER', 'SPECIALTY_CONSULT', 'FOLLOW_UP', 'DISCHARGE_INSTRUCTION', 'LIFESTYLE_CHANGE', 'DIAGNOSTIC_TEST'], default: 'ADMISSION' },
      description: String,
      priority: { type: String, enum: ['URGENT', 'HIGH', 'MEDIUM', 'LOW'], default: 'MEDIUM' },
      recommended_by: String, // Physician name
      recommended_at: { type: Date, default: Date.now },
      due_date: Date,
      status: { type: String, enum: ['PENDING', 'IN_PROGRESS', 'COMPLETED', 'DECLINED'], default: 'PENDING' },
      completion_notes: String,
      completed_at: Date
    }
  ],

  medication_requests: [
    {
      medication_name: String,
      dosage: String,
      frequency: String,
      route: String,
      indication: String,
      requested_by: String,
      prescriptionId: String,
      status: { type: String, enum: ['PENDING', 'APPROVED', 'DISPENSED', 'DECLINED'], default: 'PENDING' },
      requested_at: { type: Date, default: Date.now },
      reviewed_at: Date,
      review_notes: String,
      pharmacy_reference: String
    }
  ],

  // ===== MEDICATION ADMINISTRATION RECORDS =====
  medication_administrations: [
    {
      medication_name: String,
      medication_id: mongoose.Schema.Types.ObjectId,
      dosage: String,
      frequency: String,
      route: { type: String, enum: ['Oral', 'IV', 'IM', 'SC', 'Topical', 'Inhalation'], default: 'Oral' },
      indication: String,
      prescription_notes: String,
      administered_at: { type: Date, default: Date.now },
      recorded_at: { type: Date, default: Date.now },
      status: { type: String, enum: ['ADMINISTERED', 'PENDING', 'HELD', 'DECLINED'], default: 'ADMINISTERED' },
      administered_by: String,
      prescribed_by_role: String,
      patient_response: String
    }
  ],

  // ===== SURGERY RECORDS =====
  surgery_records: [
    {
      surgery_id: { type: mongoose.Schema.Types.ObjectId, auto: true },
      procedure_name: { type: String, required: true },
      surgery_type: { type: String, enum: ['ELECTIVE', 'EMERGENCY', 'URGENT'], default: 'ELECTIVE' },
      rationale: String,
      priority: { type: String, enum: ['URGENT', 'HIGH', 'MEDIUM', 'LOW'], default: 'MEDIUM' },
      scheduled_at: Date,
      surgeon: String,
      surgical_team: [String],
      requested_by: String,
      requested_at: { type: Date, default: Date.now },
      status: { type: String, enum: ['PENDING', 'SCHEDULED', 'IN_PROGRESS', 'COMPLETED', 'POST_OP_REVIEW', 'COMPLICATION', 'CANCELLED'], default: 'PENDING' },
      performed_at: Date,
      outcome: String,
      findings: String,
      complications: String,
      notes: String,
      recorded_by: String,
      recorded_at: { type: Date, default: Date.now },
      updated_at: { type: Date, default: Date.now }
    }
  ],

  // ===== LAB REQUESTS (Radiology, Pathology workflow) =====
  lab_requests: [
    {
      request_id: { type: mongoose.Schema.Types.ObjectId, auto: true },
      specialty: { type: String, enum: ['RADIOLOGY', 'PATHOLOGY'], required: true },
      lab_type: String, // e.g., "Chest X-ray", "CT Scan", "Blood Work", "Urinalysis"
      description: String, // Clinical indication for the test
      priority: { type: String, enum: ['URGENT', 'HIGH', 'MEDIUM', 'LOW'], default: 'MEDIUM' },
      
      // Request creation
      requested_by: String, // Physician name
      requested_at: { type: Date, default: Date.now },
      
      // Acknowledgment stage
      assigned_technician: String, // Radiologist/Pathologist who accepted
      assigned_at: Date,
      
      // Work & completion
      report_content: String, // Findings/observations from lab
      report_images: [String], // URLs to uploaded images or documents
      clinical_impression: String, // Specialist's clinical conclusion
      recommendations: [String], // Follow-up recommendations
      submitted_at: Date,
      submitted_by: String, // Specialist who submitted report
      
      // Status tracking
      status: { type: String, enum: ['PENDING', 'ACKNOWLEDGED', 'IN_PROGRESS', 'COMPLETED', 'REVIEWED', 'DECLINED'], default: 'PENDING' },
      
      // Physician review
      reviewed_at: Date,
      reviewed_by: String, // Physician who reviewed the report
      review_notes: String, // Physician's clinical interpretation
      
      // Metadata
      created_at: { type: Date, default: Date.now },
      updated_at: { type: Date, default: Date.now }
    }
  ],

  // ===== PROGRESSION NOTES (Physician documentation over time) =====
  progression_notes: [
    {
      note_type: { type: String, enum: ['PROGRESS', 'ASSESSMENT', 'PLAN_UPDATE', 'COMPLICATION', 'OUTCOME'], default: 'PROGRESS' },
      content: String,
      recorded_by: String, // Physician name
      recorded_at: { type: Date, default: Date.now },
      vitals_snapshot: {
        // Vitals at time of note
        bp: String,
        hr: Number,
        temp: Number,
        o2: Number,
        resp: Number
      },
      patient_response: String // How patient is responding
    }
  ],

  // ===== CLINICAL HANDOVER TRACKING =====
  clinical_handovers: [
    {
      clinician_id: mongoose.Schema.Types.ObjectId, // Reference to assigned clinician
      clinician_name: String,
      clinician_role: { type: String, enum: ['PHYSICIAN', 'NURSE', 'SPECIALIST'] },
      clinician_specialty: String,
      handover_type: { type: String, enum: ['INITIAL', 'UPDATE', 'FOLLOW_UP'], default: 'INITIAL' },
      handover_status: { 
        type: String, 
        enum: ['NOTIFIED', 'VIEWED', 'ACKNOWLEDGED', 'IN_PROGRESS', 'COMPLETED'], 
        default: 'NOTIFIED' 
      },
      handover_at: { type: Date, default: Date.now },
      viewed_at: Date, // When clinician first viewed the report
      acknowledged_at: Date, // When clinician acknowledged the handover
      acknowledged_notes: String, // Clinician's acknowledgment notes
      report_type: { type: String, enum: ['MANUAL', 'AI'], default: 'AI' }, // Which report version they received
      is_active: { type: Boolean, default: true } // Whether this clinician is still assigned
    }
  ],
  
  handover_completed: { type: Boolean, default: false }, // All assigned clinicians have been notified
  handover_completed_at: Date,

  status: { 
    type: String, 
    enum: [
      "TRIAGE_IN_PROGRESS",      // Nurse is triaging
      "OFFICIAL_REPORT",         // AI report generated, awaiting confirmation
      "AWAITING_PHYSICIAN",      // Handover sent, awaiting physician to see
      "SEEN",                    // Physician has viewed and acknowledged
      "DIAGNOSED",               // Physician has completed diagnosis
      "HOSPITALIZED",            // Patient admitted to hospital
      "DISCHARGED"               // Patient discharged
    ], 
    default: "TRIAGE_IN_PROGRESS" 
  },
  
  // Track status transitions for audit trail
  status_history: [
    {
      status: String,
      changed_by: mongoose.Schema.Types.ObjectId,  // User who changed status
      changed_at: { type: Date, default: Date.now },
      notes: String  // Reason for status change
    }
  ],
  severity: { type: Number, default: 1 }, // keep as numeric for UI sorting
  time_to_diagnosis_minutes: { type: Number, default: 0 },
  isWaiting: { type: Boolean, default: true },
  submittedAt: Date,
  attendedAt: Date
}, { timestamps: true });

export const Encounter = mongoose.model("Encounter", encounterSchema);