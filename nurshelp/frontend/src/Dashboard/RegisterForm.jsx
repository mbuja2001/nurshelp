import React, { useState } from "react";
import { styles } from "./styles";

const RegisterForm = ({ onRegister, onCancel }) => {
  const [step, setStep] = useState(1); // 1: Hospital, 2: Staff, 3: Beds, 4: Pharmaceuticals, 5: Departments, 6: Review
  const [form, setForm] = useState({
    // Hospital & Admin
    hospitalName: "",
    managerName: "",
    managerSurname: "",
    managerEmail: "",
    managerPassword: "",
    // Collections
    staffMembers: [],
    beds: [],
    pharmaceuticals: [],
    departments: ["ED", "ICU", "OR"]
  });

  // ====== HOSPITAL & ADMIN (STEP 1) ======
  const [newStaff, setNewStaff] = useState({
    name: "",
    surname: "",
    email: "",
    password: "",
    role: "",
    department: "ED",
    specialty: ""
  });

  // ====== BEDS (STEP 3) ======
  const [newBed, setNewBed] = useState({
    bed_id: "",
    ward_unit: "",
    sub_unit: "",
    status: "Available",
    infection_control_tier: "Standard",
    equipment_baseline: [],
    special_characteristics: [],
    compatibility_tags: {
      age_group: "All",
      gender_restriction: "None",
      max_weight_kg: 250
    }
  });
  const [selectedEquipment, setSelectedEquipment] = useState([]);
  const [selectedCharacteristics, setSelectedCharacteristics] = useState([]);

  // ====== PHARMACEUTICALS (STEP 4) ======
  const [newPharmaceutical, setNewPharmaceutical] = useState({
    name: "",
    genericName: "",
    strength: "",
    form: "Tablet",
    route: "PO",
    currentStock: 0,
    reorderPoint: 10,
    unitPrice: 0
  });

  // ====== OPTIONS & CONFIGURATIONS ======
  const wardOptions = [
    // Critical Care Wards
    "ICU",
    "NICU",
    "PICU",
    "CICU",
    "SICU",
    "Neuro ICU",
    // Step-Down & Progressive Care
    "HDU",
    "PCU",
    // Medical and Surgical Wards
    "Medical Ward",
    "Surgical Ward",
    "Orthopaedic Ward",
    "Oncology Ward",
    "Neurology Ward",
    "Gastroenterology Ward",
    // Demographic & Specialty Wards
    "Pediatric Ward",
    "Geriatric Ward",
    "Maternity Ward",
    "Psychiatric Ward",
    "Infectious Disease Ward"
  ];
  
  const subUnitOptions = {
    "ICU": ["General Critical Care", "Multi-Organ Support", "Trauma ICU", "Medical ICU"],
    "NICU": ["Level I NICU", "Level II NICU", "Level III NICU", "Isolation Nursery"],
    "PICU": ["General PICU", "Cardiac PICU", "Surgical PICU", "Neuro PICU"],
    "CICU": ["Acute Coronary", "Arrhythmia Monitoring", "Post-Intervention Recovery", "Shock Unit"],
    "SICU": ["Post-Transplant", "Major Surgery Recovery", "Trauma Surgery", "Vascular Surgery"],
    "Neuro ICU": ["Stroke Care", "Neurosurgery Recovery", "Spinal Injury", "Brain Trauma"],
    "HDU": ["Cardiac Monitoring", "Post-Acute Care", "Weaning", "Observation"],
    "PCU": ["Ventilator Weaning", "Cardiac Recovery", "Step-Down Care", "Stabilization"],
    "Medical Ward": ["Respiratory Infections", "Sepsis Management", "Chronic Disease Flare", "Internal Medicine"],
    "Surgical Ward": ["General Surgery Recovery", "Moderate Complexity Post-Op", "Drain Management", "Wound Care"],
    "Orthopaedic Ward": ["Fracture Care", "Joint Replacement Recovery", "Spinal Surgery", "Orthopedic Trauma"],
    "Oncology Ward": ["Chemotherapy", "Radiation Support", "Symptom Management", "Palliative Care"],
    "Neurology Ward": ["Stroke Recovery", "Epilepsy Monitoring", "Neurodegenerative Diseases", "Seizure Management"],
    "Gastroenterology Ward": ["Severe GI Bleeding", "Liver Failure", "Digestive Tract Disease", "Post-Endoscopy"],
    "Pediatric Ward": ["General Pediatrics", "Post-Surgical Peds", "Pediatric Acute Care", "Child Health"],
    "Geriatric Ward": ["Acute Geriatric Care", "Delirium Management", "Mobility Support", "Fall Prevention"],
    "Maternity Ward": ["Labor & Delivery", "Postpartum Recovery", "Newborn Care", "High-Risk Pregnancy"],
    "Psychiatric Ward": ["Acute Mental Health", "Behavioral Crisis", "Suicide Prevention", "Psychiatric Observation"],
    "Infectious Disease Ward": ["Airborne Isolation", "Contact Isolation", "Biocontainment", "Disease-Specific Care"]
  };
  
  // CLINICAL ROLES & SPECIALTIES
  const roleTypes = [
    "Nurse",
    "Physician",
    "Radiology Technician",
    "Radiology Technician Lead",
    "Pathology Technician",
    "Pathology Technician Lead",
    "Pharmacist",
    "Pharmacy Technician",
    "Pharmacy Technician Lead",
    "Department Head",
    "Department Assistant",
    "Other"
  ];
  
  const clinicalSpecialties = {
    "Nurse": [
      "General Nursing",
      "Emergency Nursing",
      "Critical Care Nursing",
      "ICU Nursing",
      "Operating Room Nursing",
      "Cardiac Nursing",
      "Pediatric Nursing",
      "Oncology Nursing"
    ],
    "Physician": [
      "General Practice",
      "General Medicine",
      "Internal Medicine",
      "Emergency Medicine",
      "Cardiology",
      "Pulmonology",
      "Gastroenterology",
      "Neurology",
      "Nephrology",
      "Rheumatology",
      "Infectious Disease",
      "Endocrinology",
      "Geriatrics",
      "Radiology",
      "Interventional Radiology",
      "Oncology",
      "Pathology",
      "Toxicology",
      "Dermatology",
      "Optometrist",
      "Critical Care",
      "Obstetrics/Gynecology",
      "Pediatrics",
      "Psychiatry",
      "General Surgery",
      "Orthopedics",
      "Urology",
      "ENT",
      "Cardiothoracic Surgery",
      "Plastic Surgery",
      "Pediatric Surgery",
      "Neurosurgery",
      "Thoracic Surgery",
      "Vascular Surgery",
      "Trauma Surgery",
      "Anesthesiology"
    ],
    "Radiology Technician": [
      "X-Ray Technician",
      "CT Technician",
      "MRI Technician",
      "Ultrasound Technician",
      "Nuclear Medicine Technician",
      "Interventional Radiology Tech"
    ],
    "Radiology Technician Lead": [
      "Radiology Lead",
      "Imaging Department Lead",
      "CT Lab Lead",
      "MRI Lab Lead",
      "Ultrasound Lead"
    ],
    "Pathology Technician": [
      "Lab Technician",
      "Histology Technician",
      "Hematology Technician",
      "Chemistry Lab Tech",
      "Microbiology Technician",
      "Blood Bank Technician"
    ],
    "Pathology Technician Lead": [
      "Pathology Lab Lead",
      "Hematology Lab Lead",
      "Chemistry Lab Lead",
      "Microbiology Lab Lead",
      "Blood Bank Supervisor"
    ],
    "Pharmacist": [
      "Clinical Pharmacist",
      "Hospital Pharmacist",
      "Pharmacy Manager",
      "Ambulatory Pharmacist"
    ],
    "Pharmacy Technician": [
      "Pharmacy Tech",
      "IV Compounding Tech",
      "Sterile Compounding Tech",
      "Pharmacy Inventory Tech"
    ],
    "Pharmacy Technician Lead": [
      "Pharmacy Lead Tech",
      "Compounding Lab Lead",
      "Pharmacy Operations Lead"
    ],
    "Department Head": [
      "ED Department Head",
      "ICU Department Head",
      "OR Department Head",
      "Medical-Surgical Head",
      "Pediatrics Head",
      "OB/GYN Head",
      "Psychiatry Head",
      "Radiology Department Head",
      "Pathology Department Head",
      "Pharmacy Department Head"
    ],
    "Department Assistant": [
      "ED Assistant",
      "ICU Assistant",
      "OR Assistant",
      "Medical-Surgical Assistant",
      "Pediatrics Assistant",
      "Administrative Assistant",
      "Department Coordinator"
    ],
    "Other": [
      "General",
      "Administrative",
      "Support Staff",
      "Nursing"
    ]
  };
  
  const equipmentOptions = ["High-Acuity Monitor", "Mechanical Ventilator", "Defibrillator", "Dialysis Machine", "Oxygen Delivery", "Infusion Pump", "Suction System", "Portable Ultrasound", "Cardiac Monitor", "Blood Gas Analyzer"];
  const characteristicOptions = ["Negative Pressure", "Isolation Capsule", "Laminar Flow", "Electric Bed", "Manual Bed", "Bariatric Equipment", "Pediatric Setup", "Geriatric Bed"];
  const infectionControlTiers = ["Standard", "Contact Precautions", "Droplet Precautions", "Airborne Precautions", "Contact + Droplet"];
  const ageGroups = ["All", "Pediatric", "Adult", "Geriatric"];
  const drugForms = ["Tablet", "Injection", "Infusion", "Liquid", "Inhaler", "Cream", "Patch"];
  const drugRoutes = ["PO", "IV", "IM", "SC", "PR", "Inhalation", "Topical"];

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));
  };

  const handleStaffChange = (e) => {
    const { name, value } = e.target;
    
    if (name === "role") {
      // Auto-assign department based on role
      let autoDepartment = "Medical Ward"; // default
      
      if (["Radiology Technician", "Radiology Technician Lead"].includes(value)) {
        autoDepartment = "Radiology";
      } else if (["Pathology Technician", "Pathology Technician Lead"].includes(value)) {
        autoDepartment = "Pathology";
      } else if (["Pharmacist", "Pharmacy Technician", "Pharmacy Technician Lead"].includes(value)) {
        autoDepartment = "Pharmacy";
      } else if (value === "Physician") {
        autoDepartment = "CICU"; // physicians default to CICU
      } else if (value === "Nurse") {
        autoDepartment = "ICU"; // nurses default to ICU
      }
      
      setNewStaff(prev => ({ ...prev, [name]: value, department: autoDepartment }));
    } else {
      setNewStaff(prev => ({ ...prev, [name]: value }));
    }
  };

  const handleBedChange = (e) => {
    const { name, value } = e.target;
    if (name.includes(".")) {
      const [parent, child] = name.split(".");
      setNewBed(prev => ({
        ...prev,
        [parent]: { ...prev[parent], [child]: isNaN(value) ? value : parseFloat(value) }
      }));
    } else {
      setNewBed(prev => ({ ...prev, [name]: value }));
    }
  };

  const handlePharmaceuticalChange = (e) => {
    const { name, value, type } = e.target;
    setNewPharmaceutical(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) : value
    }));
  };

  const handleEquipmentToggle = (equipment) => {
    setSelectedEquipment(prev =>
      prev.includes(equipment)
        ? prev.filter(e => e !== equipment)
        : [...prev, equipment]
    );
  };

  const handleCharacteristicToggle = (characteristic) => {
    setSelectedCharacteristics(prev =>
      prev.includes(characteristic)
        ? prev.filter(c => c !== characteristic)
        : [...prev, characteristic]
    );
  };

  // ====== ADD STAFF ======
  const handleAddStaff = (e) => {
    e.preventDefault();
    if (!newStaff.name || !newStaff.surname || !newStaff.email || !newStaff.password || !newStaff.role) {
      alert("Please fill out all staff fields");
      return;
    }
    setForm(prev => ({
      ...prev,
      staffMembers: [...prev.staffMembers, { ...newStaff }]
    }));
    setNewStaff({ name: "", surname: "", email: "", password: "", role: "", department: "ED", specialty: "" });
  };

  const handleRemoveStaff = (index) => {
    setForm(prev => ({
      ...prev,
      staffMembers: prev.staffMembers.filter((_, i) => i !== index)
    }));
  };

  // ====== ADD BED ======
  const handleAddBed = (e) => {
    e.preventDefault();
    if (!newBed.bed_id || !newBed.ward_unit) {
      alert("Please fill bed ID and ward");
      return;
    }
    setForm(prev => ({
      ...prev,
      beds: [...prev.beds, { 
        ...newBed, 
        equipment_baseline: selectedEquipment,
        special_characteristics: selectedCharacteristics
      }]
    }));
    setNewBed({
      bed_id: "",
      ward_unit: "",
      sub_unit: "",
      status: "Available",
      infection_control_tier: "Standard",
      equipment_baseline: [],
      special_characteristics: [],
      compatibility_tags: {
        age_group: "All",
        gender_restriction: "None",
        max_weight_kg: 250
      }
    });
    setSelectedEquipment([]);
    setSelectedCharacteristics([]);
  };

  const handleRemoveBed = (index) => {
    setForm(prev => ({
      ...prev,
      beds: prev.beds.filter((_, i) => i !== index)
    }));
  };

  // ====== ADD PHARMACEUTICAL ======
  const handleAddPharmaceutical = (e) => {
    e.preventDefault();
    if (!newPharmaceutical.name || !newPharmaceutical.strength) {
      alert("Please fill drug name and strength");
      return;
    }
    setForm(prev => ({
      ...prev,
      pharmaceuticals: [...prev.pharmaceuticals, { ...newPharmaceutical }]
    }));
    setNewPharmaceutical({ name: "", genericName: "", strength: "", form: "Tablet", route: "PO", currentStock: 0, reorderPoint: 10, unitPrice: 0 });
  };

  const handleRemovePharmaceutical = (index) => {
    setForm(prev => ({
      ...prev,
      pharmaceuticals: prev.pharmaceuticals.filter((_, i) => i !== index)
    }));
  };

  // ====== SUBMIT ======
  const handleSubmit = (e) => {
    e.preventDefault();
    if (!form.hospitalName || !form.managerName || !form.managerSurname || !form.managerEmail || !form.managerPassword) {
      alert("Please fill out all hospital admin fields");
      return;
    }
    onRegister({
      hospitalName: form.hospitalName,
      managerName: form.managerName,
      managerSurname: form.managerSurname,
      managerEmail: form.managerEmail,
      managerPassword: form.managerPassword,
      isHospitalAdmin: true,
      staffMembers: form.staffMembers,
      beds: form.beds,
      pharmaceuticals: form.pharmaceuticals,
      departments: form.departments
    });
  };

  // ============ STEP 1: HOSPITAL & ADMIN ============
  if (step === 1) {
    return (
      <div style={styles.patientBg}>
        <div style={styles.loginCard}>
          <div style={{ marginBottom: 20 }}>
            <div style={{ fontSize: "12px", color: "#64748b", marginBottom: 8 }}>STEP 1 OF 6</div>
            <h2 style={{ margin: 0, fontSize: "24px" }}>Hospital Registration</h2>
            <p style={{ margin: "8px 0 0 0", fontSize: "13px", color: "#64748b" }}>Create your hospital and admin account</p>
          </div>

          <form onSubmit={(e) => { e.preventDefault(); setStep(2); }}>
            <h3 style={{ fontSize: "14px", color: "#ffffff", marginTop: "20px", marginBottom: "10px" }}>Hospital Information</h3>
            <input
              name="hospitalName"
              placeholder="Hospital Name"
              value={form.hospitalName}
              onChange={handleChange}
              style={styles.loginInput}
              required
            />

            <h3 style={{ fontSize: "14px", color: "#ffffff", marginTop: "20px", marginBottom: "10px" }}>Admin Manager Account</h3>
            <div style={{ display: "flex", gap: "10px" }}>
              <input
                name="managerName"
                placeholder="First Name"
                value={form.managerName}
                onChange={handleChange}
                style={{ ...styles.loginInput, flex: 1 }}
                required
              />
              <input
                name="managerSurname"
                placeholder="Last Name"
                value={form.managerSurname}
                onChange={handleChange}
                style={{ ...styles.loginInput, flex: 1 }}
                required
              />
            </div>

            <input
              name="managerEmail"
              type="email"
              placeholder="Admin Email"
              value={form.managerEmail}
              onChange={handleChange}
              style={{ ...styles.loginInput, marginTop: "10px" }}
              required
            />

            <input
              name="managerPassword"
              type="password"
              placeholder="Admin Password"
              value={form.managerPassword}
              onChange={handleChange}
              style={{ ...styles.loginInput, marginTop: "10px" }}
              required
            />

            <div style={{ display: "flex", gap: "10px", marginTop: "20px", justifyContent: "flex-end" }}>
              <button type="button" onClick={onCancel} style={styles.btnOutline}>
                Cancel
              </button>
              <button type="submit" style={styles.btnWhite}>
                Next: Add Staff →
              </button>
            </div>
          </form>
        </div>
      </div>
    );
  }

  // ============ STEP 2: STAFF MEMBERS ============
  if (step === 2) {
    return (
      <div style={styles.patientBg}>
        <div style={{ ...styles.loginCard, maxHeight: "85vh", overflowY: "auto" }}>
          <div style={{ marginBottom: 20 }}>
            <div style={{ fontSize: "12px", color: "#64748b", marginBottom: 8 }}>STEP 2 OF 6</div>
            <h2 style={{ margin: 0, fontSize: "24px" }}>Register Staff</h2>
            <p style={{ margin: "8px 0 0 0", fontSize: "13px", color: "#64748b" }}>Add clinical staff members (optional)</p>
          </div>

          <form onSubmit={handleAddStaff}>
            <h3 style={{ fontSize: "14px", color: "#ffffff", marginBottom: "10px" }}>Add Staff Member</h3>

            <div style={{ display: "flex", gap: "10px" }}>
              <input
                name="name"
                placeholder="First Name"
                value={newStaff.name}
                onChange={handleStaffChange}
                style={{ ...styles.loginInput, flex: 1 }}
              />
              <input
                name="surname"
                placeholder="Last Name"
                value={newStaff.surname}
                onChange={handleStaffChange}
                style={{ ...styles.loginInput, flex: 1 }}
              />
            </div>

            <input
              name="email"
              type="email"
              placeholder="Staff Email"
              value={newStaff.email}
              onChange={handleStaffChange}
              style={{ ...styles.loginInput, marginTop: "10px" }}
            />

            <input
              name="password"
              type="password"
              placeholder="Staff Password"
              value={newStaff.password}
              onChange={handleStaffChange}
              style={{ ...styles.loginInput, marginTop: "10px" }}
            />

            <div style={{ display: "flex", gap: "10px", marginTop: "10px" }}>
              <select
                name="role"
                value={newStaff.role}
                onChange={handleStaffChange}
                style={{ ...styles.loginInput, flex: 1 }}
              >
                <option value="" disabled>Select Role</option>
                {roleTypes.map(role => (
                  <option key={role} value={role}>{role}</option>
                ))}
              </select>

              {newStaff.role && (
                <select
                  name="specialty"
                  value={newStaff.specialty}
                  onChange={handleStaffChange}
                  style={{ ...styles.loginInput, flex: 1 }}
                >
                  <option value="">Select Specialty</option>
                  {clinicalSpecialties[newStaff.role]?.map(specialty => (
                    <option key={specialty} value={specialty}>{specialty}</option>
                  ))}
                </select>
              )}
            </div>

            <select
              name="department"
              value={newStaff.department}
              onChange={handleStaffChange}
              style={{ ...styles.loginInput, marginTop: "10px" }}
              disabled={["Radiology Technician", "Radiology Technician Lead", "Pathology Technician", "Pathology Technician Lead", "Pharmacist", "Pharmacy Technician", "Pharmacy Technician Lead"].includes(newStaff.role)}
            >
              <option value="">Select Department (Ward Assignment)</option>
              {newStaff.role && ["Radiology Technician", "Radiology Technician Lead"].includes(newStaff.role) && (
                <option value="Radiology">Radiology Lab (Auto-Assigned)</option>
              )}
              {newStaff.role && ["Pathology Technician", "Pathology Technician Lead"].includes(newStaff.role) && (
                <option value="Pathology">Pathology Lab (Auto-Assigned)</option>
              )}
              {newStaff.role && ["Pharmacist", "Pharmacy Technician", "Pharmacy Technician Lead"].includes(newStaff.role) && (
                <option value="Pharmacy">Pharmacy (Auto-Assigned)</option>
              )}
              {!["Radiology Technician", "Radiology Technician Lead", "Pathology Technician", "Pathology Technician Lead", "Pharmacist", "Pharmacy Technician", "Pharmacy Technician Lead"].includes(newStaff.role) && (
                wardOptions.map(ward => (
                  <option key={ward} value={ward}>{ward}</option>
                ))
              )}
            </select>

            <button type="submit" style={{ ...styles.btnWhite, width: "100%", marginTop: "10px" }}>
              Add Staff Member
            </button>
          </form>

          {form.staffMembers.length > 0 && (
            <>
              <h3 style={{ fontSize: "14px", color: "#ffffff", marginTop: "20px", marginBottom: "10px" }}>
                Added Staff ({form.staffMembers.length})
              </h3>
              <div style={{ maxHeight: "200px", overflowY: "auto", border: "1px solid #475569", borderRadius: "4px", padding: "10px", backgroundColor: "#0f172a" }}>
                {form.staffMembers.map((staff, idx) => (
                  <div key={idx} style={{ padding: "10px", marginBottom: "8px", borderBottom: "1px solid #334155", display: "flex", justifyContent: "space-between", alignItems: "center", backgroundColor: "#1e293b", borderRadius: "4px" }}>
                    <div>
                      <div style={{ color: "#ffffff", fontWeight: "600" }}>{staff.name} {staff.surname}</div>
                      <div style={{ color: "#cbd5e1", fontSize: "12px", marginTop: "2px" }}>
                        {staff.role} {staff.specialty && `• ${staff.specialty}`} {staff.department && `• ${staff.department}`}
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => handleRemoveStaff(idx)}
                      style={{ padding: "4px 8px", fontSize: "12px", color: "#ef4444", backgroundColor: "transparent", border: "1px solid #fecaca", borderRadius: "4px", cursor: "pointer" }}
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            </>
          )}

          <div style={{ display: "flex", gap: "10px", marginTop: "20px", justifyContent: "flex-end" }}>
            <button type="button" onClick={() => setStep(1)} style={styles.btnOutline}>
              Back
            </button>
            <button type="button" onClick={() => setStep(3)} style={styles.btnWhite}>
              Next: Register Beds →
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ============ STEP 3: BED REGISTRATION ============
  if (step === 3) {
    return (
      <div style={styles.patientBg}>
        <div style={{ ...styles.loginCard, maxHeight: "85vh", overflowY: "auto" }}>
          <div style={{ marginBottom: 20 }}>
            <div style={{ fontSize: "12px", color: "#64748b", marginBottom: 8 }}>STEP 3 OF 6</div>
            <h2 style={{ margin: 0, fontSize: "24px" }}>Register Beds</h2>
            <p style={{ margin: "8px 0 0 0", fontSize: "13px", color: "#64748b" }}>Add hospital beds and ward configurations</p>
          </div>

          <form onSubmit={handleAddBed}>
            <h3 style={{ fontSize: "14px", color: "#ffffff", marginBottom: "10px" }}>Add Bed</h3>

            <div style={{ display: "flex", gap: "10px" }}>
              <input
                name="bed_id"
                placeholder="Bed ID (e.g., ER-RESUS-01)"
                value={newBed.bed_id}
                onChange={handleBedChange}
                style={{ ...styles.loginInput, flex: 1 }}
              />
              <select
                name="ward_unit"
                value={newBed.ward_unit}
                onChange={handleBedChange}
                style={{ ...styles.loginInput, flex: 1 }}
              >
                <option value="">Select Ward</option>
                {wardOptions.map(ward => (
                  <option key={ward} value={ward}>{ward}</option>
                ))}
              </select>
            </div>

            {newBed.ward_unit && (
              <select
                name="sub_unit"
                value={newBed.sub_unit}
                onChange={handleBedChange}
                style={{ ...styles.loginInput, marginTop: "10px" }}
              >
                <option value="">Select Sub-Unit</option>
                {subUnitOptions[newBed.ward_unit]?.map(sub => (
                  <option key={sub} value={sub}>{sub}</option>
                ))}
              </select>
            )}

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", marginTop: "10px" }}>
              <select
                name="status"
                value={newBed.status}
                onChange={handleBedChange}
                style={styles.loginInput}
              >
                <option value="Available">Available</option>
                <option value="Occupied">Occupied</option>
                <option value="Cleaning">Cleaning</option>
                <option value="Maintenance">Maintenance</option>
              </select>

              <select
                name="infection_control_tier"
                value={newBed.infection_control_tier}
                onChange={handleBedChange}
                style={styles.loginInput}
              >
                {infectionControlTiers.map(tier => (
                  <option key={tier} value={tier}>{tier}</option>
                ))}
              </select>
            </div>

            <h4 style={{ fontSize: "13px", color: "#cbd5e1", marginTop: "15px", marginBottom: "10px" }}>Equipment Baseline:</h4>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
              {equipmentOptions.map(equip => (
                <label key={equip} style={{ display: "flex", alignItems: "center", gap: "8px", cursor: "pointer", color: "#cbd5e1" }}>
                  <input
                    type="checkbox"
                    checked={selectedEquipment.includes(equip)}
                    onChange={() => handleEquipmentToggle(equip)}
                    style={{ cursor: "pointer" }}
                  />
                  <span style={{ fontSize: "12px" }}>{equip}</span>
                </label>
              ))}
            </div>

            <h4 style={{ fontSize: "13px", color: "#cbd5e1", marginTop: "15px", marginBottom: "10px" }}>Special Characteristics:</h4>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
              {characteristicOptions.map(char => (
                <label key={char} style={{ display: "flex", alignItems: "center", gap: "8px", cursor: "pointer", color: "#cbd5e1" }}>
                  <input
                    type="checkbox"
                    checked={selectedCharacteristics.includes(char)}
                    onChange={() => handleCharacteristicToggle(char)}
                    style={{ cursor: "pointer" }}
                  />
                  <span style={{ fontSize: "12px" }}>{char}</span>
                </label>
              ))}
            </div>

            <h4 style={{ fontSize: "13px", color: "#cbd5e1", marginTop: "15px", marginBottom: "10px" }}>Compatibility Tags:</h4>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
              <select
                name="compatibility_tags.age_group"
                value={newBed.compatibility_tags.age_group}
                onChange={handleBedChange}
                style={styles.loginInput}
              >
                {ageGroups.map(age => (
                  <option key={age} value={age}>{age}</option>
                ))}
              </select>

              <select
                name="compatibility_tags.gender_restriction"
                value={newBed.compatibility_tags.gender_restriction}
                onChange={handleBedChange}
                style={styles.loginInput}
              >
                <option value="None">No Restriction</option>
                <option value="M">Male Only</option>
                <option value="F">Female Only</option>
              </select>
            </div>

            <input
              name="compatibility_tags.max_weight_kg"
              type="number"
              placeholder="Max Weight (kg)"
              value={newBed.compatibility_tags.max_weight_kg}
              onChange={handleBedChange}
              style={{ ...styles.loginInput, marginTop: "10px" }}
            />

            <button type="submit" style={{ ...styles.btnWhite, width: "100%", marginTop: "15px" }}>
              Add Bed
            </button>
          </form>

          {form.beds.length > 0 && (
            <>
              <h3 style={{ fontSize: "14px", color: "#ffffff", marginTop: "20px", marginBottom: "10px" }}>
                Registered Beds ({form.beds.length})
              </h3>
              <div style={{ maxHeight: "250px", overflowY: "auto", border: "1px solid #475569", borderRadius: "4px", padding: "10px", backgroundColor: "#0f172a" }}>
                {form.beds.map((bed, idx) => (
                  <div key={idx} style={{ padding: "10px", marginBottom: "8px", borderBottom: "1px solid #334155", display: "flex", justifyContent: "space-between", alignItems: "flex-start", backgroundColor: "#1e293b", borderRadius: "4px" }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ color: "#ffffff", fontWeight: "600" }}>{bed.bed_id}</div>
                      <div style={{ color: "#cbd5e1", fontSize: "11px", marginTop: "3px" }}>
                        {bed.ward_unit} {bed.sub_unit && `• ${bed.sub_unit}`}
                      </div>
                      <div style={{ color: "#94a3b8", fontSize: "11px", marginTop: "2px" }}>
                        {bed.status} • {bed.infection_control_tier}
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => handleRemoveBed(idx)}
                      style={{ padding: "4px 8px", fontSize: "12px", color: "#ef4444", backgroundColor: "transparent", border: "1px solid #fecaca", borderRadius: "4px", cursor: "pointer", whiteSpace: "nowrap" }}
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            </>
          )}

          <div style={{ display: "flex", gap: "10px", marginTop: "20px", justifyContent: "flex-end" }}>
            <button type="button" onClick={() => setStep(2)} style={styles.btnOutline}>
              Back
            </button>
            <button type="button" onClick={() => setStep(4)} style={styles.btnWhite}>
              Next: Add Pharmaceuticals →
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ============ STEP 4: PHARMACEUTICAL REGISTRATION ============
  if (step === 4) {
    return (
      <div style={styles.patientBg}>
        <div style={{ ...styles.loginCard, maxHeight: "85vh", overflowY: "auto" }}>
          <div style={{ marginBottom: 20 }}>
            <div style={{ fontSize: "12px", color: "#64748b", marginBottom: 8 }}>STEP 4 OF 6</div>
            <h2 style={{ margin: 0, fontSize: "24px" }}>Register Pharmaceuticals</h2>
            <p style={{ margin: "8px 0 0 0", fontSize: "13px", color: "#64748b" }}>Add drugs to pharmacy inventory</p>
          </div>

          <form onSubmit={handleAddPharmaceutical}>
            <h3 style={{ fontSize: "14px", color: "#ffffff", marginBottom: "10px" }}>Add Drug</h3>

            <input
              name="name"
              placeholder="Drug Name (Brand)"
              value={newPharmaceutical.name}
              onChange={handlePharmaceuticalChange}
              style={styles.loginInput}
            />

            <input
              name="genericName"
              placeholder="Generic Name (optional)"
              value={newPharmaceutical.genericName}
              onChange={handlePharmaceuticalChange}
              style={{ ...styles.loginInput, marginTop: "10px" }}
            />

            <input
              name="strength"
              placeholder="Strength/Dosage (e.g., 500mg)"
              value={newPharmaceutical.strength}
              onChange={handlePharmaceuticalChange}
              style={{ ...styles.loginInput, marginTop: "10px" }}
            />

            <div style={{ display: "flex", gap: "10px", marginTop: "10px" }}>
              <select
                name="form"
                value={newPharmaceutical.form}
                onChange={handlePharmaceuticalChange}
                style={{ ...styles.loginInput, flex: 1 }}
              >
                {drugForms.map(f => (
                  <option key={f} value={f}>{f}</option>
                ))}
              </select>

              <select
                name="route"
                value={newPharmaceutical.route}
                onChange={handlePharmaceuticalChange}
                style={{ ...styles.loginInput, flex: 1 }}
              >
                {drugRoutes.map(r => (
                  <option key={r} value={r}>{r}</option>
                ))}
              </select>
            </div>

            <div style={{ display: "flex", gap: "10px", marginTop: "10px" }}>
              <input
                name="currentStock"
                type="number"
                placeholder="Current Stock"
                value={newPharmaceutical.currentStock}
                onChange={handlePharmaceuticalChange}
                style={{ ...styles.loginInput, flex: 1 }}
              />
              <input
                name="reorderPoint"
                type="number"
                placeholder="Reorder Point"
                value={newPharmaceutical.reorderPoint}
                onChange={handlePharmaceuticalChange}
                style={{ ...styles.loginInput, flex: 1 }}
              />
            </div>

            <input
              name="unitPrice"
              type="number"
              placeholder="Unit Price (ZAR)"
              step="0.01"
              value={newPharmaceutical.unitPrice}
              onChange={handlePharmaceuticalChange}
              style={{ ...styles.loginInput, marginTop: "10px" }}
            />

            <button type="submit" style={{ ...styles.btnWhite, width: "100%", marginTop: "15px" }}>
              Add Pharmaceutical
            </button>
          </form>

          {form.pharmaceuticals.length > 0 && (
            <>
              <h3 style={{ fontSize: "14px", color: "#ffffff", marginTop: "20px", marginBottom: "10px" }}>
                Added Pharmaceuticals ({form.pharmaceuticals.length})
              </h3>
              <div style={{ maxHeight: "200px", overflowY: "auto", border: "1px solid #475569", borderRadius: "4px", padding: "10px", backgroundColor: "#0f172a" }}>
                {form.pharmaceuticals.map((drug, idx) => (
                  <div key={idx} style={{ padding: "10px", marginBottom: "8px", borderBottom: "1px solid #334155", display: "flex", justifyContent: "space-between", alignItems: "center", backgroundColor: "#1e293b", borderRadius: "4px" }}>
                    <div>
                      <div style={{ color: "#ffffff", fontWeight: "600" }}>{drug.name}</div>
                      <div style={{ color: "#cbd5e1", fontSize: "12px", marginTop: "2px" }}>
                        {drug.strength} • {drug.form} • {drug.route}
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => handleRemovePharmaceutical(idx)}
                      style={{ padding: "4px 8px", fontSize: "12px", color: "#ef4444", backgroundColor: "transparent", border: "1px solid #fecaca", borderRadius: "4px", cursor: "pointer" }}
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            </>
          )}

          <div style={{ display: "flex", gap: "10px", marginTop: "20px", justifyContent: "flex-end" }}>
            <button type="button" onClick={() => setStep(3)} style={styles.btnOutline}>
              Back
            </button>
            <button type="button" onClick={() => setStep(5)} style={styles.btnWhite}>
              Next: Configure Departments →
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ============ STEP 5: DEPARTMENTS & RESOURCES ============
  if (step === 5) {
    return (
      <div style={styles.patientBg}>
        <div style={{ ...styles.loginCard, maxHeight: "85vh", overflowY: "auto" }}>
          <div style={{ marginBottom: 20 }}>
            <div style={{ fontSize: "12px", color: "#64748b", marginBottom: 8 }}>STEP 5 OF 6</div>
            <h2 style={{ margin: 0, fontSize: "24px" }}>Configure Departments</h2>
            <p style={{ margin: "8px 0 0 0", fontSize: "13px", color: "#64748b" }}>Enable clinical departments and resources</p>
          </div>

          <div style={{ padding: "15px", backgroundColor: "#334155", borderRadius: "8px", marginBottom: "15px", border: "1px solid #475569" }}>
            <h3 style={{ fontSize: "14px", color: "#ffffff", marginBottom: "12px" }}>Available Departments</h3>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
              {wardOptions.map(dept => (
                <label key={dept} style={{ display: "flex", alignItems: "center", gap: "8px", cursor: "pointer", color: "#cbd5e1" }}>
                  <input
                    type="checkbox"
                    checked={form.departments.includes(dept)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setForm(prev => ({ ...prev, departments: [...prev.departments, dept] }));
                      } else {
                        setForm(prev => ({ ...prev, departments: prev.departments.filter(d => d !== dept) }));
                      }
                    }}
                    style={{ cursor: "pointer" }}
                  />
                  {dept}
                </label>
              ))}
            </div>
          </div>

          <div style={{ padding: "15px", backgroundColor: "#0c4a6e", borderRadius: "8px", marginBottom: "15px", border: "1px solid #0284c7", color: "#bfdbfe", fontSize: "13px", lineHeight: "1.6" }}>
            <div style={{ marginBottom: "10px", fontWeight: "600" }}>📋 Clinical Features Enabled:</div>
            <div>✓ AI Clinical Decision Support (ESI Triage)</div>
            <div>✓ Clinical Handover & Nursing Notes</div>
            <div>✓ Patient Intake & Vital Tracking</div>
            <div>✓ Physician Dashboard & Reports</div>
            <div style={{ marginTop: "10px", fontWeight: "600" }}>🏥 Resource Management Enabled:</div>
            <div>✓ Bed Management & Tracking</div>
            <div>✓ Pharmacy & Medication Inventory</div>
            <div>✓ Radiology Request Management</div>
            <div>✓ Pathology Lab Orders</div>
          </div>

          <div style={{ display: "flex", gap: "10px", marginTop: "20px", justifyContent: "flex-end" }}>
            <button type="button" onClick={() => setStep(4)} style={styles.btnOutline}>
              Back
            </button>
            <button type="button" onClick={() => setStep(6)} style={styles.btnWhite}>
              Next: Review & Create →
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ============ STEP 6: REVIEW & CREATE ============
  if (step === 6) {
    return (
      <div style={styles.patientBg}>
        <div style={{ ...styles.loginCard, maxHeight: "85vh", overflowY: "auto" }}>
          <div style={{ marginBottom: 20 }}>
            <div style={{ fontSize: "12px", color: "#64748b", marginBottom: 8 }}>STEP 6 OF 6</div>
            <h2 style={{ margin: 0, fontSize: "24px" }}>Review & Create</h2>
            <p style={{ margin: "8px 0 0 0", fontSize: "13px", color: "#64748b" }}>Review all information before creating hospital</p>
          </div>

          <div style={{ padding: "15px", backgroundColor: "#334155", borderRadius: "8px", marginBottom: "15px", fontSize: "13px", border: "1px solid #475569" }}>
            <h4 style={{ margin: "0 0 12px 0", color: "#38bdf8", fontSize: "14px", fontWeight: "600", textTransform: "uppercase", letterSpacing: "0.5px" }}>Hospital Details</h4>
            <div style={{ marginBottom: "10px" }}>
              <div style={{ color: "#cbd5e1", fontSize: "12px", textTransform: "uppercase", letterSpacing: "0.5px" }}>Hospital Name</div>
              <div style={{ fontWeight: "600", color: "#ffffff", fontSize: "15px" }}>{form.hospitalName || "—"}</div>
            </div>

            <h4 style={{ margin: "15px 0 12px 0", color: "#38bdf8", fontSize: "14px", fontWeight: "600", textTransform: "uppercase", letterSpacing: "0.5px" }}>Admin Account</h4>
            <div style={{ marginBottom: "10px" }}>
              <div style={{ color: "#cbd5e1", fontSize: "12px", textTransform: "uppercase", letterSpacing: "0.5px" }}>Name</div>
              <div style={{ fontWeight: "600", color: "#ffffff", fontSize: "15px" }}>{form.managerName} {form.managerSurname}</div>
            </div>

            {form.staffMembers.length > 0 && (
              <>
                <h4 style={{ margin: "15px 0 12px 0", color: "#60a5fa", fontSize: "13px", fontWeight: "600" }}>👥 Staff Members ({form.staffMembers.length})</h4>
                {form.staffMembers.slice(0, 3).map((staff, idx) => (
                  <div key={idx} style={{ marginBottom: "8px", paddingBottom: "8px", borderBottom: "1px solid #475569" }}>
                    <div style={{ fontWeight: "600", color: "#ffffff", fontSize: "13px" }}>{staff.name} {staff.surname}</div>
                    <div style={{ color: "#cbd5e1", fontSize: "12px", marginTop: "2px" }}>{staff.role} {staff.specialty && `• ${staff.specialty}`} • {staff.department}</div>
                  </div>
                ))}
                {form.staffMembers.length > 3 && (
                  <div style={{ color: "#94a3b8", fontSize: "12px" }}>+{form.staffMembers.length - 3} more...</div>
                )}
              </>
            )}

            {form.beds.length > 0 && (
              <>
                <h4 style={{ margin: "15px 0 12px 0", color: "#60a5fa", fontSize: "13px", fontWeight: "600" }}>🛏️  Beds Registered ({form.beds.length})</h4>
                {form.beds.slice(0, 3).map((bed, idx) => (
                  <div key={idx} style={{ marginBottom: "8px", paddingBottom: "8px", borderBottom: "1px solid #475569" }}>
                    <div style={{ fontWeight: "600", color: "#ffffff", fontSize: "13px" }}>{bed.bed_id}</div>
                    <div style={{ color: "#cbd5e1", fontSize: "12px", marginTop: "2px" }}>{bed.ward_unit} {bed.sub_unit && `• ${bed.sub_unit}`} • {bed.status}</div>
                  </div>
                ))}
                {form.beds.length > 3 && (
                  <div style={{ color: "#94a3b8", fontSize: "12px" }}>+{form.beds.length - 3} more...</div>
                )}
              </>
            )}

            {form.pharmaceuticals.length > 0 && (
              <>
                <h4 style={{ margin: "15px 0 12px 0", color: "#60a5fa", fontSize: "13px", fontWeight: "600" }}>💊 Pharmaceuticals ({form.pharmaceuticals.length})</h4>
                {form.pharmaceuticals.slice(0, 3).map((drug, idx) => (
                  <div key={idx} style={{ marginBottom: "8px", paddingBottom: "8px", borderBottom: "1px solid #475569" }}>
                    <div style={{ fontWeight: "600", color: "#ffffff", fontSize: "13px" }}>{drug.name}</div>
                    <div style={{ color: "#cbd5e1", fontSize: "12px", marginTop: "2px" }}>{drug.strength} • {drug.form}</div>
                  </div>
                ))}
                {form.pharmaceuticals.length > 3 && (
                  <div style={{ color: "#94a3b8", fontSize: "12px" }}>+{form.pharmaceuticals.length - 3} more...</div>
                )}
              </>
            )}
          </div>

          <div style={{ padding: "15px", backgroundColor: "#0c4a6e", borderRadius: "8px", marginBottom: "15px", fontSize: "13px", lineHeight: "1.6", color: "#bfdbfe", border: "1px solid #0284c7" }}>
            <strong style={{ color: "#60a5fa" }}>✓</strong> Hospital created with multi-tenant isolation <br />
            <strong style={{ color: "#60a5fa" }}>✓</strong> Admin account ready to login <br />
            <strong style={{ color: "#60a5fa" }}>✓</strong> {form.staffMembers.length} staff members will be created <br />
            <strong style={{ color: "#60a5fa" }}>✓</strong> {form.beds.length} beds registered in system <br />
            <strong style={{ color: "#60a5fa" }}>✓</strong> {form.pharmaceuticals.length} pharmaceuticals added to pharmacy <br />
            <strong style={{ color: "#60a5fa" }}>✓</strong> Clinical decision support ready <br />
            <strong style={{ color: "#60a5fa" }}>✓</strong> Resource management enabled
          </div>

          <form onSubmit={handleSubmit}>
            <div style={{ display: "flex", gap: "10px", marginTop: "20px", justifyContent: "flex-end" }}>
              <button type="button" onClick={() => setStep(5)} style={styles.btnOutline}>
                Back
              </button>
              <button type="button" onClick={onCancel} style={{ ...styles.btnOutline, color: "#64748b" }}>
                Cancel
              </button>
              <button type="submit" style={styles.btnWhite}>
                Create Hospital ✓
              </button>
            </div>
          </form>
        </div>
      </div>
    );
  }
};

export default RegisterForm;
