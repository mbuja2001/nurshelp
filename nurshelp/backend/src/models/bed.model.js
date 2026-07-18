import mongoose from 'mongoose';

/**
 * Comprehensive Hospital Bed Model
 * Supports multi-tenant isolation and all ward types
 * Linked to Hospital for data retrieval at hospital level
 */
const bedSchema = new mongoose.Schema(
  {
    // ============ HOSPITAL CONTEXT ============
    hospitalId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Hospital',
      required: true,
      index: true,
    },

    // ============ BED IDENTIFICATION ============
    bed_id: {
      type: String,
      required: true,
      index: true,
      // Format: HOSP-WardCode-BedNumber (e.g., HOSP-ICU-001)
    },

    // Composite index: bed_id is unique per hospital
    // Created via: db.beds.createIndex({ hospitalId: 1, bed_id: 1 }, { unique: true })

    // ============ BED LOCATION & ASSIGNMENT ============
    ward_unit: {
      type: String,
      enum: [
        // Critical Care (6)
        'ICU', 'NICU', 'PICU', 'CICU', 'SICU', 'Neuro ICU',
        // Step-Down (2)
        'HDU', 'PCU',
        // Medical/Surgical (6)
        'Medical Ward', 'Surgical Ward', 'Orthopaedic Ward', 'Oncology Ward', 'Neurology Ward', 'Gastroenterology Ward',
        // Demographic/Specialty (5)
        'Pediatric Ward', 'Geriatric Ward', 'Maternity Ward', 'Psychiatric Ward', 'Infectious Disease Ward',
        // Support departments
        'Radiology', 'Pathology', 'Pharmacy'
      ],
      required: true,
      index: true,
    },

    sub_unit: {
      type: String,
      // Examples:
      // ICU: "General Critical Care", "Multi-Organ Support", "Trauma ICU", "Medical ICU"
      // NICU: "Level I NICU", "Level II NICU", "Level III NICU", "Isolation Nursery"
      // Radiology: "X-Ray Lab", "CT Lab", "MRI Lab", "Ultrasound Lab"
      required: false,
    },

    // ============ BED STATUS ============
    status: {
      type: String,
      enum: ['Available', 'Occupied', 'Maintenance', 'Contaminated', 'Reserved', 'Out Of Service'],
      default: 'Available',
      index: true,
    },

    // ============ CURRENT OCCUPANCY ============
    current_occupancy: {
      patientId: {
        type: String,
        index: true,
        default: null,
      },
      patientName: String,
      admittedAt: Date,
      expectedDischargeDate: Date,
      notes: String,
    },

    // ============ INFECTION CONTROL ============
    infection_control_tier: {
      type: String,
      enum: ['Standard', 'Contact', 'Droplet', 'Airborne', 'Enhanced Precautions'],
      default: 'Standard',
    },

    lastCleanedAt: Date,
    cleaningInterval: {
      type: Number, // hours
      default: 24,
    },

    // ============ EQUIPMENT BASELINE ============
    equipment_baseline: [
      {
        type: String,
        enum: [
          'High-Acuity Monitor',
          'Mechanical Ventilator',
          'Defibrillator',
          'Dialysis Machine',
          'Oxygen Delivery',
          'Infusion Pump',
          'Suction System',
          'Portable Ultrasound',
          'Cardiac Monitor',
          'Blood Gas Analyzer'
        ],
      }
    ],

    // ============ SPECIAL CHARACTERISTICS ============
    special_characteristics: [
      {
        type: String,
        enum: [
          'Negative Pressure',
          'Isolation Capsule',
          'Laminar Flow',
          'Electric Bed',
          'Manual Bed',
          'Bariatric Equipment',
          'Pediatric Setup',
          'Geriatric Bed'
        ],
      }
    ],

    // ============ COMPATIBILITY TAGS ============
    compatibility_tags: {
      age_group: {
        type: String,
        enum: ['Neonatal', 'Pediatric', 'Adult', 'Geriatric', 'All'],
        default: 'All',
      },
      acuity_level: {
        type: String,
        enum: ['Critical', 'High Acuity', 'Moderate', 'Low Acuity', 'All'],
        default: 'All',
      },
      isolation_capable: {
        type: Boolean,
        default: false,
      },
      bariatric_capable: {
        type: Boolean,
        default: false,
      },
    },

    // ============ LOGISTICS METRICS ============
    logistics_metrics: {
      occupancy_rate: {
        type: Number,
        min: 0,
        max: 100,
        default: 0,
      },
      average_length_of_stay_hours: {
        type: Number,
        default: 0,
      },
      turnaround_time_minutes: {
        type: Number,
        default: 0,
      },
      maintenance_incidents_30_days: {
        type: Number,
        default: 0,
      },
    },

    // ============ AUDIT & METADATA ============
    createdAt: {
      type: Date,
      default: Date.now,
    },
    updatedAt: {
      type: Date,
      default: Date.now,
    },
    createdBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Manager',
    },
    lastModifiedBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Manager',
    },

    // ============ HISTORY & AUDIT TRAIL ============
    occupancy_history: [
      {
        patientId: String,
        admittedAt: Date,
        dischargedAt: Date,
        length_of_stay_hours: Number,
      }
    ],

    status_changes: [
      {
        from_status: String,
        to_status: String,
        changedAt: Date,
        changedBy: mongoose.Schema.Types.ObjectId,
        reason: String,
      }
    ],
  },
  {
    timestamps: true,
    collection: 'beds',
  }
);

// ============ INDEXES ============
// Multi-tenant queries
bedSchema.index({ hospitalId: 1, bed_id: 1 }, { unique: true });
bedSchema.index({ hospitalId: 1, ward_unit: 1 });
bedSchema.index({ hospitalId: 1, status: 1 });
bedSchema.index({ hospitalId: 1, 'current_occupancy.patientId': 1 });
bedSchema.index({ hospitalId: 1, sub_unit: 1 });
bedSchema.index({ hospitalId: 1, 'infection_control_tier': 1 });

// Single bed lookups (within hospital context)
bedSchema.index({ bed_id: 1 });
bedSchema.index({ ward_unit: 1 });
bedSchema.index({ status: 1 });

export const Bed = mongoose.model('Bed', bedSchema);

export default Bed;
