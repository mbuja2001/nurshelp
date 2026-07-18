import express from "express";
import dotenv from "dotenv";
import mongoose from "mongoose";
import cors from "cors";

// ============================================
// SERVICE IMPORTS
// ============================================
import { initializeScheduledJobs } from "./services/scheduledJobs.js";

// ============================================
// ROUTE IMPORTS
// ============================================

// Authentication (Unified Login)
import authRouter from "./routes/auth.route.js";

// Phase 1: Hospital Registration
import hospitalRouter from "./routes/hospital.route.js";

// Phase 2-3: Manager Dashboard (Staff & Shift Management)
import managerRouter from "./routes/manager.route.js";

// Phase 4: Staff Clock In/Out
import staffRouter from "./routes/staff.route.js"; // router handles staff (nurse/physician)

// Full Day Simulation - Dashboard Configuration & Role-Based Rendering
import dashboardRouter from "./routes/dashboard.route.js";

// Additional routes
import encounterRouter from "./routes/encounter.route.js";
import triageRouter from "./routes/triage.route.js";
import physicianRouter from "./routes/physician.route.js";
import reportRouter from "./routes/nurseConfirmedReport.route.js";
import bedRouter from "./routes/bed.route.js";
import pharmacyRouter from "./routes/pharmacy.route.js";
import specialistRouter from "./routes/specialist.route.js";
import shiftRouter from "./routes/shift.route.js";
import seedRouter from "./routes/seed.route.js";
import clinicalRouter from "./routes/clinical.routes.js";
import labRouter from "./routes/lab.routes.js"; // Lab request workflow

// EHR System - New Routes
import notificationRouter from "./routes/notification.routes.js";
import patientRouter from "./routes/patient.routes.js";

dotenv.config();

const app = express();

// ============================================
// API FUNCTION BREAKDOWN - ALL CAPABILITIES
// ============================================
/**
 * HOSPITAL TRIAGE & CLINICAL WORKFLOW SYSTEM
 * Multi-tenant hospital platform with role-based dashboards
 *
 * ====== AUTHENTICATION ======
 * POST   /api/auth/login
 *        Unified login for all user types (managers, clinical staff)
 *        Returns JWT token with role for RBAC
 *
 * ====== HOSPITAL SETUP (PHASE 1) ======
 * POST   /api/hospital/register
 *        Register new hospital with admin credentials
 * POST   /api/hospital/login
 *        Admin hospital login (legacy - use /api/auth/login)
 *
 * ====== STAFF MANAGEMENT (PHASE 2) ======
 * POST   /api/manager/staff/register
 *        Add new clinical staff (nurse/physician) to hospital
 * GET    /api/manager/staff/list
 *        View all staff in hospital
 * PUT    /api/manager/staff/:staffID
 *        Update staff details (department, role, etc.)
 * DELETE /api/manager/staff/:staffID
 *        Deactivate staff member
 * GET    /api/manager/staff-status
 *        Dashboard overview of staff status
 * GET    /api/manager/pending-approvals
 *        View pending shift/schedule approvals
 * POST   /api/manager/bulk-approvals
 *        Process multiple approval requests
 *
 * ====== SHIFT SCHEDULING (PHASE 3) ======
 * POST   /api/shifts/create
 *        Define new shift (time block, department, roles needed)
 * POST   /api/shifts/:shiftId/assign-staff
 *        Assign staff to shift roles (TRIAGE_NURSE, CLINICAL_NURSE, PHYSICIAN)
 * GET    /api/shifts
 *        View all shifts
 * GET    /api/shifts/:shiftId
 *        View specific shift details
 *
 * ====== STAFF CLOCK IN/OUT (PHASE 4) ======
 * POST   /api/staff/clock-in
 *        Staff clocks in for their shift
 * POST   /api/staff/clock-out
 *        Staff clocks out at end of shift
 * POST   /api/staff/break-start
 *        Mark break start
 * POST   /api/staff/break-end
 *        Mark break end
 * GET    /api/staff/available-shifts
 *        View available shifts for staff member
 * POST   /api/staff/approve-late-arrival
 *        Manager approval for late clock-in
 * POST   /api/staff/approve-break
 *        Manager approval for unscheduled break
 *
 * ====== DASHBOARD & ROLE-BASED RENDERING ======
 * GET    /api/dashboard/config
 *        Get role-specific dashboard configuration for authenticated staff
 *        Decodes JWT → finds today's shift → checks team assignment
 *        Returns: TRIAGE_NURSE | CLINICAL_NURSE | PHYSICIAN config
 * GET    /api/dashboard/permissions
 *        Get role permissions reference for frontend
 * GET    /api/dashboard/queue
 *        Get patient queue filtered by staff's assigned context
 *
 * ====== PATIENT TRIAGE & ENCOUNTERS ======
 * POST   /api/triage/hospital-request
 *        AI triage with clinical handover (frontend sends hospitalId, patientId, triageNurseId)
 *        Returns physician + clinical nurse assignments from real-time available staff
 * POST   /api/triage/run
 *        Legacy manual triage endpoint
 * GET    /api/triage/staff
 *        Get available staff for hospital
 * PATCH  /api/triage/:id/confirm
 *        Confirm triage assessment
 * PATCH  /api/triage/:id/specialist-feedback
 *        Add specialist feedback to triage
 * GET    /api/encounters
 *        Get all patient encounters
 * GET    /api/encounters/waiting
 *        Get waiting patients (not yet assigned)
 * POST   /api/encounters/:id/mark-attended
 *        Mark patient as attended
 *
 * ====== PHYSICIAN ENDPOINTS ======
 * GET    /api/physicians
 *        Get physician dashboard/queue
 * GET    /api/physicians/:id
 *        Get physician details
 *
 * ====== REPORTING ======
 * GET    /api/reports/nurse-confirmed
 *        Get nurse-confirmed patient reports
 * POST   /api/reports/nurse-confirmed
 *        Submit nurse-confirmed report
 *
 * ====== BED MANAGEMENT ======
 * GET    /api/beds
 *        Get available beds
 * POST   /api/beds
 *        Create new bed
 * PUT    /api/beds/:bedId
 *        Update bed details
 *
 * ====== DEVELOPMENT & TESTING ======
 * GET    /debug/routes
 *        List all registered routes (development only)
 * GET    /seed/*
 *        Database seeding endpoints for testing
 *
 */

// ============================================
// MIDDLEWARE - JSON PARSING (MUST BE FIRST)
// ============================================

app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

// ============================================
// CORS
// ============================================

app.use(cors({
  origin: ["http://localhost:5173", "http://localhost:5174"],
  credentials: true,
}));

// ============================================
// MONGODB CONNECTION
// ============================================

const MONGO_URI = process.env.MONGO_URI;

if (!MONGO_URI) {
  console.error("❌ MONGO_URI missing");
  process.exit(1);
}

mongoose.connect(MONGO_URI)
  .then(() => console.log("✅ MongoDB connected"))
  .catch(err => {
    console.error("❌ MongoDB error:", err);
    process.exit(1);
  });

// ============================================
// AUTHENTICATION ROUTES
// ============================================

/**
 * Unified Authentication Endpoint
 * POST /api/auth/login - Login for all user types (managers, staff)
 * Checks both Manager and ClinicalStaff collections
 * Returns JWT token with role for RBAC
 */
app.use("/api/auth", authRouter);

// ============================================
// HOSPITAL WORKFLOW ROUTES (PHASE-BASED)
// ============================================

/**
 * PHASE 1: Hospital Registration & Admin Authentication
 * POST /api/hospital/register - Register new hospital with admin
 * POST /api/hospital/login - Admin login (deprecated, use /api/auth/login)
 */
app.use("/api/hospital", hospitalRouter);

/**
 * PHASE 2-3: Manager Dashboard - Staff Management & Shift Scheduling
 * Protected by RBAC middleware (authorize, restrictByDepartment, auditLog)
 * 
 * Staff Management:
 * POST /api/manager/staff/register - Add new staff (nurse/physician)
 * GET /api/manager/staff/list - View all staff
 * PUT /api/manager/staff/:staffID - Update staff
 * DELETE /api/manager/staff/:staffID - Deactivate staff
 * 
 * Shift Management & Approvals:
 * GET /api/manager/staff-status - Dashboard overview
 * GET /api/manager/pending-approvals - View pending approvals
 * POST /api/manager/bulk-approvals - Process approvals
 */
app.use("/api/manager", managerRouter);

/**
 * PHASE 4: Staff Clock In/Out - Staff (clinical staff members)
 * Public endpoints: /login, /register
 * Protected endpoints: /clock-in, /clock-out, /break-start, /break-end, /available-shifts
 * Manager approval: /approve-late-arrival, /approve-break
 */
// Mount staff routes under /api/staff (refactored from nurse)
app.use("/api/staff", staffRouter);

// ============================================
// DASHBOARD ROUTES - FULL DAY SIMULATION
// ============================================

/**
 * Dashboard Configuration & Role-Based Rendering
 * GET /api/dashboard/config - Get role-based dashboard config for authenticated staff
 *   Decodes JWT token → finds today's shift → checks teamAssignment → returns role-specific config
 *   Returns 403 if staff not assigned to any role in today's shift
 * GET /api/dashboard/permissions - Get role permissions reference
 * GET /api/dashboard/queue - Get patient queue for staff's assigned context
 */
app.use("/api/dashboard", dashboardRouter);

// ============================================
// ADDITIONAL ROUTES
// ============================================

app.use("/api/encounters", encounterRouter);
app.use("/api/triage", triageRouter);
app.use("/api/physicians", physicianRouter);  // Unified physician endpoint
app.use("/api/specialist", specialistRouter);  // Specialist queues and results
app.use("/api/reports/nurse-confirmed", reportRouter);
app.use("/api/beds", bedRouter);
app.use("/api/pharmacy", pharmacyRouter);
app.use("/api/lab-requests", labRouter);  // Lab request workflow (Radiology, Pathology)
app.use("/api/shifts", shiftRouter);
app.use("/api/seed", seedRouter);  // Seed data endpoint
app.use("/api/clinical", clinicalRouter);

// EHR System - Patient Lookup & Notifications
app.use("/api/patients", patientRouter);  // Patient search and EHR retrieval
app.use("/api/notifications", notificationRouter);  // Handover notifications

// ============================================
// HEALTH CHECK ENDPOINT
// ============================================

app.get("/health", (req, res) => {
  res.json({ ok: true });
});

// ============================================
// DEBUG ENDPOINT (Development Only)
// ============================================

if (process.env.NODE_ENV !== 'production') {
  app.get('/debug/routes', (req, res) => {
    try {
      const routes = [];
      app._router.stack.forEach((middleware) => {
        if (middleware.route) {
          // Direct route
          const methods = Object.keys(middleware.route.methods).join(',').toUpperCase();
          routes.push({ path: middleware.route.path, methods });
        } else if (middleware.name === 'router' && middleware.handle && middleware.handle.stack) {
          // Router with sub-routes
          middleware.handle.stack.forEach((handler) => {
            if (handler.route) {
              const methods = Object.keys(handler.route.methods).join(',').toUpperCase();
              routes.push({ path: handler.route.path, methods });
            }
          });
        }
      });
      res.json({ success: true, count: routes.length, routes });
    } catch (err) {
      res.status(500).json({ success: false, message: err.message });
    }
  });
}

// ============================================
// GLOBAL ERROR HANDLER
// ============================================

app.use((err, req, res, next) => {
  console.error("🔥 GLOBAL ERROR:", err);
  res.status(500).json({ 
    success: false,
    message: err.message || "Server error" 
  });
});

// ============================================
// START SERVER & SCHEDULED JOBS
// ============================================

const PORT = process.env.PORT || 5001;

app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
  console.log(`\n📋 Hospital Workflow Phases:`);
  console.log(`  Phase 1: POST /api/hospital/register (hospital setup)`);
  console.log(`  Phase 1: POST /api/hospital/login (admin login)`);
  console.log(`  Phase 2: POST /api/manager/staff/register (add staff)`);
  console.log(`  Phase 3: POST /api/shifts/create (define shifts)`);
  console.log(`  Phase 4: POST /api/staff/login (staff login)`);
  console.log(`  Phase 4: POST /api/staff/clock-in (staff clock in)`);
  
  // Initialize background scheduled jobs
  initializeScheduledJobs();
});