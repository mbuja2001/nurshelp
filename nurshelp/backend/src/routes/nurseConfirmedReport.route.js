import express from "express";
import { protect } from "../middleware/authmiddleware.js";
import {
  createNurseConfirmedReport,
  updateNurseConfirmedReport,
  getReportByEncounter
} from "../controllers/nurseConfirmedReport.controller.js";

const router = express.Router();

// Create a nurse-confirmed report
router.post("/", protect, createNurseConfirmedReport);

// Update a nurse-confirmed report
router.patch("/:id", protect, updateNurseConfirmedReport);

// Get latest nurse-confirmed report for an encounter
router.get("/encounter/:encounterId", protect, getReportByEncounter);

export default router;
