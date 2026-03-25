import express from "express";
import {
  startEncounter,
  getEncounterById,
  confirmEncounter,
  attendEncounter,
  assignPhysician,
  getWaitingEncounters,
  runTriage
} from "../controllers/encounter.controller.js";
import { protect } from "../middleware/authmiddleware.js";

const router = express.Router();

// Public create triage endpoint is via /api/triage/run (handled in triage router)
// But we allow manual creation via protected route:
router.post("/", protect, startEncounter);

// Run triage (from frontend) — public (because it might be called before login)
// we intentionally make triage route public in /api/triage (see triage.route.js)
// Keep here for completeness if you want to restrict the endpoint.
// router.post("/triage", protect, runTriage);

// Waiting queue (must come BEFORE /:id so '/waiting' isn't treated as an id)
router.get("/waiting", protect, getWaitingEncounters);

// Get all encounters for authenticated nurse
router.get("/", protect, async (req, res) => {
  try {
    const nurseId = req.user?._id;
    if (!nurseId) return res.status(401).json({ message: "Not authorized" });

    const encounters = await import("../models/encounter.model.js").then(mod => mod.Encounter.find({ nurse_id: nurseId }).sort({ createdAt: -1 }));
    return res.status(200).json(encounters);
  } catch (err) {
    console.error("List encounters error:", err);
    return res.status(500).json({ message: "Failed to list encounters" });
  }
});

// Confirm patient report (only the assigned nurse may confirm)
router.patch("/:id/confirm", protect, confirmEncounter);

// Mark encounter as attended (physician)
router.put("/:id/attend", protect, attendEncounter);

// Assign physician / ward / bed
router.patch("/:id/assign", protect, assignPhysician);

// Get single encounter by id (must belong to nurse)
router.get("/:id", protect, getEncounterById);

export default router;