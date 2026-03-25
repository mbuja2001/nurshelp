import express from "express";
import { runTriage } from "../controllers/encounter.controller.js";

const router = express.Router();

// Frontend calls: POST /api/triage/run
router.post("/run", runTriage);

export default router;