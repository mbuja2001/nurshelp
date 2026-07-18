import express from "express";
import { 
  getRecommendations, 
  getPhysicianCases, 
  getWardBeds, 
  submitDisposition 
} from "../controllers/physician.controller.js";
import { authenticateToken } from "../middleware/auth.middleware.js";

const router = express.Router();

// Existing endpoint
router.get("/recommendations", getRecommendations);

// NEW: Physician Dashboard Endpoints
// GET /api/physician/cases?context=ER - Fetch 3-lane view (active, observation, escalations)
router.get("/cases", authenticateToken, getPhysicianCases);

// GET /api/ward/beds?context=ER - Fetch bed occupancy
router.get("/beds", authenticateToken, getWardBeds);

// POST /api/disposition - Submit physician decision
router.post("/disposition", authenticateToken, submitDisposition);

export default router;
