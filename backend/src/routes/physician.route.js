import express from "express";
import { getRecommendations } from "../controllers/physician.controller.js";

const router = express.Router();

// GET /api/physicians/recommendations?encounterId=<id>
router.get("/recommendations", getRecommendations);

export default router;
