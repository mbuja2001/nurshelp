import express from "express";
import { recordVitals } from "../controllers/vitals.controller.js";

const router = express.Router();

router.post("/record", recordVitals);

export default router;