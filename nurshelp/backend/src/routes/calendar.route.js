import express from "express";
import { protect } from "../middleware/authmiddleware.js";
import { getCalendarEvents } from "../controllers/calendar.controller.js";

const router = express.Router();

router.get("/events", protect, getCalendarEvents);

export default router;
