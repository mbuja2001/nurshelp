import express from "express";
import { logInteraction } from "../controllers/interaction.controller.js";

const router = express.Router();

router.post("/log", logInteraction);

export default router;