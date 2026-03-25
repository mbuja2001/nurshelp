// routes/nurse.route.js
import express from "express";
import { registerNurse, loginNurse } from "../controllers/nurse.controller.js";

const router = express.Router();

// ----------------------
// REGISTER
// ----------------------
router.post("/register", registerNurse);

// ----------------------
// LOGIN
// ✅ Fixed to match frontend
router.post("/login", loginNurse);

export default router;