import { Nurse } from "../models/nurse.model.js";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";

// ----------------------
// REGISTER
// ----------------------
export const registerNurse = async (req, res) => {
  try {
    console.log("REGISTER BODY:", req.body);

    const { name, surname, email, password } = req.body;

    if (!name || !surname || !email || !password) {
      return res.status(400).json({ message: "All fields required" });
    }

    const existing = await Nurse.findOne({ email });

    if (existing) {
      return res.status(400).json({ message: "Email already exists" });
    }

    const nurse = new Nurse({ name, surname, email, password });
    await nurse.save();

    res.status(201).json({
      message: "Registered successfully",
      nurse: {
        _id: nurse._id,
        name: nurse.name,
        surname: nurse.surname,
        email: nurse.email
      }
    });

  } catch (err) {
    console.error("REGISTER ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

// ----------------------
// LOGIN
// ----------------------
export const loginNurse = async (req, res) => {
  try {
    console.log("LOGIN BODY:", req.body);

    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({ message: "Email & password required" });
    }

    const nurse = await Nurse.findOne({ email });

    if (!nurse) {
      return res.status(401).json({ message: "Invalid credentials" });
    }

    const valid = await bcrypt.compare(password, nurse.password);

    if (!valid) {
      return res.status(401).json({ message: "Invalid credentials" });
    }

    const token = jwt.sign(
      {
        id: nurse._id,
        email: nurse.email,
        name: nurse.name
      },
      process.env.JWT_SECRET || "supersecretkey",
      { expiresIn: "1d" }
    );

    res.status(200).json({
      token,
      nurse: {
        _id: nurse._id,
        name: nurse.name,
        surname: nurse.surname,
        email: nurse.email
      }
    });

  } catch (err) {
    console.error("LOGIN ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};