import { Vital } from "../models/vitals.model.js";

export const recordVitals = async (req, res) => {
    try {
        const vitals = await Vital.create(req.body);
        res.status(201).json(vitals);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
};