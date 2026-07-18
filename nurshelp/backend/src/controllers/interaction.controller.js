import { Interaction } from "../models/interaction.model.js";

export const logInteraction = async (req, res) => {
    try {
        const interaction = await Interaction.create(req.body);
        res.status(201).json(interaction);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
};