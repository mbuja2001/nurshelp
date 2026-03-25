import mongoose, { Schema } from "mongoose";
import bcrypt from "bcrypt";

const nurseSchema = new Schema({
  name: { type: String, required: true },
  surname: { type: String, required: true },
  email: { type: String, required: true, unique: true, lowercase: true, trim: true },
  password: { type: String, required: true },
}, { timestamps: true });

// hash password before save
nurseSchema.pre("save", async function () {
  if (this.isModified("password")) {
    this.password = await bcrypt.hash(this.password, 10);
  }
});

export const Nurse = mongoose.model("Nurse", nurseSchema);