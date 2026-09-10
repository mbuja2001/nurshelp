import jwt from "jsonwebtoken";
import { ClinicalStaff } from "../models/clinicalstaff.model.js";
import { Manager } from "../models/manager.model.js";

export const protect = async (req, res, next) => {
  try {
    // Log ALL incoming requests to diagnose routing issues
    console.log('\n[AUTH-MIDDLEWARE] Incoming request:', {
      method: req.method,
      path: req.path,
      fullUrl: req.originalUrl,
      bodyKeys: Object.keys(req.body || {}),
      headers: {
        'content-type': req.headers['content-type'],
        'authorization': req.headers.authorization ? 'present' : 'missing'
      }
    });

    let token = null;

    if (req.headers.authorization?.startsWith("Bearer")) {
      token = req.headers.authorization.split(" ")[1];
    }

    if (!token) {
      console.log("[AUTH] No token provided");
      return res.status(401).json({ message: "Not authorized" });
    }

    console.log("[AUTH] Token verified:", token.substring(0, 20) + "...");
    
    const decoded = jwt.verify(
      token,
      process.env.JWT_SECRET || "supersecretkey"
    );

    console.log("[AUTH] Token decoded successfully");
    console.log("  ID:", decoded.id);
    console.log("  Role:", decoded.role);
    console.log("  Email:", decoded.email);
    console.log("  HospitalID:", decoded.hospitalId);
    console.log("  HospitalName:", decoded.hospitalName);

    // Try to find as Nurse first, then as Manager
    console.log("[AUTH] Searching ClinicalStaff collection for ID:", decoded.id);
    let user = await ClinicalStaff.findById(decoded.id).select("-password");
    console.log("[AUTH] ClinicalStaff lookup result:", user ? `Found: ${user.name}` : "Not found");
    
    if (!user) {
      console.log("[AUTH] Searching Manager collection for ID:", decoded.id);
      user = await Manager.findById(decoded.id).select("-password");
      console.log("[AUTH] Manager lookup result:", user ? `Found: ${user.name}` : "Not found");
    }

    if (!user) {
      console.error("[AUTH] User not found in either collection");
      return res.status(401).json({ message: "User not found" });
    }

    console.log("[AUTH] User authenticated successfully:", user.email);
    
    req.user = user;
    req.user.role = decoded.role;  // Ensure role from token is available
    req.user.hospitalId = decoded.hospitalId;
    req.userRole = decoded.role;
    req.hospitalId = decoded.hospitalId;
    next();
  } catch (err) {
    console.error("[AUTH] ERROR:", err.message);
    res.status(401).json({ message: "Invalid token" });
  }
};