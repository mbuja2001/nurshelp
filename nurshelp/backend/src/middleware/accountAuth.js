import jwt from 'jsonwebtoken';
import { SandboxAccount } from '../models/sandboxAccount.model.js';

export const protectSandboxAccount = async (req, res, next) => {
  try {
    const token = req.headers.authorization?.startsWith('Bearer ')
      ? req.headers.authorization.slice(7)
      : null;
    if (!token) return res.status(401).json({ success: false, message: 'Account authentication required' });

    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'supersecretkey');
    if (decoded.identityType !== 'ACCOUNT' || !decoded.id) {
      return res.status(403).json({ success: false, message: 'Sandbox account access required' });
    }

    const account = await SandboxAccount.findById(decoded.id).select('_id name email isActive');
    if (!account || !account.isActive) {
      return res.status(401).json({ success: false, message: 'Account is unavailable' });
    }

    req.sandboxAccount = account;
    return next();
  } catch (error) {
    return res.status(401).json({ success: false, message: 'Invalid account token' });
  }
};