import { protect } from './authmiddleware.js';

// Alias for backward-compatible import paths used across routes
export const authenticateToken = protect;

export default authenticateToken;
