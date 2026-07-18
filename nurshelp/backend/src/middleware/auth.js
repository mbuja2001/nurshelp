import { protect } from './authmiddleware.js';

// Export authenticateToken for use in routes
export const authenticateToken = protect;

export default authenticateToken;
