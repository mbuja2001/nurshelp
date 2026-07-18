/**
 * Role-Based Access Control (RBAC) Middleware
 * Enforces role-based access to protected routes
 * 
 * Usage:
 *   router.post('/create', protect, authorize('MANAGER', 'ADMIN'), controllerFunc)
 */

export const authorize = (...allowedRoles) => {
  return (req, res, next) => {
    // Check if user is authenticated (protect middleware should run first)
    if (!req.user) {
      return res.status(401).json({ 
        success: false,
        message: 'Authentication required' 
      });
    }

    // Get user role from request
    const userRole = req.user.role?.toUpperCase();

    // Check if user's role is in allowed roles
    if (!allowedRoles.includes(userRole)) {
      return res.status(403).json({
        success: false,
        message: `Access denied. Required roles: ${allowedRoles.join(', ')}. Your role: ${userRole || 'UNKNOWN'}`,
        requiredRoles: allowedRoles,
        userRole: userRole
      });
    }

    // User is authorized, proceed to next middleware/controller
    next();
  };
};

/**
 * Permission-based access control
 * Checks specific permissions within a role
 * 
 * Usage:
 *   router.post('/create-shift', protect, requirePermission('canCreateShifts'), controllerFunc)
 */
export const requirePermission = (permissionKey) => {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({
        success: false,
        message: 'Authentication required'
      });
    }

    // Get permissions from user (for managers)
    const permissions = req.user.permissions || {};

    // Check if user has the permission
    if (!permissions[permissionKey] && req.user.role !== 'ADMIN') {
      return res.status(403).json({
        success: false,
        message: `Permission denied. Required permission: ${permissionKey}`,
        requiredPermission: permissionKey,
        userPermissions: permissions
      });
    }

    // User has permission, proceed
    next();
  };
};

/**
 * Department-based access control
 * Managers can only manage staff in their assigned department
 * Admins can manage all departments
 * 
 * Usage:
 *   router.get('/department-shifts', protect, restrictByDepartment(), controllerFunc)
 */
export const restrictByDepartment = () => {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({
        success: false,
        message: 'Authentication required'
      });
    }

    // Get user role (use token role if available)
    const userRole = req.user.role?.toUpperCase();

    // Admins can access all departments
    if (userRole === 'ADMIN') {
      req.departmentFilter = null; // No filter for admins
      return next();
    }

    // Managers are restricted to their department
    if (userRole === 'MANAGER') {
      const userDepartment = req.user.department;
      
      if (!userDepartment || userDepartment === 'All') {
        // Managers with 'All' department access can see everything
        req.departmentFilter = null;
      } else {
        // Restrict to specific department
        req.departmentFilter = userDepartment;
      }
      return next();
    }

    // Other roles can only see their own data
    req.departmentFilter = req.user.department || null;
    next();
  };
};

/**
 * Ownership Verification
 * Ensures users can only modify their own data
 * Admins can modify any data
 * 
 * Usage:
 *   router.put('/shift/:shiftID', protect, verifyOwnership(), controllerFunc)
 */
export const verifyOwnership = (resourceKey = 'createdBy') => {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({
        success: false,
        message: 'Authentication required'
      });
    }

    // Admins can modify anything
    if (req.user.role === 'ADMIN') {
      return next();
    }

    // For other roles, ownership verification depends on controller implementation
    // This middleware just flags that verification is needed
    req.requireOwnershipCheck = true;
    req.resourceOwner = req.user._id;
    
    next();
  };
};

/**
 * Rate limiting per role
 * Different roles have different rate limits
 * 
 * Usage:
 *   router.post('/action', protect, rateLimit({ NURSE: 100, MANAGER: 1000 }), controllerFunc)
 */
const rateLimitStore = new Map();

export const rateLimit = (limits = {}) => {
  const DEFAULT_LIMITS = {
    NURSE: 100,
    PHYSICIAN: 100,
    MANAGER: 500,
    ADMIN: 10000
  };

  const finalLimits = { ...DEFAULT_LIMITS, ...limits };

  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({
        success: false,
        message: 'Authentication required'
      });
    }

    const userRole = req.user.role?.toUpperCase() || 'UNKNOWN';
    const limit = finalLimits[userRole] || DEFAULT_LIMITS.NURSE;
    const key = `${req.user._id}_${req.path}`;
    const now = Date.now();

    if (!rateLimitStore.has(key)) {
      rateLimitStore.set(key, { count: 1, resetTime: now + 60000 });
      return next();
    }

    const record = rateLimitStore.get(key);

    if (now > record.resetTime) {
      rateLimitStore.set(key, { count: 1, resetTime: now + 60000 });
      return next();
    }

    record.count++;

    if (record.count > limit) {
      return res.status(429).json({
        success: false,
        message: `Rate limit exceeded. Max ${limit} requests per minute for ${userRole} role`,
        retryAfter: Math.ceil((record.resetTime - now) / 1000)
      });
    }

    next();
  };
};

/**
 * Action Logging for Audit Trail
 * Logs all sensitive actions for compliance
 * 
 * Usage:
 *   router.post('/action', protect, auditLog('STAFF_CREATION'), controllerFunc)
 */
export const auditLog = (actionType) => {
  return (req, res, next) => {
    if (!req.user) {
      return next();
    }

    // Store audit info in request for controller to log
    req.auditInfo = {
      action: actionType,
      userId: req.user._id,
      userEmail: req.user.email,
      userName: `${req.user.name} ${req.user.surname}`,
      role: req.user.role,
      timestamp: new Date(),
      ipAddress: req.ip,
      method: req.method,
      path: req.path,
      body: req.body
    };

    next();
  };
};
