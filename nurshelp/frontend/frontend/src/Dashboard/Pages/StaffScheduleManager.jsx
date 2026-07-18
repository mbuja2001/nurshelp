import React, { useState, useEffect } from 'react';
// Staff view for non-admins
import ClockInOutBoard from './ClockInOutBoard';
import { FiX, FiPlus, FiSearch, FiFilter, FiClock, FiUsers } from 'react-icons/fi';
import { decodeJwt } from '../../utils/jwt';
import SidebarLayout from '../SidebarLayout';

const StaffScheduleManager = () => {
  const token = localStorage.getItem('token');
  const decoded = decodeJwt(token);
  const userRole = decoded?.role || 'NURSE';
  const isAdmin = userRole === 'ADMIN' || userRole === 'MANAGER';
  
  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

  // ============ STATE ============
  const [activeTab, setActiveTab] = useState('schedule'); // 'schedule', 'search', 'add-shift'
  const [targetDate, setTargetDate] = useState(new Date().toISOString().split('T')[0]);
  
  // Staff search state
  const [staffList, setStaffList] = useState([]);
  const [filteredStaff, setFilteredStaff] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [departmentFilter, setDepartmentFilter] = useState('all');
  const [roleFilter, setRoleFilter] = useState('all');
  const [searchLoading, setSearchLoading] = useState(false);

  // Schedule view state
  const [todaySchedule, setTodaySchedule] = useState([]);
  const [scheduleLoading, setScheduleLoading] = useState(false);

  // Add shift form state (admin only)
  const [shiftForm, setShiftForm] = useState({
    shiftName: '',
    department: 'ED',
    shiftStart: '08:00',
    shiftEnd: '16:00',
    shiftDate: new Date().toISOString().split('T')[0], // Today's date YYYY-MM-DD
    maxStaffRequired: 1,
    assignedStaff: []
  });
  
  // Role assignments (integrated into shift creation)
  const [roleAssignments, setRoleAssignments] = useState({
    triageNurseId: '',
    clinicalNurseId: '',
    clinicalNurseContext: 'CLINICAL_NURSE_ER',
    physicianId: '',
    physicianContext: 'PHYSICIAN_ER'
  });
  
  const [selectedStaffForShift, setSelectedStaffForShift] = useState([]);
  const [shiftMessages, setShiftMessages] = useState({ success: '', error: '' });
  
  // Per-staff role assignment mapping: { [staffId]: { role: 'TRIAGE_NURSE'|'CLINICAL_NURSE'|'PHYSICIAN', context: 'CLINICAL_NURSE_ER'|... } }
  const [selectedStaffRoles, setSelectedStaffRoles] = useState({});
  
  // Staff assignment search/filter state
  const [staffSearchQuery, setStaffSearchQuery] = useState('');
  const [staffDepartmentFilter, setStaffDepartmentFilter] = useState('all');
  const [staffRoleFilter, setStaffRoleFilter] = useState('all');

  // Messages
  const [messages, setMessages] = useState({ success: '', error: '' });

  // ============ DEPARTMENTS ============
  const DEPARTMENTS = ['ED', 'ICU', 'OR', 'OB', 'Medical-Surgical', 'Pediatrics', 'Psychiatry'];
  const ROLES = ['NURSE', 'PHYSICIAN', 'SPECIALIST'];

  // ============ FETCH STAFF ============
  const fetchStaff = async () => {
    if (!token) {
      setMessages({ success: '', error: 'Not authenticated' });
      return;
    }
    
    setSearchLoading(true);
    try {
      let url = `${BACKEND_URL}/api/manager/staff/list?`;
      if (departmentFilter !== 'all') url += `department=${departmentFilter}&`;
      if (roleFilter !== 'all') url += `role=${roleFilter}`;

      const res = await fetch(url, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }
      
      const data = await res.json();
      
      if (data.success) {
        let staff = data.staff || [];
        
        // Filter by search query
        if (searchQuery) {
          staff = staff.filter(s => 
            (s.firstName || s.name)?.toLowerCase().includes(searchQuery.toLowerCase()) ||
            (s.lastName || s.surname)?.toLowerCase().includes(searchQuery.toLowerCase()) ||
            s.email?.toLowerCase().includes(searchQuery.toLowerCase())
          );
        }
        
        setFilteredStaff(staff);
      } else {
        setMessages({ success: '', error: 'Failed to load staff' });
      }
    } catch (err) {
      setMessages({ success: '', error: err.message });
    } finally {
      setSearchLoading(false);
    }
  };

  // If user is not admin/manager, render staff Clock In/Out view within the same route
  const [staffActiveTab, setStaffActiveTab] = useState('clock'); // 'clock' | 'myshifts'

  if (!isAdmin) {
    return (
      <SidebarLayout username={decoded?.name || 'Staff'}>
        <div style={{ maxWidth: 1200 }}>
          {/* Header (styled like Settings) */}
          <div style={{ marginBottom: 32 }}>
            <h1 style={{ margin: '0 0 8px 0', fontSize: 28, fontWeight: 700, color: '#0f172a' }}>
              <FiClock size={32} style={{ marginRight: 12, verticalAlign: 'middle', color: '#0369a1' }} />
              Staff Dashboard
            </h1>
            <p style={{ margin: 0, fontSize: 14, color: '#6b7280' }}>Clock in/out, view assigned shifts, and manage your schedule</p>
          </div>

          {/* Minimal Tab Navigation to match Settings styling */}
          <div style={{ display: 'flex', gap: 12, marginBottom: 24, borderBottom: '2px solid #e5e7eb', overflowX: 'auto' }}>
            {[
              { id: 'clock', label: 'Clock' },
              { id: 'myshifts', label: 'My Shifts' },
              { id: 'history', label: 'Attendance History' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setStaffActiveTab(tab.id)}
                style={{
                  padding: '12px 16px',
                  backgroundColor: 'transparent',
                  border: 'none',
                  borderBottom: staffActiveTab === tab.id ? '3px solid #0f172a' : 'none',
                  cursor: 'pointer',
                  fontSize: 14,
                  fontWeight: staffActiveTab === tab.id ? 700 : 500,
                  color: staffActiveTab === tab.id ? '#0f172a' : '#6b7280',
                  transition: 'all 0.2s',
                  whiteSpace: 'nowrap'
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Card container matching Settings content */}
          <div style={{
            backgroundColor: 'white',
            borderRadius: 12,
            padding: 24,
            boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
            border: '1px solid #e5e7eb',
          }}>
            <ClockInOutBoard
              token={token}
              user={decoded}
              activeTab={staffActiveTab}
              onLogout={() => { localStorage.removeItem('token'); window.location.href = '/'; }}
            />
          </div>
        </div>
      </SidebarLayout>
    );
  }

  // ============ FETCH TODAY'S SCHEDULE ============
  const fetchTodaySchedule = async () => {
    if (!token) {
      setMessages({ success: '', error: 'Not authenticated' });
      return;
    }
    
    setScheduleLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/shifts/day?date=${targetDate}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }
      
      const data = await res.json();
      
      if (data.success) {
        // Format shifts for 24-hour view
        setTodaySchedule(data.shifts || []);
      } else {
        setMessages({ success: '', error: 'Failed to load schedule' });
      }
    } catch (err) {
      setMessages({ success: '', error: err.message });
    } finally {
      setScheduleLoading(false);
    }
  };

  // ============ CREATE SHIFT (Admin only) ============
  // ============ CREATE SHIFT & ASSIGN ROLES ============
  const handleCreateShift = async (e) => {
    e.preventDefault();
    
    console.log('[FRONTEND] CREATE SHIFT CALLED');
    console.log('  Shift Form:', shiftForm);
    console.log('  Selected Staff IDs:', selectedStaffForShift);
    console.log('  Role Assignments:', roleAssignments);
    
    if (!shiftForm.shiftName || !shiftForm.department) {
      console.log('  ❌ VALIDATION FAILED: Missing required fields');
      setShiftMessages({ success: '', error: 'Please fill all required fields' });
      return;
    }

    // VALIDATION: If staff are selected, they must have a role assigned
    if (selectedStaffForShift.length > 0) {
      const unassignedStaff = selectedStaffForShift.filter(staffId => 
        !selectedStaffRoles[staffId] || !selectedStaffRoles[staffId].role
      );
      if (unassignedStaff.length > 0) {
        console.log('  ❌ VALIDATION FAILED: Some staff members do not have roles assigned');
        const unassignedNames = unassignedStaff.map(id => {
          const staff = filteredStaff.find(s => s._id === id);
          return staff?.firstName || staff?.name || id;
        }).join(', ');
        setShiftMessages({ 
          success: '', 
          error: `Please assign a role for: ${unassignedNames}. Select from the role dropdown.` 
        });
        return;
      }
    }

    try {
      // Convert frontend role names to backend enum values
      const convertRoleToBackendType = (frontendRole) => {
        const roleMap = {
          'TRIAGE_NURSE': 'NURSE',
          'CLINICAL_NURSE': 'NURSE',
          'PHYSICIAN': 'PHYSICIAN',
          'SPECIALIST': 'SPECIALIST',
          'SUPPORT': 'SUPPORT',
          'MANAGER': 'MANAGER'
        };
        return roleMap[frontendRole] || 'NURSE';
      };

      // Map selected staff IDs to staff objects with full details
      const assignedStaffData = selectedStaffForShift.map(staffId => {
        const staff = filteredStaff.find(s => s._id === staffId);
        // Use the selected role from the dropdown, or fall back to staff's system role, then to 'NURSE'
        const assignedRole = selectedStaffRoles[staffId]?.role || (staff?.systemRole || staff?.role) || 'NURSE';
        // Convert to backend enum value
        const backendType = convertRoleToBackendType(assignedRole);
        return {
          staffID: staffId,
          type: backendType,
          name: (staff?.firstName || staff?.name) + ' ' + (staff?.lastName || staff?.surname) || '',
          surname: staff?.lastName || staff?.surname || '',
          specialty: staff?.specialty || '',
          status: 'ASSIGNED'
        };
      });

      console.log('  ✓ Staff Data:', assignedStaffData);

      const shiftData = {
        shiftName: shiftForm.shiftName,
        department: shiftForm.department,
        shiftStart: shiftForm.shiftStart,
        shiftEnd: shiftForm.shiftEnd,
        date: shiftForm.shiftDate, // Include the date!
        maxStaffRequired: parseInt(shiftForm.maxStaffRequired),
        assignedStaff: assignedStaffData
      };

      console.log('  📤 Sending to backend:', JSON.stringify(shiftData, null, 2));
      console.log(`  URL: ${BACKEND_URL}/api/shifts/create`);
      console.log(`  Token: ${token?.substring(0, 20)}...`);

      const res = await fetch(`${BACKEND_URL}/api/shifts/create`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify(shiftData)
      });

      console.log('  Status:', res.status);

      const data = await res.json();
      console.log('  📥 Response:', data);

      if (data.success) {
        console.log('  ✓ Shift created successfully!');
        console.log('  📍 Shift DB ID:', data.shift._id);
        
        // STEP 2: Assign roles to the shift immediately
        // Prefer per-staff role selections; fall back to explicit roleAssignments fields
        let triageNurseId = roleAssignments.triageNurseId || null;
        let clinicalNurseId = roleAssignments.clinicalNurseId || null;
        let clinicalNurseContext = roleAssignments.clinicalNurseContext || 'CLINICAL_NURSE_ER';
        let physicianId = roleAssignments.physicianId || null;
        let physicianContext = roleAssignments.physicianContext || 'PHYSICIAN_ER';

        if (selectedStaffRoles && Object.keys(selectedStaffRoles).length > 0) {
          for (const [sid, info] of Object.entries(selectedStaffRoles)) {
            if (!info || !info.role) continue;
            if (info.role === 'TRIAGE_NURSE' && !triageNurseId) {
              triageNurseId = sid;
            }
            if (info.role === 'CLINICAL_NURSE' && !clinicalNurseId) {
              clinicalNurseId = sid;
              clinicalNurseContext = info.context || clinicalNurseContext;
            }
            if (info.role === 'PHYSICIAN' && !physicianId) {
              physicianId = sid;
              physicianContext = info.context || physicianContext;
            }
          }
        }

        const rolePayload = {
          shiftId: data.shift._id,
          shiftID: data.shift._id, // legacy key compatibility
          triageNurseId: triageNurseId || null,
          triageNurse: triageNurseId || null, // legacy
          clinicalNurseId: clinicalNurseId || null,
          clinicalNurse: clinicalNurseId || null, // legacy
          clinicalNurseContext: clinicalNurseContext,
          physicianId: physicianId || null,
          physician: physicianId || null, // legacy
          physicianContext: physicianContext,
          physicianSpecialty: roleAssignments.physicianSpecialty || 'GENERAL'
        };

        console.log('  🎯 ASSIGNING ROLES:', rolePayload);

        // Try plural path first, fallback to singular if 404. Read raw text first to avoid body-consume errors.
        const tryAssign = async (url) => {
          const res = await fetch(url, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${token}`
            },
            body: JSON.stringify(rolePayload)
          });
          const raw = await res.text();
          let parsed;
          try {
            parsed = raw ? JSON.parse(raw) : { success: res.ok };
          } catch (e) {
            parsed = { success: res.ok, message: raw || `HTTP ${res.status}` };
          }
          return { res, parsed };
        };

        let roleRes, roleData;
        ({ res: roleRes, parsed: roleData } = await tryAssign(`${BACKEND_URL}/api/shifts/assign-clinical-team`));
        if (roleRes.status === 404) {
          console.warn('assign-clinical-team returned 404 at /api/shifts, trying /api/shift');
          ({ res: roleRes, parsed: roleData } = await tryAssign(`${BACKEND_URL}/api/shift/assign-clinical-team`));
        }

        console.log('  📥 Role Assignment Response:', roleData);

        if (roleData && roleData.success) {
          console.log('  ✅ BOTH shift creation AND role assignment succeeded!');
          setShiftMessages({ 
            success: 'Shift created and roles assigned! Staff can now see their dashboards.', 
            error: '' 
          });
        } else {
          console.log('  ⚠️ Shift created but role assignment failed:', roleData.message);
          setShiftMessages({ 
            success: 'Shift created, but role assignment failed. You can reassign roles later.', 
            error: roleData.message 
          });
        }

        // Reset forms
        setShiftForm({
          shiftName: '',
          department: 'ED',
          shiftStart: '08:00',
          shiftEnd: '16:00',
          shiftDate: new Date().toISOString().split('T')[0],
          maxStaffRequired: 1,
          assignedStaff: []
        });
        setRoleAssignments({
          triageNurseId: '',
          clinicalNurseId: '',
          clinicalNurseContext: 'CLINICAL_NURSE_ER',
          physicianId: '',
          physicianContext: 'PHYSICIAN_ER'
        });
        setSelectedStaffRoles({});
        setSelectedStaffForShift([]);
        
        // Refresh schedule
        await fetchTodaySchedule();
        setTimeout(() => setActiveTab('schedule'), 2000);
      } else {
        console.log('  ❌ Backend error:', data.message);
        setShiftMessages({ success: '', error: data.message || 'Failed to create shift' });
      }
    } catch (err) {
      console.log('  ❌ Exception:', err);
      setShiftMessages({ success: '', error: err.message });
    }
  };

  // ============ ASSIGN STAFF TO SHIFT ============
  const handleAssignStaffToShift = async (shiftId, staffId) => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/shifts/${shiftId}/assign-staff`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({ staffID: staffId })
      });

      const data = await res.json();
      if (data.success) {
        setMessages({ success: 'Staff assigned successfully', error: '' });
        await fetchTodaySchedule();
      } else {
        setMessages({ success: '', error: data.message || 'Failed to assign staff' });
      }
    } catch (err) {
      setMessages({ success: '', error: err.message });
    }
  };

  // ============ AUTO-LOAD DATA ============
  useEffect(() => {
    if (activeTab === 'search') {
      fetchStaff();
    } else if (activeTab === 'schedule') {
      fetchTodaySchedule();
    } else if (activeTab === 'add-shift') {
      fetchStaff(); // Load staff for shift assignment
    }
  }, [activeTab, searchQuery, departmentFilter, roleFilter, targetDate]);

  // ============ FILTER STAFF FOR ASSIGNMENT ============
  const getFilteredStaffForAssignment = () => {
    return filteredStaff.filter(staff => {
      // Search by name, surname, or email
      const matchesSearch = !staffSearchQuery || 
        (staff.firstName || staff.name)?.toLowerCase().includes(staffSearchQuery.toLowerCase()) ||
        (staff.lastName || staff.surname)?.toLowerCase().includes(staffSearchQuery.toLowerCase()) ||
        staff.email?.toLowerCase().includes(staffSearchQuery.toLowerCase());

      // Filter by department
      const matchesDept = staffDepartmentFilter === 'all' || (staff.assignment?.department || staff.department) === staffDepartmentFilter;

      // Filter by role
      const matchesRole = staffRoleFilter === 'all' || (staff.systemRole || staff.role) === staffRoleFilter;

      return matchesSearch && matchesDept && matchesRole;
    });
  };

  // ============ RENDER ============
  return (
    <SidebarLayout username={decoded?.name || 'Staff'} handleLogout={() => {}}>
      <div style={{ padding: '20px', maxWidth: '1400px', margin: '0 auto' }}>
        {/* Header */}
        <h1 style={{ marginTop: 0, marginBottom: '20px', color: '#1e293b' }}>
          Staff Scheduling
        </h1>

        {/* Tab Navigation (styled like Settings tabs) */}
        <div style={{ display: 'flex', gap: 12, marginBottom: 20, borderBottom: '2px solid #e5e7eb', overflowX: 'auto' }}>
          {[
            { id: 'schedule', label: '24-Hour Schedule', icon: <FiClock size={18} /> },
            { id: 'search', label: 'Staff Search', icon: <FiSearch size={18} /> },
            ...(isAdmin ? [{ id: 'add-shift', label: 'Add Shift', icon: <FiPlus size={18} /> }] : [])
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => {
                setActiveTab(tab.id);
                // Reset assignment filters when leaving
                setStaffSearchQuery('');
                setStaffDepartmentFilter('all');
                setStaffRoleFilter('all');
              }}
              style={{
                padding: '12px 16px',
                backgroundColor: 'transparent',
                border: 'none',
                borderBottom: activeTab === tab.id ? '3px solid #0f172a' : 'none',
                cursor: 'pointer',
                fontSize: 14,
                fontWeight: activeTab === tab.id ? 700 : 500,
                color: activeTab === tab.id ? '#0f172a' : '#64748b',
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                whiteSpace: 'nowrap'
              }}
            >
              {tab.icon} {tab.label}
            </button>
          ))}
        </div>

        {/* Messages */}
        {messages.success && (
          <div style={{ padding: '12px', background: '#dcfce7', color: '#166534', borderRadius: '4px', marginBottom: '15px' }}>
            ✓ {messages.success}
          </div>
        )}
        {messages.error && (
          <div style={{ padding: '12px', background: '#fee2e2', color: '#991b1b', borderRadius: '4px', marginBottom: '15px' }}>
            ✕ {messages.error}
          </div>
        )}

        {/* 24-Hour Schedule Tab */}
        {activeTab === 'schedule' && (
          <div style={{ background: '#fff', padding: '20px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
            <div style={{ marginBottom: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h2 style={{ marginTop: 0, color: '#1e293b' }}>24-Hour Schedule</h2>
              <input
                type="date"
                value={targetDate}
                onChange={(e) => setTargetDate(e.target.value)}
                style={{
                  padding: '8px 12px',
                  border: '1px solid #cbd5e1',
                  borderRadius: '4px',
                  fontSize: '14px'
                }}
              />
            </div>

            {scheduleLoading ? (
              <div style={{ textAlign: 'center', padding: '40px', color: '#64748b' }}>Loading schedule...</div>
            ) : todaySchedule.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px', color: '#64748b' }}>
                No shifts scheduled for this date
              </div>
            ) : (
              <div style={{ display: 'grid', gap: '15px' }}>
                {todaySchedule.map((shift) => (
                  <div
                    key={shift._id}
                    style={{
                      padding: '15px',
                      border: '1px solid #cbd5e1',
                      borderRadius: '6px',
                      background: '#f8fafc'
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '10px' }}>
                      <div>
                        <h3 style={{ margin: '0 0 5px 0', color: '#1e293b' }}>{shift.shiftName}</h3>
                        <div style={{ fontSize: '14px', color: '#64748b' }}>
                          <span style={{ fontWeight: '500' }}>{shift.department}</span> • 
                          <span style={{ marginLeft: '5px' }}>{shift.shiftStart} - {shift.shiftEnd}</span>
                        </div>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <div style={{ fontSize: '12px', color: '#64748b' }}>
                          {shift.assignedStaff?.length || 0} / {shift.maxStaffRequired} assigned
                        </div>
                        {shift.status && (
                          <span style={{
                            display: 'inline-block',
                            padding: '4px 8px',
                            borderRadius: '4px',
                            fontSize: '12px',
                            fontWeight: '500',
                            background: shift.status === 'PUBLISHED' ? '#dcfce7' : '#fef3c7',
                            color: shift.status === 'PUBLISHED' ? '#166534' : '#92400e',
                            marginTop: '5px'
                          }}>
                            {shift.status}
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Assigned Staff */}
                    <div style={{ marginTop: '15px' }}>
                      <h4 style={{ margin: '0 0 10px 0', fontSize: '13px', color: '#64748b' }}>Assigned Staff:</h4>
                      {shift.assignedStaff && shift.assignedStaff.length > 0 ? (
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                          {shift.assignedStaff.map((staff, idx) => (
                            <div
                              key={idx}
                              style={{
                                padding: '6px 10px',
                                background: '#ecfdf5',
                                border: '1px solid #a7f3d0',
                                borderRadius: '4px',
                                fontSize: '13px',
                                color: '#065f46'
                              }}
                            >
                              {staff.firstName || staff.name} {staff.lastName || staff.surname} ({staff.type || staff.systemRole})
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div style={{ fontSize: '13px', color: '#cbd5e1', fontStyle: 'italic' }}>
                          No staff assigned yet
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Staff Search Tab */}
        {activeTab === 'search' && (
          <div style={{ background: '#fff', padding: '20px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
            <h2 style={{ marginTop: 0, color: '#1e293b', marginBottom: '20px' }}>Search Staff</h2>

            {/* Filters */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '15px',
              marginBottom: '20px',
              padding: '15px',
              background: '#f8fafc',
              borderRadius: '6px'
            }}>
              <div>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: '500', marginBottom: '5px', color: '#475569' }}>
                  Search by Name/Email
                </label>
                <input
                  type="text"
                  placeholder="Search..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    border: '1px solid #cbd5e1',
                    borderRadius: '4px',
                    fontSize: '14px',
                    boxSizing: 'border-box'
                  }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: '500', marginBottom: '5px', color: '#475569' }}>
                  Department
                </label>
                <select
                  value={departmentFilter}
                  onChange={(e) => setDepartmentFilter(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    border: '1px solid #cbd5e1',
                    borderRadius: '4px',
                    fontSize: '14px'
                  }}
                >
                  <option value="all">All Departments</option>
                  {DEPARTMENTS.map(dept => (
                    <option key={dept} value={dept}>{dept}</option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: '500', marginBottom: '5px', color: '#475569' }}>
                  Role
                </label>
                <select
                  value={roleFilter}
                  onChange={(e) => setRoleFilter(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    border: '1px solid #cbd5e1',
                    borderRadius: '4px',
                    fontSize: '14px'
                  }}
                >
                  <option value="all">All Roles</option>
                  {ROLES.map(role => (
                    <option key={role} value={role}>{role}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Staff List */}
            {searchLoading ? (
              <div style={{ textAlign: 'center', padding: '40px', color: '#64748b' }}>Loading staff...</div>
            ) : filteredStaff.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px', color: '#64748b' }}>
                No staff members found
              </div>
            ) : (
              <div style={{ display: 'grid', gap: '12px' }}>
                {filteredStaff.map((staff) => (
                  <div
                    key={staff._id}
                    style={{
                      padding: '15px',
                      border: '1px solid #cbd5e1',
                      borderRadius: '6px',
                      background: '#f8fafc',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center'
                    }}
                  >
                    <div>
                      <h4 style={{ margin: '0 0 5px 0', color: '#1e293b' }}>
                        {staff.firstName || staff.name} {staff.lastName || staff.surname}
                      </h4>
                      <div style={{ fontSize: '13px', color: '#64748b' }}>
                        {staff.systemRole || staff.role} • {staff.assignment?.department || staff.department}
                        {(staff.clinicalSpecialty || staff.specialty) && ` • ${staff.clinicalSpecialty || staff.specialty}`}
                      </div>
                      <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '3px' }}>
                        {staff.email}
                      </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      {(staff.isActive === true || staff.status === 'ACTIVE') ? (
                        <span style={{
                          display: 'inline-block',
                          padding: '4px 8px',
                          background: '#dcfce7',
                          color: '#166534',
                          borderRadius: '4px',
                          fontSize: '12px',
                          fontWeight: '500'
                        }}>
                          Active
                        </span>
                      ) : (
                        <span style={{
                          display: 'inline-block',
                          padding: '4px 8px',
                          background: '#fee2e2',
                          color: '#991b1b',
                          borderRadius: '4px',
                          fontSize: '12px',
                          fontWeight: '500'
                        }}>
                          Inactive
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Add Shift Tab (Admin only) */}
        {activeTab === 'add-shift' && isAdmin && (
          <div style={{ background: '#fff', padding: '20px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
            <h2 style={{ marginTop: 0, color: '#1e293b', marginBottom: '20px' }}>Create New Shift</h2>

            {shiftMessages.success && (
              <div style={{ padding: '12px', background: '#dcfce7', color: '#166534', borderRadius: '4px', marginBottom: '15px' }}>
                ✓ {shiftMessages.success}
              </div>
            )}
            {shiftMessages.error && (
              <div style={{ padding: '12px', background: '#fee2e2', color: '#991b1b', borderRadius: '4px', marginBottom: '15px' }}>
                ✕ {shiftMessages.error}
              </div>
            )}

            <form onSubmit={handleCreateShift} style={{ maxWidth: '600px' }}>
              <div style={{ marginBottom: '15px' }}>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: '500', marginBottom: '5px', color: '#475569' }}>
                  Shift Name *
                </label>
                <input
                  type="text"
                  placeholder="e.g., Morning ED Shift"
                  value={shiftForm.shiftName}
                  onChange={(e) => setShiftForm({ ...shiftForm, shiftName: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    border: '1px solid #cbd5e1',
                    borderRadius: '4px',
                    fontSize: '14px',
                    boxSizing: 'border-box'
                  }}
                  required
                />
              </div>

              <div style={{ marginBottom: '15px' }}>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: '500', marginBottom: '5px', color: '#475569' }}>
                  Department *
                </label>
                <select
                  value={shiftForm.department}
                  onChange={(e) => setShiftForm({ ...shiftForm, department: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    border: '1px solid #cbd5e1',
                    borderRadius: '4px',
                    fontSize: '14px'
                  }}
                  required
                >
                  {DEPARTMENTS.map(dept => (
                    <option key={dept} value={dept}>{dept}</option>
                  ))}
                </select>
              </div>

              <div style={{ marginBottom: '15px' }}>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: '500', marginBottom: '5px', color: '#475569' }}>
                  Shift Date *
                </label>
                <input
                  type="date"
                  value={shiftForm.shiftDate}
                  onChange={(e) => setShiftForm({ ...shiftForm, shiftDate: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    border: '1px solid #cbd5e1',
                    borderRadius: '4px',
                    fontSize: '14px',
                    boxSizing: 'border-box'
                  }}
                />
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', marginBottom: '15px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '13px', fontWeight: '500', marginBottom: '5px', color: '#475569' }}>
                    Start Time
                  </label>
                  <input
                    type="time"
                    value={shiftForm.shiftStart}
                    onChange={(e) => setShiftForm({ ...shiftForm, shiftStart: e.target.value })}
                    style={{
                      width: '100%',
                      padding: '8px 12px',
                      border: '1px solid #cbd5e1',
                      borderRadius: '4px',
                      fontSize: '14px',
                      boxSizing: 'border-box'
                    }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '13px', fontWeight: '500', marginBottom: '5px', color: '#475569' }}>
                    End Time
                  </label>
                  <input
                    type="time"
                    value={shiftForm.shiftEnd}
                    onChange={(e) => setShiftForm({ ...shiftForm, shiftEnd: e.target.value })}
                    style={{
                      width: '100%',
                      padding: '8px 12px',
                      border: '1px solid #cbd5e1',
                      borderRadius: '4px',
                      fontSize: '14px',
                      boxSizing: 'border-box'
                    }}
                  />
                </div>
              </div>

              <div style={{ marginBottom: '15px' }}>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: '500', marginBottom: '5px', color: '#475569' }}>
                  Staff Required
                </label>
                <input
                  type="number"
                  min="1"
                  value={shiftForm.maxStaffRequired}
                  onChange={(e) => setShiftForm({ ...shiftForm, maxStaffRequired: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    border: '1px solid #cbd5e1',
                    borderRadius: '4px',
                    fontSize: '14px',
                    boxSizing: 'border-box'
                  }}
                />
              </div>

              

              {/* Staff Selection Section */}
              <div style={{ marginBottom: '15px', paddingTop: '15px', borderTop: '1px solid #cbd5e1' }}>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: '500', marginBottom: '10px', color: '#475569' }}>
                  Assign Staff Members
                </label>

                {/* Search and Filter Controls */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '10px', marginBottom: '10px' }}>
                  <div>
                    <input
                      type="text"
                      placeholder="Search by name or email..."
                      value={staffSearchQuery}
                      onChange={(e) => setStaffSearchQuery(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '8px 12px',
                        border: '1px solid #cbd5e1',
                        borderRadius: '4px',
                        fontSize: '13px',
                        boxSizing: 'border-box'
                      }}
                    />
                  </div>

                  <div>
                    <select
                      value={staffDepartmentFilter}
                      onChange={(e) => setStaffDepartmentFilter(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '8px 12px',
                        border: '1px solid #cbd5e1',
                        borderRadius: '4px',
                        fontSize: '13px'
                      }}
                    >
                      <option value="all">All Departments</option>
                      {DEPARTMENTS.map(dept => (
                        <option key={dept} value={dept}>{dept}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <select
                      value={staffRoleFilter}
                      onChange={(e) => setStaffRoleFilter(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '8px 12px',
                        border: '1px solid #cbd5e1',
                        borderRadius: '4px',
                        fontSize: '13px'
                      }}
                    >
                      <option value="all">All Roles</option>
                      {ROLES.map(role => (
                        <option key={role} value={role}>{role}</option>
                      ))}
                    </select>
                  </div>
                </div>

                {searchLoading ? (
                  <div style={{ textAlign: 'center', padding: '20px', color: '#64748b' }}>Loading available staff...</div>
                ) : getFilteredStaffForAssignment().length === 0 ? (
                  <div style={{ textAlign: 'center', padding: '20px', color: '#64748b' }}>
                    No staff members match your filters
                  </div>
                ) : (
                  <div style={{ display: 'grid', gap: '10px', maxHeight: '400px', overflowY: 'auto', border: '1px solid #cbd5e1', padding: '10px', borderRadius: '4px' }}>
                    {getFilteredStaffForAssignment().map((staff) => (
                      <label
                        key={staff._id}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          padding: '10px',
                          border: '1px solid #e2e8f0',
                          borderRadius: '4px',
                          background: selectedStaffForShift.includes(staff._id) ? '#ecfdf5' : '#f8fafc',
                          cursor: 'pointer',
                          transition: 'all 200ms ease'
                        }}
                      >
                        <input
                          type="checkbox"
                          checked={selectedStaffForShift.includes(staff._id)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              console.log(`[STAFF SELECTION] Added: ${staff.firstName || staff.name} ${staff.lastName || staff.surname} (${staff._id})`);
                              setSelectedStaffForShift([...selectedStaffForShift, staff._id]);
                            } else {
                              console.log(`[STAFF SELECTION] Removed: ${staff.firstName || staff.name} ${staff.lastName || staff.surname} (${staff._id})`);
                              setSelectedStaffForShift(selectedStaffForShift.filter(id => id !== staff._id));
                            }
                          }}
                          style={{
                            marginRight: '10px',
                            cursor: 'pointer'
                          }}
                        />
                        <div style={{ flex: 1 }}>
                          <div style={{ fontSize: '14px', fontWeight: '500', color: '#1e293b' }}>
                            {staff.firstName || staff.name} {staff.lastName || staff.surname}
                          </div>
                          <div style={{ fontSize: '12px', color: '#64748b' }}>
                            {staff.systemRole || staff.role} • {staff.assignment?.department || staff.department}
                          </div>
                          {/* Role assignment dropdown for this staff member */}
                          <div style={{ marginTop: '8px', display: 'flex', gap: '8px', alignItems: 'center' }}>
                            <div style={{ flex: 1 }}>
                              <label style={{ display: 'block', fontSize: '11px', fontWeight: '600', marginBottom: '4px', color: '#dc2626' }}>
                                ROLE * (required)
                              </label>
                              <select
                                value={selectedStaffRoles[staff._id]?.role || ''}
                                onChange={(e) => {
                                  const newRole = e.target.value;
                                  setSelectedStaffRoles(prev => ({
                                    ...prev,
                                    [staff._id]: {
                                      role: newRole,
                                      context: prev[staff._id]?.context || (newRole === 'PHYSICIAN' ? 'PHYSICIAN_ER' : 'CLINICAL_NURSE_ER')
                                    }
                                  }));
                                  // ensure staff is selected when assigning a role
                                  if (newRole && !selectedStaffForShift.includes(staff._id)) {
                                    setSelectedStaffForShift(prev => [...prev, staff._id]);
                                  }
                                }}
                                style={{ 
                                  padding: '8px 10px', 
                                  borderRadius: 6, 
                                  border: !selectedStaffRoles[staff._id]?.role ? '2px solid #dc2626' : '1px solid #e2e8f0', 
                                  fontSize: 13,
                                  backgroundColor: '#fff',
                                  fontWeight: 500,
                                  color: selectedStaffRoles[staff._id]?.role ? '#1e293b' : '#999'
                                }}
                                required
                              >
                                <option value="">← Select Role</option>
                                <option value="TRIAGE_NURSE">🔵 Triage Nurse</option>
                                <option value="CLINICAL_NURSE">🔴 Clinical Nurse</option>
                                <option value="PHYSICIAN">🟩 Physician</option>
                              </select>
                            </div>

                            <div style={{ flex: 1 }}>
                              <label style={{ display: 'block', fontSize: '11px', fontWeight: '600', marginBottom: '4px', color: '#6b7280' }}>
                                CONTEXT
                              </label>
                              <select
                                value={selectedStaffRoles[staff._id]?.context || ''}
                                onChange={(e) => setSelectedStaffRoles(prev => ({
                                  ...prev,
                                  [staff._id]: { ...(prev[staff._id] || {}), context: e.target.value }
                                }))}
                                disabled={!selectedStaffRoles[staff._id] || !['CLINICAL_NURSE','PHYSICIAN'].includes(selectedStaffRoles[staff._id].role)}
                                style={{ 
                                  padding: '8px 10px', 
                                  borderRadius: 6, 
                                  border: '1px solid #e2e8f0', 
                                  fontSize: 13,
                                  backgroundColor: (!selectedStaffRoles[staff._id] || !['CLINICAL_NURSE','PHYSICIAN'].includes(selectedStaffRoles[staff._id].role)) ? '#f3f4f6' : '#fff',
                                  opacity: (!selectedStaffRoles[staff._id] || !['CLINICAL_NURSE','PHYSICIAN'].includes(selectedStaffRoles[staff._id].role)) ? 0.5 : 1
                                }}
                              >
                                <option value="">Select Context</option>
                                <option value="CLINICAL_NURSE_ER">ED</option>
                                <option value="CLINICAL_NURSE_ICU">ICU</option>
                                <option value="CLINICAL_NURSE_SURGICAL">Surgical</option>
                                <option value="PHYSICIAN_ER">ED (Phys.)</option>
                                <option value="PHYSICIAN_ICU">ICU (Phys.)</option>
                                <option value="PHYSICIAN_SURGICAL">Surgical (Phys.)</option>
                              </select>
                            </div>
                          </div>
                        </div>
                        <span style={{
                          fontSize: '11px',
                          padding: '3px 8px',
                          background: (staff.isActive === true || staff.status === 'ACTIVE') ? '#dcfce7' : '#fee2e2',
                          color: (staff.isActive === true || staff.status === 'ACTIVE') ? '#166534' : '#991b1b',
                          borderRadius: '3px',
                          fontWeight: '500'
                        }}>
                          {(staff.isActive === true || staff.status === 'ACTIVE') ? 'Active' : 'Inactive'}
                        </span>
                      </label>
                    ))}
                  </div>
                )}
                <div style={{ fontSize: '12px', color: '#64748b', marginTop: '8px' }}>
                  {selectedStaffForShift.length} staff member{selectedStaffForShift.length !== 1 ? 's' : ''} selected
                </div>
              </div>

              {/* Role Assignment Summary */}
              {selectedStaffForShift.length > 0 && (
                <div style={{
                  marginTop: '20px',
                  padding: '15px',
                  background: '#f0fdf4',
                  border: '1px solid #bbf7d0',
                  borderRadius: '8px'
                }}>
                  <h4 style={{ margin: '0 0 12px 0', color: '#15803d', fontSize: '14px', fontWeight: '600' }}>
                    📋 Role Assignment Summary
                  </h4>
                  <div style={{ display: 'grid', gap: '8px' }}>
                    {selectedStaffForShift.map(staffId => {
                      const staff = filteredStaff.find(s => s._id === staffId);
                      const roleInfo = selectedStaffRoles[staffId];
                      const roleName = roleInfo?.role || '(No role assigned)';
                      const context = roleInfo?.context || '(No context)';
                      
                      return (
                        <div key={staffId} style={{
                          padding: '8px 10px',
                          background: '#fff',
                          border: '1px solid #d1fae5',
                          borderRadius: '6px',
                          fontSize: '13px',
                          color: '#047857'
                        }}>
                          <strong>{staff?.firstName || staff?.name} {staff?.lastName || staff?.surname}</strong>
                          {roleInfo ? (
                            <>
                              <span style={{ display: 'block', fontSize: '12px', marginTop: '4px' }}>
                                → {roleName} {context !== '(No context)' ? `(${context.replace(/_/g, ' ')})` : ''}
                              </span>
                            </>
                          ) : (
                            <span style={{ display: 'block', fontSize: '12px', marginTop: '4px', color: '#ca8a04' }}>
                              ⚠️ No role assigned - will be added as generic staff
                            </span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              <button
                type="submit"
                style={{
                  marginTop: '20px',
                  padding: '12px 16px',
                  backgroundColor: '#10b981',
                  color: '#fff',
                  border: 'none',
                  borderRadius: 8,
                  cursor: 'pointer',
                  fontWeight: 600,
                  fontSize: 14,
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 8
                }}
              >
                ✓ Create Shift
              </button>
            </form>
          </div>
        )}
      </div>
    </SidebarLayout>
  );
};

export default StaffScheduleManager;
