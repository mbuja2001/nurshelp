import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FiSearch, FiUsers, FiMessageCircle, FiClock, FiPause, FiPower, FiArrowRight } from 'react-icons/fi';
import SidebarLayout from '../SidebarLayout';
import { decodeJwt } from '../../utils/jwt';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

const fallbackStaff = [
  {
    _id: 'staff-1',
    name: 'Dr. Amina Hassan',
    role: 'Physician',
    department: 'ED',
    status: 'CLOCKED_IN',
    lastActiveAt: new Date(Date.now() - 60_000).toISOString(),
    location: 'ED - Zone 1',
    isOnShift: true,
    shiftNote: 'Leading morning triage',
  },
  {
    _id: 'staff-2',
    name: 'Nurse Samuel Opoku',
    role: 'RN',
    department: 'ICU',
    status: 'ON_BREAK',
    lastActiveAt: new Date(Date.now() - 8 * 60_1000).toISOString(),
    location: 'ICU - Bed 4',
    isOnShift: true,
    shiftNote: 'Lunch break',
  },
  {
    _id: 'staff-3',
    name: 'Phlebotomist Julia Mwangi',
    role: 'Lab Tech',
    department: 'Lab',
    status: 'CLOCKED_OUT',
    lastActiveAt: new Date(Date.now() - 75 * 60_1000).toISOString(),
    location: 'Specimen Collection',
    isOnShift: true,
    shiftNote: 'Scheduled for afternoon return',
  },
  {
    _id: 'staff-4',
    name: 'Dr. Nia Chen',
    role: 'Radiologist',
    department: 'Radiology',
    status: 'CLOCKED_OUT',
    lastActiveAt: new Date(Date.now() - 180 * 60_1000).toISOString(),
    location: 'Radiology Review',
    isOnShift: false,
    shiftNote: 'Off shift',
  },
  {
    _id: 'staff-5',
    name: 'Nurse Isaac Awuah',
    role: 'RN',
    department: 'OR',
    status: 'CLOCKED_IN',
    lastActiveAt: new Date(Date.now() - 2 * 60_1000).toISOString(),
    location: 'Operating Room 2',
    isOnShift: true,
    shiftNote: 'Circulating nurse for OR case',
  }
];

const statusColors = {
  CLOCKED_IN: '#10b981',
  ON_BREAK: '#f59e0b',
  CLOCKED_OUT: '#6b7280',
};

const statusLabels = {
  CLOCKED_IN: 'On Shift',
  ON_BREAK: 'Lunch Break',
  CLOCKED_OUT: 'Offline',
};

const isRecentActive = (lastActiveAt) => {
  if (!lastActiveAt) return false;
  return Date.now() - new Date(lastActiveAt).getTime() < 5 * 60_1000;
};

const StaffTeamPage = () => {
  const navigate = useNavigate();
  const token = localStorage.getItem('token');
  const decoded = decodeJwt(token) || {};
  const [staff, setStaff] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');

  useEffect(() => {
    const fetchStatus = async () => {
      setLoading(true);
      setError('');
      try {
        const res = await fetch(`${BACKEND_URL}/api/manager/staff-status`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const data = await res.json();
        if (data.success && Array.isArray(data.staff)) {
          setStaff(data.staff.map((row) => ({
            _id: row._id || row.id || `${row.name}-${row.role}`,
            name: row.nurse?.name || row.name || 'Unknown',
            role: row.nurse?.role || row.role || row.systemRole || 'Staff',
            department: row.location || row.department || 'Unknown',
            status: row.status || (row.clockInTime ? 'CLOCKED_IN' : 'CLOCKED_OUT'),
            lastActiveAt: row.lastActiveAt || row.updatedAt || row.clockInTime,
            location: row.location || row.department || 'Unknown',
            isOnShift: row.isOnShift ?? !!row.clockInTime,
            shiftNote: row.notes || row.shiftNote || '',
          })));
        } else {
          throw new Error(data.message || 'No staff status available');
        }
      } catch (err) {
        setError('Unable to load live staff status. Showing roster preview.');
        setStaff(fallbackStaff);
      } finally {
        setLoading(false);
      }
    };

    fetchStatus();
  }, [token]);

  const processedStaff = useMemo(() => {
    const filtered = staff.filter((member) => {
      const query = search.trim().toLowerCase();
      const matchesSearch = !query ||
        member.name.toLowerCase().includes(query) ||
        member.role.toLowerCase().includes(query) ||
        member.department.toLowerCase().includes(query) ||
        member.location.toLowerCase().includes(query);

      const matchesStatus =
        statusFilter === 'all' ||
        (statusFilter === 'online' && isRecentActive(member.lastActiveAt)) ||
        (statusFilter === 'offline' && !isRecentActive(member.lastActiveAt) && member.status === 'CLOCKED_OUT') ||
        (statusFilter === 'on-shift' && member.status === 'CLOCKED_IN') ||
        (statusFilter === 'break' && member.status === 'ON_BREAK') ||
        (statusFilter === 'offline-shift' && member.isOnShift && member.status === 'CLOCKED_OUT');

      return matchesSearch && matchesStatus;
    });

    return filtered.sort((a, b) => a.name.localeCompare(b.name));
  }, [staff, search, statusFilter]);

  const summary = useMemo(() => {
    return staff.reduce(
      (acc, member) => {
        const online = isRecentActive(member.lastActiveAt);
        if (member.status === 'CLOCKED_IN') acc.onShift += 1;
        if (member.status === 'ON_BREAK') acc.break += 1;
        if (online) acc.online += 1;
        if (member.isOnShift && member.status === 'CLOCKED_OUT') acc.offlineOnShift += 1;
        if (member.status === 'CLOCKED_OUT') acc.offline += 1;
        return acc;
      },
      { onShift: 0, break: 0, online: 0, offlineOnShift: 0, offline: 0 }
    );
  }, [staff]);

  return (
    <SidebarLayout username={decoded?.name || 'Staff'}>
      <div style={{ width: '100%', padding: 24, maxWidth: 1320, margin: '0 auto' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 18, alignItems: 'flex-start', marginBottom: 24 }}>
          <div>
            <h1 style={{ margin: '0 0 8px 0', fontSize: 34, color: '#0f172a' }}>Staff Team</h1>
            <p style={{ margin: 0, color: '#475569', fontSize: 15, maxWidth: 680 }}>
              View the current on-shift roster, lunch break coverage, online team members, and staff who remain assigned even while offline.
            </p>
          </div>
          <button
            onClick={() => navigate('/staff-chat')}
            style={{ display: 'inline-flex', alignItems: 'center', gap: 10, border: 'none', borderRadius: 16, padding: '14px 18px', background: '#2563eb', color: '#fff', cursor: 'pointer', fontWeight: 600 }}
          >
            <FiMessageCircle size={18} /> Team Chat
          </button>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 16, marginBottom: 24 }}>
          {[
            { title: 'On Shift', value: summary.onShift, color: '#d1fae5', accent: '#10b981', icon: <FiClock size={18} /> },
            { title: 'Lunch Break', value: summary.break, color: '#fef3c7', accent: '#d97706', icon: <FiPause size={18} /> },
            { title: 'Online Now', value: summary.online, color: '#dbeafe', accent: '#2563eb', icon: <FiUsers size={18} /> },
            { title: 'Offline but Assigned', value: summary.offlineOnShift, color: '#e2e8f0', accent: '#6b7280', icon: <FiPower size={18} /> },
          ].map((card) => (
            <div key={card.title} style={{ borderRadius: 20, padding: 20, background: card.color, border: `1px solid ${card.accent}20` }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
                <div style={{ fontSize: 15, fontWeight: 700, color: '#0f172a' }}>{card.title}</div>
                <div style={{ color: card.accent }}>{card.icon}</div>
              </div>
              <div style={{ fontSize: 36, fontWeight: 700, color: '#0f172a' }}>{card.value}</div>
            </div>
          ))}
        </div>

        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 18 }}>
          <input
            type="search"
            aria-label="Search staff"
            placeholder="Search staff name, role, department, or location"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ flex: 1, minWidth: 240, padding: '12px 14px', borderRadius: 16, border: '1px solid #cbd5e1', background: '#fff' }}
          />

          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            style={{ padding: '12px 14px', borderRadius: 16, border: '1px solid #cbd5e1', background: '#fff', minWidth: 190 }}
          >
            <option value="all">All statuses</option>
            <option value="on-shift">On Shift</option>
            <option value="break">Lunch Break</option>
            <option value="online">Online</option>
            <option value="offline-shift">Offline but Assigned</option>
            <option value="offline">Offline</option>
          </select>
        </div>

        <div style={{ borderRadius: 24, border: '1px solid #e2e8f0', overflow: 'hidden', background: '#fff' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1.8fr 1fr 1fr 1fr 1fr', gap: 0, background: '#f8fafc', padding: '16px 20px', fontSize: 13, color: '#64748b', fontWeight: 700 }}>
            <div>Name</div>
            <div>Status</div>
            <div>Department</div>
            <div>Location</div>
            <div>Last Active</div>
          </div>
          {loading ? (
            <div style={{ padding: 28, color: '#475569' }}>Loading staff roster…</div>
          ) : processedStaff.length === 0 ? (
            <div style={{ padding: 28, color: '#64748b' }}>No staff match the current filter.</div>
          ) : (
            processedStaff.map((member) => {
              const online = isRecentActive(member.lastActiveAt);
              const offlineButAssigned = member.isOnShift && member.status === 'CLOCKED_OUT';
              return (
                <div key={member._id} style={{ display: 'grid', gridTemplateColumns: '1.8fr 1fr 1fr 1fr 1fr', gap: 0, alignItems: 'center', padding: '18px 20px', borderTop: '1px solid #e5e7eb' }}>
                  <div>
                    <div style={{ fontWeight: 700, color: '#0f172a' }}>{member.name}</div>
                    <div style={{ marginTop: 6, fontSize: 13, color: '#64748b' }}>{member.role}</div>
                  </div>
                  <div>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, padding: '8px 10px', borderRadius: 999, background: `${statusColors[member.status] || '#e2e8f0'}33`, color: statusColors[member.status] || '#475569', fontWeight: 700, fontSize: 12 }}>
                      {statusLabels[member.status] || member.status}
                    </span>
                    {offlineButAssigned && (
                      <div style={{ marginTop: 6, fontSize: 11, color: '#475569' }}>Assigned but offline</div>
                    )}
                  </div>
                  <div>{member.department}</div>
                  <div>{member.location}</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    <span style={{ color: online ? '#0f172a' : '#64748b', fontWeight: 600 }}>{online ? 'Online' : 'Offline'}</span>
                    <span style={{ fontSize: 12, color: '#64748b' }}>{member.lastActiveAt ? new Date(member.lastActiveAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : 'Unknown'}</span>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {error && (
          <div style={{ marginTop: 18, padding: 16, borderRadius: 16, background: '#fef2f2', color: '#991b1b' }}>
            {error}
          </div>
        )}

        <div style={{ marginTop: 24, display: 'flex', justifyContent: 'flex-end' }}>
          <button
            onClick={() => navigate('/staff-chat')}
            style={{ display: 'inline-flex', alignItems: 'center', gap: 10, border: '1px solid #cbd5e1', borderRadius: 16, background: '#fff', padding: '12px 18px', color: '#0f172a', cursor: 'pointer' }}
          >
            <FiArrowRight size={18} /> Open Team Chat
          </button>
        </div>
      </div>
    </SidebarLayout>
  );
};

export default StaffTeamPage;
