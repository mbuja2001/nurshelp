import React, { useState, useEffect } from 'react';
import { FiX, FiEdit2, FiTrash2, FiPlus, FiDownload, FiUpload, FiActivity, FiAward } from 'react-icons/fi';
import SidebarLayout from '../SidebarLayout';
import { useUserFromToken } from '../../hooks/useUserFromToken';
import { decodeJwt } from '../../utils/jwt';

const BedsAllocation = () => {
  const { userName } = useUserFromToken();
  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

  // ============ STATE ============
  const [beds, setBeds] = useState([]);
  const [wards, setWards] = useState([]);
  const [selectedWard, setSelectedWard] = useState(null);
  const [hospitalId, setHospitalId] = useState(null);
  const [hospitalStats, setHospitalStats] = useState({
    total: 0,
    available: 0,
    occupied: 0,
    cleaning: 0,
    maintenance: 0,
    occupancyRate: 0
  });
  const [loading, setLoading] = useState(false);
  const [statsLoading, setStatsLoading] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [modalMode, setModalMode] = useState('view'); // 'view', 'edit', 'add', 'assign'
  const [selectedBed, setSelectedBed] = useState(null);
  const [editForm, setEditForm] = useState({ bedNumber: '', bedType: 'Standard', status: 'available' });
  const [assignForm, setAssignForm] = useState({ encounterId: '', patientName: '', notes: '' });
  const [filter, setFilter] = useState({ status: 'all', bedType: 'all' });
  const [successMessage, setSuccessMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState(null);

  const BED_TYPES = ['Standard', 'Monitored', 'ICU', 'Isolation', 'Pediatric', 'Maternity'];
  const BED_STATUSES = [
    { value: 'Available', label: 'Available', color: '#10b981', bgColor: '#dcfce7' },
    { value: 'Occupied', label: 'Occupied', color: '#0369a1', bgColor: '#f0f9ff' },
    { value: 'Cleaning', label: 'Cleaning', color: '#9333ea', bgColor: '#f3e8ff' },
    { value: 'Maintenance', label: 'Maintenance', color: '#b91c1c', bgColor: '#fee2e2' }
  ];

  // ============ INITIALIZE - GET HOSPITAL CONTEXT ============
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      const decoded = decodeJwt(token);
      if (decoded && decoded.hospitalId) {
        setHospitalId(decoded.hospitalId);
        console.log('Hospital context:', decoded.hospitalId);
      }
    }
  }, []);

  // ============ FETCH WARDS ============
  const fetchWards = async () => {
    if (!hospitalId) return [];
    
    try {
      const token = localStorage.getItem('token');
      console.log('🏥 Fetching wards for hospital:', hospitalId);
      
      const res = await fetch(`${BACKEND_URL}/api/beds/wards?hospitalId=${hospitalId}`, {
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      });
      if (!res.ok) throw new Error('Failed to fetch wards');
      const data = await res.json();
      console.log('✅ Wards fetched:', data);
      
      const wardList = data.success ? data.data : (Array.isArray(data) ? data : []);
      setWards(wardList);
      
      // Select first ward by default
      if (wardList.length > 0 && !selectedWard) {
        setSelectedWard(wardList[0]._id);
        console.log('📍 Selected first ward:', wardList[0]._id, wardList[0].wardName);
      }
      
      return wardList;
    } catch (err) {
      console.error('❌ Error fetching wards:', err);
      setErrorMessage('Error loading wards: ' + err.message);
      return [];
    }
  };

  // ============ FETCH HOSPITAL-WIDE STATS ============
  const fetchHospitalStats = async () => {
    if (!hospitalId) {
      console.warn('⚠️ Cannot fetch hospital stats: hospitalId missing');
      return;
    }
    
    setStatsLoading(true);
    try {
      const token = localStorage.getItem('token');
      console.log('📊 Fetching hospital-wide bed stats for:', hospitalId);
      
      const res = await fetch(`${BACKEND_URL}/api/beds/stats?hospitalId=${hospitalId}`, {
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      });
      
      console.log('📡 Hospital stats response status:', res.status, res.statusText);
      
      if (!res.ok) {
        const errorText = await res.text();
        console.error('❌ API error response (status=' + res.status + '):', errorText);
        throw new Error(`Server error: ${res.status} ${res.statusText}`);
      }
      
      const data = await res.json();
      console.log('✅ Hospital stats response:', data);
      
      if (data.success && data.data) {
        console.log('📊 Setting hospital stats:', {
          total: data.data.total,
          available: data.data.available,
          occupied: data.data.occupied,
          cleaning: data.data.cleaning,
          maintenance: data.data.maintenance,
          occupancyRate: data.data.occupancyRate
        });
        
        setHospitalStats({
          total: data.data.total || 0,
          available: data.data.available || 0,
          occupied: data.data.occupied || 0,
          cleaning: data.data.cleaning || 0,
          maintenance: data.data.maintenance || 0,
          occupancyRate: data.data.occupancyRate || 0
        });
      } else {
        console.warn('⚠️ Unexpected response format:', data);
      }
    } catch (err) {
      console.error('❌ Error fetching hospital stats:', err.message);
      setErrorMessage('Failed to load hospital bed statistics: ' + err.message);
    } finally {
      setStatsLoading(false);
    }
  };

  // Fetch wards and hospital stats when hospital context is available
  useEffect(() => {
    if (hospitalId) {
      fetchWards();
      fetchHospitalStats();
      
      // Auto-refresh stats every 10 seconds for live updates
      const interval = setInterval(() => fetchHospitalStats(), 10000);
      return () => clearInterval(interval);
    }
  }, [hospitalId]);

  // ============ FETCH BEDS ============
  const fetchBeds = async (wardId = selectedWard) => {
    if (!wardId || !hospitalId) {
      console.warn('⚠️ Cannot fetch beds: wardId or hospitalId missing', { wardId, hospitalId });
      return;
    }
    
    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      console.log('🛏️  Fetching beds for ward:', wardId, 'hospital:', hospitalId);
      
      const res = await fetch(`${BACKEND_URL}/api/beds/ward/${wardId}?hospitalId=${hospitalId}`, {
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      });
      
      console.log('📡 Response status:', res.status, res.statusText);
      
      if (!res.ok) {
        console.error('❌ Fetch failed with status:', res.status);
        throw new Error(`Failed to fetch beds: ${res.statusText}`);
      }
      
      const data = await res.json();
      console.log('✅ Raw API response:', data);
      
      // Map API response to component format
      // Check if beds are in data.beds or data.data.beds
      const bedsList = data.data?.beds || data.beds || [];
      if (bedsList && bedsList.length > 0) {
        console.log('📊 Found beds:', bedsList.length);
        
        const mappedBeds = bedsList.map((bed, idx) => {
          // Log raw bed object to see actual field structure
          console.log(`🔍 Raw bed object [${idx}]:`, {
            _id: bed._id,
            bedId: bed.bedId,
            status: bed.status,
            roomNumber: bed.roomNumber,
            bedType: bed.bedType,
            hasMonitor: bed.hasMonitor,
            hasVentilator: bed.hasVentilator
          });
          
          // All fields should be present and defined
          const mappedBed = {
            _id: bed._id || `bed-${idx}`,
            bedNumber: bed.bedId || 'Unknown',
            bedId: bed.bedId || null,
            bedType: bed.bedType || 'Standard',
            status: bed.status || 'Available', // Status is now titlecase from backend
            roomNumber: bed.roomNumber || null,
            occupant: bed.currentEncounterId ? {
              patientId: bed.currentEncounterId,
              name: 'Patient', // API doesn't have patient name
              encounterId: bed.currentEncounterId,
              admittedAt: bed.occupancyStartTime || new Date().toISOString(),
              expectedDischargeDate: bed.expectedDischargeTime || null,
              notes: bed.maintenanceNotes || ''
            } : null,
            equipment: [
              bed.hasMonitor && 'Monitor',
              bed.hasVentilator && 'Ventilator',
              bed.hasDialysis && 'Dialysis Machine'
            ].filter(Boolean),
            characteristics: [
              bed.canIsolate && 'Isolation Capable',
              bed.hasMonitor && 'Patient Monitoring',
              bed.hasVentilator && 'Ventilation Support',
              bed.hasDialysis && 'Dialysis Ready'
            ].filter(Boolean),
            compatibility: {
              canIsolate: bed.canIsolate || false,
              hasMonitor: bed.hasMonitor || false,
              hasVentilator: bed.hasVentilator || false,
              hasDialysis: bed.hasDialysis || false
            },
            infectionControl: bed.canIsolate ? 'Isolation' : 'Standard',
            acuityLevel: bed.acuityLevel || 'MODERATE',
            lastCleaned: bed.lastCleanedAt ? new Date(bed.lastCleanedAt).toLocaleString() : null,
            lastCleanedFormatted: bed.lastCleanedAt ? new Date(bed.lastCleanedAt).toLocaleDateString() : null,
            notes: bed.maintenanceNotes || '',
            wardId: bed.wardId || hospitalId,
            // Preserve all raw fields for reference
            rawBed: {
              canIsolate: bed.canIsolate,
              hasMonitor: bed.hasMonitor,
              hasVentilator: bed.hasVentilator,
              hasDialysis: bed.hasDialysis,
              acuityLevel: bed.acuityLevel
            }
          };
          
          console.log(`✅ Mapped bed [${idx}]: ${mappedBed.bedNumber}`, {
            _id: mappedBed._id,
            status: mappedBed.status,
            equipment: mappedBed.equipment.length,
            occupant: mappedBed.occupant?.name || 'None'
          });
          
          return mappedBed;
        });
        
        console.log('✨ Mapped beds final:', mappedBeds);
        setBeds(mappedBeds);
      } else {
        console.warn('⚠️ No beds in response:', { success: data.success, hasData: !!data.data, hasBeds: !!data.data?.beds });
        setBeds([]);
      }
      setErrorMessage('');
    } catch (err) {
      console.error('❌ Error fetching beds:', err);
      setErrorMessage('Error loading beds: ' + err.message);
      setBeds([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedWard && hospitalId) {
      fetchBeds(selectedWard);
      
      // Auto-refresh beds every 10 seconds for live updates
      const interval = setInterval(() => fetchBeds(selectedWard), 10000);
      return () => clearInterval(interval);
    }
  }, [selectedWard, hospitalId]);

  // ============ MODAL HANDLERS ============
  const openAddModal = () => {
    setEditForm({
      bedNumber: '',
      bedType: BED_TYPES[0],
      status: 'available'
    });
    setModalMode('add');
    setShowModal(true);
  };

  const openEditModal = (bed) => {
    setSelectedBed(bed);
    setEditForm({
      bedNumber: bed.bedNumber || '',
      bedType: bed.bedType || BED_TYPES[0],
      status: bed.status || 'available'
    });
    setModalMode('edit');
    setShowModal(true);
  };

  const openAssignModal = (bed) => {
    setSelectedBed(bed);
    setAssignForm({
      encounterId: bed.occupant?.encounterId || '',
      patientName: bed.occupant?.name || '',
      notes: bed.notes || ''
    });
    setModalMode('assign');
    setShowModal(true);
  };

  const closeModal = () => {
    setShowModal(false);
    setModalMode('view');
    setSelectedBed(null);
  };

  // ============ SAVE BED ============
  const saveBed = async () => {
    if (!editForm.bedNumber.trim()) {
      setErrorMessage('Bed number is required');
      return;
    }

    try {
      const token = localStorage.getItem('token');
      const headers = {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      };

      let res;
      if (modalMode === 'add') {
        res = await fetch(`${BACKEND_URL}/api/beds`, {
          method: 'POST',
          headers,
          body: JSON.stringify({
            bedNumber: editForm.bedNumber,
            bedType: editForm.bedType,
            status: editForm.status,
          }),
        });
      } else if (modalMode === 'edit') {
        res = await fetch(`${BACKEND_URL}/api/beds/${selectedBed._id}`, {
          method: 'PATCH',
          headers,
          body: JSON.stringify({
            bedNumber: editForm.bedNumber,
            bedType: editForm.bedType,
            status: editForm.status,
          }),
        });
      }

      if (!res || !res.ok) throw new Error('Failed to save bed');
      setSuccessMessage(modalMode === 'add' ? 'Bed added successfully' : 'Bed updated successfully');
      closeModal();
      await fetchBeds();
      setTimeout(() => setSuccessMessage(''), 3000);
    } catch (err) {
      setErrorMessage(err.message);
    }
  };

  // ============ ASSIGN PATIENT ============
  const assignPatient = async () => {
    if (!assignForm.encounterId.trim() || !assignForm.patientName.trim()) {
      setErrorMessage('Encounter ID and patient name are required');
      return;
    }

    try {
      const token = localStorage.getItem('token');
      const res = await fetch(`${BACKEND_URL}/api/beds/${selectedBed._id}/assign`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          encounterId: assignForm.encounterId,
          patientName: assignForm.patientName,
          notes: assignForm.notes,
        }),
      });

      if (!res || !res.ok) throw new Error('Failed to assign patient');
      setSuccessMessage('Patient assigned successfully');
      closeModal();
      await fetchBeds();
      setTimeout(() => setSuccessMessage(''), 3000);
    } catch (err) {
      setErrorMessage(err.message);
    }
  };

  // ============ UNASSIGN PATIENT ============
  const unassignPatient = async (bed) => {
    try {
      const token = localStorage.getItem('token');
      const res = await fetch(`${BACKEND_URL}/api/beds/${bed._id}/unassign`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      });

      if (!res || !res.ok) throw new Error('Failed to unassign patient');
      setSuccessMessage('Patient unassigned successfully');
      await fetchBeds();
      setTimeout(() => setSuccessMessage(''), 3000);
    } catch (err) {
      setErrorMessage(err.message);
    }
  };

  // ============ DELETE BED ============
  const deleteBed = async () => {
    if (!deleteTarget) return;
    try {
      const token = localStorage.getItem('token');
      const res = await fetch(`${BACKEND_URL}/api/beds/${deleteTarget._id}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      });
      if (!res || !res.ok) throw new Error('Failed to delete bed');
      setSuccessMessage('Bed removed successfully');
      setShowDeleteConfirm(false);
      setDeleteTarget(null);
      await fetchBeds();
      setTimeout(() => setSuccessMessage(''), 3000);
    } catch (err) {
      setErrorMessage(err.message);
    }
  };

  // ============ FILTER & COMPUTE DISPLAY ============
  const filteredBeds = beds.filter(bed => {
    if (filter.status !== 'all' && bed.status !== filter.status) return false;
    if (filter.bedType !== 'all' && bed.bedType !== filter.bedType) return false;
    return true;
  });

  // Use hospital-wide stats instead of ward-filtered stats
  const stats = hospitalStats;

  const getStatusBadge = (status) => {
    const s = BED_STATUSES.find(x => x.value === status);
    return s || BED_STATUSES[0];
  };

  // Format date for occupancy display
  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };

  // Calculate length of stay
  const calculateLOS = (admittedAt) => {
    if (!admittedAt) return 'N/A';
    const admit = new Date(admittedAt);
    const now = new Date();
    const hours = Math.floor((now - admit) / (1000 * 60 * 60));
    if (hours < 24) return `${hours}h`;
    return `${Math.floor(hours / 24)}d`;
  };

  // ============ RENDER ============
  return (
    <SidebarLayout username={userName} handleLogout={() => {}}>
      <div style={{ backgroundColor: '#f5f7fa', minHeight: '100vh', padding: '32px' }}>
        
        {/* Professional Header */}
        <div style={{ marginBottom: '32px' }}>
          <h1 style={{ margin: '0 0 8px 0', fontSize: '32px', fontWeight: 800, color: '#0f172a', letterSpacing: '-0.5px' }}>
            Bed Management System
          </h1>
          <p style={{ margin: 0, color: '#64748b', fontSize: '15px' }}>
            Real-time bed availability, occupancy tracking, and patient allocation across hospital wards
          </p>
        </div>

        {/* Alerts */}
        {successMessage && (
          <div style={{ marginBottom: '16px', padding: '12px 16px', backgroundColor: '#d1fae5', border: '1px solid #6ee7b7', borderRadius: '8px', color: '#065f46', fontSize: '14px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>✓ {successMessage}</span>
            <button onClick={() => setSuccessMessage('')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#065f46', fontSize: '18px' }}>✕</button>
          </div>
        )}
        {errorMessage && (
          <div style={{ marginBottom: '16px', padding: '12px 16px', backgroundColor: '#fee2e2', border: '1px solid #fecaca', borderRadius: '8px', color: '#991b1b', fontSize: '14px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>⚠ {errorMessage}</span>
            <button onClick={() => setErrorMessage('')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#991b1b', fontSize: '18px' }}>✕</button>
          </div>
        )}

        {/* Professional Stats Grid with Occupancy Visualization */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '16px', marginBottom: '32px' }}>
          {[
            { label: 'Total Beds', value: stats.total, color: '#6366f1', bg: '#eef2ff', icon: '🛏️' },
            { label: 'Available', value: stats.available, color: '#10b981', bg: '#ecfdf5', icon: '✓', showBar: true, percent: stats.total > 0 ? Math.round((stats.available / stats.total) * 100) : 0 },
            { label: 'Occupied', value: stats.occupied, color: '#0369a1', bg: '#f0f9ff', icon: '👤', showBar: true, percent: stats.total > 0 ? Math.round((stats.occupied / stats.total) * 100) : 0 },
            { label: 'Cleaning', value: stats.cleaning, color: '#8b5cf6', bg: '#faf5ff', icon: '🧹', showBar: true, percent: stats.total > 0 ? Math.round((stats.cleaning / stats.total) * 100) : 0 },
            { label: 'Maintenance', value: stats.maintenance, color: '#dc2626', bg: '#fef2f2', icon: '🔧' }
          ].map(stat => (
            <div key={stat.label} style={{ 
              backgroundColor: stat.bg, 
              borderRadius: '12px', 
              padding: '16px', 
              border: `2px solid ${stat.color}22`,
              boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)',
              display: 'flex',
              flexDirection: 'column'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '10px' }}>
                <div style={{ fontSize: '22px' }}>{stat.icon}</div>
                {stat.showBar && (
                  <div style={{ 
                    fontSize: '12px', 
                    fontWeight: 700, 
                    color: stat.color,
                    backgroundColor: 'rgba(255,255,255,0.5)',
                    padding: '2px 8px',
                    borderRadius: '4px'
                  }}>
                    {stat.percent}%
                  </div>
                )}
              </div>
              <p style={{ margin: '0 0 8px 0', fontSize: '11px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px' }}>{stat.label}</p>
              <div style={{ fontSize: '26px', fontWeight: 800, color: stat.color, marginBottom: '8px' }}>{stat.value}</div>
              {stat.showBar && (
                <div style={{
                  width: '100%',
                  height: '6px',
                  backgroundColor: 'rgba(0,0,0,0.1)',
                  borderRadius: '3px',
                  overflow: 'hidden',
                  marginTop: 'auto'
                }}>
                  <div style={{
                    height: '100%',
                    width: `${stat.percent}%`,
                    backgroundColor: stat.color,
                    transition: 'width 0.3s ease',
                    borderRadius: '3px'
                  }} />
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Controls - Professional Toolbar */}
        <div style={{ 
          backgroundColor: '#fff', 
          borderRadius: '12px', 
          padding: '20px', 
          marginBottom: '32px', 
          border: '1px solid #e2e8f0',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)'
        }}>
          <div style={{ display: 'flex', gap: '16px', alignItems: 'center', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center', minWidth: '280px' }}>
              <label style={{ fontSize: '13px', fontWeight: 700, color: '#475569', whiteSpace: 'nowrap', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Ward:</label>
              <select 
                value={selectedWard || ''} 
                onChange={(e) => {
                  setSelectedWard(e.target.value);
                  console.log('🏥 Ward selected:', e.target.value);
                }}
                style={{ 
                  flex: 1,
                  padding: '10px 12px', 
                  borderRadius: '8px', 
                  border: '2px solid #e2e8f0', 
                  backgroundColor: '#fff', 
                  fontSize: '14px', 
                  cursor: 'pointer',
                  fontWeight: 600,
                  color: '#0f172a',
                  transition: 'border-color 0.2s'
                }}
              >
                <option value="">Select a Ward...</option>
                {wards.map(ward => (
                  <option key={ward._id} value={ward._id}>
                    {ward.wardName} ({ward.department || 'No Dept'})
                  </option>
                ))}
              </select>
            </div>

            <button 
              onClick={openAddModal} 
              style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '8px', 
                padding: '10px 16px', 
                backgroundColor: '#10b981', 
                color: '#fff', 
                border: 'none', 
                borderRadius: '8px', 
                cursor: 'pointer', 
                fontWeight: 600, 
                fontSize: '14px',
                transition: 'background-color 0.2s'
              }}
              onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#059669'}
              onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#10b981'}
            >
              <FiPlus size={16} /> Add Bed
            </button>

            <div style={{ flex: 1 }} />

            <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
              <select 
                value={filter.bedType} 
                onChange={(e) => setFilter(prev => ({ ...prev, bedType: e.target.value }))} 
                style={{ 
                  padding: '10px 12px', 
                  borderRadius: '8px', 
                  border: '1px solid #e2e8f0', 
                  backgroundColor: '#fff', 
                  fontSize: '13px', 
                  cursor: 'pointer',
                  fontWeight: 500
                }}
              >
                <option value="all">All Types</option>
                {BED_TYPES.map(type => <option key={type} value={type}>{type}</option>)}
              </select>

              <select 
                value={filter.status} 
                onChange={(e) => setFilter(prev => ({ ...prev, status: e.target.value }))} 
                style={{ 
                  padding: '10px 12px', 
                  borderRadius: '8px', 
                  border: '1px solid #e2e8f0', 
                  backgroundColor: '#fff', 
                  fontSize: '13px', 
                  cursor: 'pointer',
                  fontWeight: 500
                }}
              >
                <option value="all">All Status</option>
                {BED_STATUSES.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
              </select>
            </div>
          </div>
        </div>

        {/* Professional Beds Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: '20px', marginBottom: '32px' }}>
          {loading ? (
            <div style={{ gridColumn: '1 / -1', padding: '60px 40px', textAlign: 'center', color: '#94a3b8', backgroundColor: '#fff', borderRadius: '12px' }}>
              <div style={{ fontSize: '16px', fontWeight: 600, marginBottom: '8px' }}>Loading Beds...</div>
              <div style={{ fontSize: '13px' }}>Fetching real-time bed data from hospital database</div>
            </div>
          ) : filteredBeds.length === 0 ? (
            <div style={{ gridColumn: '1 / -1', padding: '60px 40px', textAlign: 'center', color: '#94a3b8', backgroundColor: '#fff', borderRadius: '12px' }}>
              <div style={{ fontSize: '16px', fontWeight: 600, marginBottom: '8px' }}>No Beds Found</div>
              <div style={{ fontSize: '13px' }}>Try selecting a different ward or adjusting filters</div>
            </div>
          ) : (
            filteredBeds.map(bed => {
              const statusBadge = getStatusBadge(bed.status);
              const isOccupied = bed.status === 'Occupied' && bed.occupant;
              
              return (
                <div key={bed._id} style={{ 
                  backgroundColor: '#fff', 
                  borderRadius: '12px', 
                  border: isOccupied ? `2px solid ${statusBadge.color}` : '1px solid #e2e8f0',
                  overflow: 'hidden',
                  boxShadow: isOccupied ? `0 4px 12px ${statusBadge.color}15` : '0 1px 3px rgba(0, 0, 0, 0.05)',
                  transition: 'all 0.3s ease',
                  display: 'flex',
                  flexDirection: 'column'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.boxShadow = `0 12px 24px rgba(0, 0, 0, 0.12)`;
                  e.currentTarget.style.transform = 'translateY(-4px)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.boxShadow = isOccupied ? `0 4px 12px ${statusBadge.color}15` : '0 1px 3px rgba(0, 0, 0, 0.05)';
                  e.currentTarget.style.transform = 'translateY(0)';
                }}>
                  
                  {/* Bed Number - Clinical Header */}
                  <div style={{ 
                    padding: '18px 20px', 
                    backgroundColor: statusBadge.bgColor, 
                    borderBottom: `2px solid ${statusBadge.color}`,
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'center'
                  }}>
                    <div>
                      <div style={{ fontSize: '11px', fontWeight: 700, color: statusBadge.color, textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '4px' }}>
                        Bed Unit
                      </div>
                      <h3 style={{ margin: 0, fontSize: '24px', fontWeight: 800, color: statusBadge.color }}>
                        {bed.bedNumber}
                      </h3>
                    </div>
                    <span style={{ 
                      padding: '8px 14px', 
                      borderRadius: '8px', 
                      fontSize: '12px', 
                      fontWeight: 700,
                      color: statusBadge.color, 
                      backgroundColor: statusBadge.bgColor,
                      border: `2px solid ${statusBadge.color}`
                    }}>
                      {statusBadge.label}
                    </span>
                  </div>

                  {/* Bed Info */}
                  <div style={{ padding: '16px 20px' }}>
                    <div style={{ display: 'flex', gap: '12px', marginBottom: '14px' }}>
                      <div style={{ 
                        padding: '6px 10px', 
                        backgroundColor: '#f8fafc', 
                        borderRadius: '6px', 
                        fontSize: '12px', 
                        fontWeight: 600, 
                        color: '#475569' 
                      }}>
                        {bed.bedType}
                      </div>
                      {bed.infectionControl !== 'Standard' && (
                        <div style={{ 
                          padding: '6px 10px', 
                          backgroundColor: '#fee2e2', 
                          borderRadius: '6px', 
                          fontSize: '11px', 
                          fontWeight: 700, 
                          color: '#991b1b',
                          textTransform: 'uppercase'
                        }}>
                          🔴 {bed.infectionControl}
                        </div>
                      )}
                    </div>

                    {/* Occupancy Info - Enhanced Professional Display */}
                    {isOccupied ? (
                      <div style={{ 
                        padding: '0',
                        marginBottom: '14px',
                        backgroundColor: '#f8fafc',
                        borderRadius: '8px',
                        overflow: 'hidden',
                        border: `2px solid ${statusBadge.color}`
                      }}>
                        {/* Patient Avatar Header */}
                        <div style={{
                          padding: '12px 16px',
                          backgroundColor: statusBadge.color,
                          color: '#fff',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '12px'
                        }}>
                          <div style={{
                            width: '44px',
                            height: '44px',
                            backgroundColor: 'rgba(255,255,255,0.3)',
                            borderRadius: '50%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: '20px',
                            fontWeight: 800,
                            border: '2px solid rgba(255,255,255,0.6)'
                          }}>
                            {bed.occupant.name?.charAt(0).toUpperCase() || 'P'}
                          </div>
                          <div style={{ flex: 1 }}>
                            <div style={{ fontSize: '13px', fontWeight: 700, opacity: 0.9 }}>PATIENT</div>
                            <div style={{ fontSize: '16px', fontWeight: 800, marginTop: '2px' }}>
                              {bed.occupant.name}
                            </div>
                          </div>
                          <div style={{
                            padding: '6px 10px',
                            backgroundColor: 'rgba(255,255,255,0.2)',
                            borderRadius: '6px',
                            fontSize: '11px',
                            fontWeight: 700,
                            textAlign: 'center'
                          }}>
                            {bed.acuityLevel || 'MODERATE'}
                          </div>
                        </div>

                        {/* Patient Details Grid */}
                        <div style={{ padding: '14px 16px' }}>
                          <div style={{
                            display: 'grid',
                            gridTemplateColumns: '1fr 1fr',
                            gap: '12px',
                            marginBottom: '12px'
                          }}>
                            {/* LOS Card */}
                            <div style={{
                              padding: '10px',
                              backgroundColor: '#f0f9ff',
                              borderRadius: '6px',
                              borderLeft: '3px solid #0369a1'
                            }}>
                              <div style={{ fontSize: '10px', fontWeight: 700, color: '#0c4a6e', textTransform: 'uppercase', marginBottom: '4px' }}>📊 Length of Stay</div>
                              <div style={{ fontSize: '18px', fontWeight: 800, color: '#0369a1' }}>
                                {calculateLOS(bed.occupant.admittedAt)}
                              </div>
                            </div>

                            {/* Admission Date Card */}
                            <div style={{
                              padding: '10px',
                              backgroundColor: '#fef3c7',
                              borderRadius: '6px',
                              borderLeft: '3px solid #d97706'
                            }}>
                              <div style={{ fontSize: '10px', fontWeight: 700, color: '#92400e', textTransform: 'uppercase', marginBottom: '4px' }}>📅 Admitted</div>
                              <div style={{ fontSize: '13px', fontWeight: 700, color: '#b45309' }}>
                                {formatDate(bed.occupant.admittedAt)}
                              </div>
                            </div>

                            {/* Expected Discharge Card */}
                            <div style={{
                              padding: '10px',
                              backgroundColor: '#fce7f3',
                              borderRadius: '6px',
                              borderLeft: '3px solid #be185d'
                            }}>
                              <div style={{ fontSize: '10px', fontWeight: 700, color: '#831843', textTransform: 'uppercase', marginBottom: '4px' }}>🩹 Expected DC</div>
                              <div style={{ fontSize: '13px', fontWeight: 700, color: '#be185d' }}>
                                {formatDate(bed.occupant.expectedDischargeDate) || 'TBD'}
                              </div>
                            </div>

                            {/* Patient ID Card */}
                            <div style={{
                              padding: '10px',
                              backgroundColor: '#e0e7ff',
                              borderRadius: '6px',
                              borderLeft: '3px solid #4f46e5'
                            }}>
                              <div style={{ fontSize: '10px', fontWeight: 700, color: '#3730a3', textTransform: 'uppercase', marginBottom: '4px' }}>🆔 Patient ID</div>
                              <div style={{ fontSize: '12px', fontWeight: 700, color: '#4f46e5', fontFamily: 'monospace' }}>
                                {bed.occupant.patientId}
                              </div>
                            </div>
                          </div>

                          {/* Notes Section */}
                          {bed.occupant.notes && (
                            <div style={{
                              padding: '10px',
                              backgroundColor: '#f5f3ff',
                              borderRadius: '6px',
                              borderLeft: '3px solid #7c3aed'
                            }}>
                              <div style={{ fontSize: '10px', fontWeight: 700, color: '#5b21b6', textTransform: 'uppercase', marginBottom: '4px' }}>📝 Notes</div>
                              <div style={{ fontSize: '12px', color: '#6b21a8', lineHeight: '1.4' }}>
                                {bed.occupant.notes}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    ) : (
                      <div style={{ 
                        padding: '20px', 
                        backgroundColor: '#ecfdf5', 
                        borderRadius: '8px', 
                        marginBottom: '14px',
                        borderLeft: '4px solid #10b981',
                        textAlign: 'center',
                        color: '#065f46'
                      }}>
                        <div style={{ fontSize: '28px', marginBottom: '8px' }}>🛏️</div>
                        <div style={{ fontSize: '14px', fontWeight: 700 }}>Ready for Admission</div>
                        <div style={{ fontSize: '12px', marginTop: '4px', opacity: 0.8 }}>Available for new patient</div>
                      </div>
                    )}

                    {/* Room Details - Extracted from JSON */}
                    <div style={{ marginBottom: '14px', padding: '12px 16px', backgroundColor: '#f0f9ff', borderRadius: '8px', border: '1px solid #bae6fd' }}>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                        {/* Room Number */}
                        <div>
                          <div style={{ fontSize: '10px', fontWeight: 700, color: '#0c4a6e', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '4px' }}>🚪 Room</div>
                          <div style={{ fontSize: '16px', fontWeight: 800, color: '#0369a1' }}>
                            {bed.roomNumber || 'N/A'}
                          </div>
                        </div>
                        {/* Bed ID/Reference */}
                        <div>
                          <div style={{ fontSize: '10px', fontWeight: 700, color: '#0c4a6e', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '4px' }}>🏷️ ID</div>
                          <div style={{ fontSize: '12px', fontWeight: 700, color: '#0369a1', fontFamily: 'monospace' }}>
                            {bed.bedId || 'N/A'}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Equipment & Features - Comprehensive Display */}
                    <div style={{ marginBottom: '14px' }}>
                      {/* Equipment Section */}
                      {bed.equipment?.length > 0 && (
                        <div style={{ marginBottom: '10px' }}>
                          <div style={{ fontSize: '10px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '8px' }}>⚙️ Equipment & Monitoring</div>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                            {bed.equipment.map((equip, idx) => {
                              const equipIcon = equip === 'Monitor' ? '📊' : equip === 'Ventilator' ? '🫁' : equip === 'Dialysis Machine' ? '🔄' : '⚙️';
                              return (
                                <div key={idx} style={{ 
                                  padding: '8px 12px', 
                                  backgroundColor: '#faf5ff', 
                                  color: '#7c3aed', 
                                  borderRadius: '6px', 
                                  fontSize: '12px', 
                                  fontWeight: 600,
                                  border: '1px solid #e9d5ff',
                                  display: 'flex',
                                  alignItems: 'center',
                                  gap: '6px'
                                }}>
                                  <span>{equipIcon}</span>
                                  <span>{equip}</span>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}

                      {/* Features & Characteristics */}
                      {(bed.characteristics?.length > 0 || bed.compatibility?.canIsolate) && (
                        <div style={{ marginTop: '10px' }}>
                          <div style={{ fontSize: '10px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '8px' }}>✨ Features & Capabilities</div>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                            {bed.characteristics?.filter((char, idx, arr) => arr.indexOf(char) === idx).map((char, idx) => (
                              <div key={idx} style={{ 
                                padding: '6px 10px', 
                                backgroundColor: '#dbeafe', 
                                color: '#0c4a6e', 
                                borderRadius: '4px', 
                                fontSize: '11px', 
                                fontWeight: 600,
                                border: '1px solid #bae6fd'
                              }}>
                                {char}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Bed Specifications - Extracted from JSON */}
                    <div style={{ marginBottom: '14px', padding: '12px 16px', backgroundColor: '#f8fafc', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                        {/* Bed Type */}
                        <div>
                          <div style={{ fontSize: '10px', fontWeight: 700, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '4px' }}>🛏️ Type</div>
                          <div style={{ fontSize: '14px', fontWeight: 700, color: '#334155' }}>
                            {bed.bedType || 'Standard'}
                          </div>
                        </div>
                        {/* Acuity Level */}
                        <div>
                          <div style={{ fontSize: '10px', fontWeight: 700, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '4px' }}>📈 Acuity</div>
                          <div style={{ fontSize: '14px', fontWeight: 700, color: '#334155' }}>
                            {bed.acuityLevel || 'MODERATE'}
                          </div>
                        </div>
                        {/* Infection Control */}
                        {bed.infectionControl && bed.infectionControl !== 'Standard' && (
                          <div>
                            <div style={{ fontSize: '10px', fontWeight: 700, color: '#7c2d12', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '4px' }}>⚠️ Infection</div>
                            <div style={{ fontSize: '12px', fontWeight: 700, color: '#b45309' }}>
                              {bed.infectionControl}
                            </div>
                          </div>
                        )}
                        {/* Last Cleaned */}
                        {bed.lastCleanedFormatted && (
                          <div>
                            <div style={{ fontSize: '10px', fontWeight: 700, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '4px' }}>🧹 Last Cleaned</div>
                            <div style={{ fontSize: '12px', fontWeight: 600, color: '#475569' }}>
                              {bed.lastCleanedFormatted}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Action Buttons - Clinical Footer */}
                  <div style={{ 
                    padding: '12px 16px', 
                    backgroundColor: '#f8fafc', 
                    borderTop: '1px solid #e2e8f0',
                    display: 'flex', 
                    gap: '8px',
                    marginTop: 'auto',
                    flexWrap: 'wrap'
                  }}>
                    {bed.status === 'Available' ? (
                      <>
                        <button 
                          onClick={() => {
                            console.log('📋 Opening assign modal for bed:', bed.bedNumber);
                            openAssignModal(bed);
                          }} 
                          style={{ 
                            flex: 1,
                            minWidth: '120px',
                            padding: '12px 14px', 
                            borderRadius: '6px', 
                            backgroundColor: '#10b981', 
                            color: '#fff', 
                            border: 'none', 
                            cursor: 'pointer', 
                            fontWeight: 700, 
                            fontSize: '13px',
                            transition: 'all 0.2s',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            gap: '6px',
                            boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.backgroundColor = '#059669';
                            e.currentTarget.style.transform = 'scale(1.02)';
                            e.currentTarget.style.boxShadow = '0 4px 8px rgba(16,185,129,0.3)';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.backgroundColor = '#10b981';
                            e.currentTarget.style.transform = 'scale(1)';
                            e.currentTarget.style.boxShadow = '0 1px 3px rgba(0,0,0,0.1)';
                          }}
                        >
                          <span>➕</span> Admit
                        </button>
                      </>
                    ) : isOccupied ? (
                      <>
                        <button 
                          onClick={() => {
                            console.log('📋 Viewing patient details for:', bed.occupant.name);
                          }} 
                          style={{ 
                            flex: 1,
                            minWidth: '100px',
                            padding: '12px 14px', 
                            borderRadius: '6px', 
                            backgroundColor: '#0369a1', 
                            color: '#fff', 
                            border: 'none', 
                            cursor: 'pointer', 
                            fontWeight: 700, 
                            fontSize: '13px',
                            transition: 'all 0.2s',
                            boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.backgroundColor = '#0284c7';
                            e.currentTarget.style.transform = 'scale(1.02)';
                            e.currentTarget.style.boxShadow = '0 4px 8px rgba(3,105,161,0.3)';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.backgroundColor = '#0369a1';
                            e.currentTarget.style.transform = 'scale(1)';
                            e.currentTarget.style.boxShadow = '0 1px 3px rgba(0,0,0,0.1)';
                          }}
                        >
                          👤 View
                        </button>
                        <button 
                          onClick={() => {
                            console.log('📤 Unassigning patient from bed:', bed.bedNumber);
                            unassignPatient(bed);
                          }} 
                          style={{ 
                            flex: 1,
                            minWidth: '100px',
                            padding: '12px 14px', 
                            borderRadius: '6px', 
                            backgroundColor: '#dc2626', 
                            color: '#fff', 
                            border: 'none', 
                            cursor: 'pointer', 
                            fontWeight: 700, 
                            fontSize: '13px',
                            transition: 'all 0.2s',
                            boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.backgroundColor = '#b91c1c';
                            e.currentTarget.style.transform = 'scale(1.02)';
                            e.currentTarget.style.boxShadow = '0 4px 8px rgba(220,38,38,0.3)';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.backgroundColor = '#dc2626';
                            e.currentTarget.style.transform = 'scale(1)';
                            e.currentTarget.style.boxShadow = '0 1px 3px rgba(0,0,0,0.1)';
                          }}
                        >
                          🏥 Discharge
                        </button>
                      </>
                    ) : null}
                    <button 
                      onClick={() => {
                        console.log('✏️  Editing bed:', bed.bedNumber);
                        openEditModal(bed);
                      }} 
                      style={{ 
                        padding: '10px 12px', 
                        backgroundColor: '#e0e7ff', 
                        color: '#3730a3', 
                        border: '1px solid #c7d2fe',
                        borderRadius: '6px', 
                        cursor: 'pointer', 
                        fontSize: '12px', 
                        fontWeight: 600,
                        transition: 'all 0.2s',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px'
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.backgroundColor = '#c7d2fe';
                        e.currentTarget.style.boxShadow = '0 2px 4px rgba(55,48,163,0.2)';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor = '#e0e7ff';
                        e.currentTarget.style.boxShadow = 'none';
                      }}
                    >
                      <span>✏️</span> Edit
                    </button>
                    <button 
                      onClick={() => {
                        console.log('🗑️  Deleting bed:', bed.bedNumber);
                        setDeleteTarget(bed); 
                        setShowDeleteConfirm(true); 
                      }} 
                      style={{ 
                        padding: '10px 12px', 
                        backgroundColor: '#fee2e2', 
                        color: '#991b1b', 
                        border: '1px solid #fecaca',
                        borderRadius: '6px', 
                        cursor: 'pointer', 
                        fontSize: '12px', 
                        fontWeight: 600,
                        transition: 'all 0.2s',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px'
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.backgroundColor = '#fecaca';
                        e.currentTarget.style.boxShadow = '0 2px 4px rgba(153,27,27,0.2)';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor = '#fee2e2';
                        e.currentTarget.style.boxShadow = 'none';
                      }}
                    >
                      <span>🗑️</span>
                    </button>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </SidebarLayout>
  );
};

export default BedsAllocation;
