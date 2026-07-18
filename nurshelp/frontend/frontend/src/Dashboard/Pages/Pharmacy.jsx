import React, { useState, useEffect } from 'react';
import { FiSearch, FiPlus, FiMinus, FiAlertCircle, FiRefreshCw, FiFilter, FiEye, FiSettings, FiTrendingUp } from 'react-icons/fi';
import { FaPills } from 'react-icons/fa';
import SidebarLayout from '../SidebarLayout';
import { decodeJwt } from '../../utils/jwt';

const Pharmacy = () => {
  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";
  
  const [inventory, setInventory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [filterSchedule, setFilterSchedule] = useState('all');
  const [selectedMed, setSelectedMed] = useState(null);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [showHandoverModal, setShowHandoverModal] = useState(false);
  const [handoverPatient, setHandoverPatient] = useState('');
  const [handoverQuantity, setHandoverQuantity] = useState(1);
  const [successMessage, setSuccessMessage] = useState('');
  const [userInfo, setUserInfo] = useState(null);

  // Dispense Modal State
  const [showDispenseModal, setShowDispenseModal] = useState(false);
  const [dispenseForm, setDispenseForm] = useState({
    pharmacistName: '',
    pharmacistNotes: '',
    frequency: 'once daily',
    scheduledTimes: ['08:00'],
    duration: '7 days',
    dosagePerTime: '1',
    instructions: '',
    prescribedByName: '',
    prescribedByRole: 'PHYSICIAN',
    patientId: '',
    patientConditionBefore: '',
    scheduleLevel: 'OTC'
  });

  // Dispensing History Modal State
  const [showDispensingHistoryModal, setShowDispensingHistoryModal] = useState(false);
  const [dispensingHistory, setDispensingHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  // Add Medication Modal State
  const [showAddMedModal, setShowAddMedModal] = useState(false);
  const [newMedForm, setNewMedForm] = useState({
    name: '',
    strength: '',
    form: 'Tablet',
    route: 'PO',
    stock: '',
    unitPrice: '',
    minStockLevel: '',
    maxStockLevel: '',
    scheduleLevel: 'OTC'
  });

  // Refill Modal State
  const [showRefillModal, setShowRefillModal] = useState(false);
  const [refillForm, setRefillForm] = useState({
    quantity: '',
    batchNumber: '',
    expiryDate: '',
    manufacturer: ''
  });

  // Administration Recording Modal State
  const [showAdminModal, setShowAdminModal] = useState(false);
  const [adminForm, setAdminForm] = useState({
    staffName: '',
    staffRole: 'NURSE',
    administeredTime: '',
    dosageGiven: '',
    routeUsed: 'PO',
    patientConditionAfter: '',
    sideEffectsObserved: ''
  });

  // CSV Upload State
  const [showCSVUploadModal, setShowCSVUploadModal] = useState(false);
  const [csvFile, setCSVFile] = useState(null);
  const [csvUploadLoading, setCSVUploadLoading] = useState(false);
  const [csvUploadResult, setCSVUploadResult] = useState(null);
  const [csvExportLoading, setCSVExportLoading] = useState(false);

  // Mock pharmacy inventory data
  const mockInventory = [
    {
      id: 'MED-001',
      name: 'Amoxicillin 500mg',
      category: 'Antibiotic',
      quantity: 120,
      minStock: 50,
      unit: 'tablets',
      expiry: '2025-12-31',
      location: 'Shelf A1',
      manufacturer: 'Pharma Corp',
      batchNumber: 'BATCH-2024-001',
      schedule: 'V'
    },
    {
      id: 'MED-002',
      name: 'Ibuprofen 200mg',
      category: 'Pain Relief',
      quantity: 85,
      minStock: 100,
      unit: 'tablets',
      expiry: '2025-11-15',
      location: 'Shelf B2',
      manufacturer: 'Pain Relief Inc',
      batchNumber: 'BATCH-2024-002',
      schedule: 'IV'
    },
    {
      id: 'MED-003',
      name: 'Metformin 500mg',
      category: 'Diabetes',
      quantity: 30,
      minStock: 75,
      unit: 'tablets',
      expiry: '2025-08-20',
      location: 'Shelf C3',
      manufacturer: 'Diabetes Pharma',
      batchNumber: 'BATCH-2024-003',
      schedule: 'V'
    },
    {
      id: 'MED-004',
      name: 'Lisinopril 10mg',
      category: 'Cardiovascular',
      quantity: 45,
      minStock: 40,
      unit: 'tablets',
      expiry: '2026-03-10',
      location: 'Shelf D1',
      manufacturer: 'Heart Health',
      batchNumber: 'BATCH-2024-004',
      schedule: 'IV'
    },
    {
      id: 'MED-005',
      name: 'Omeprazole 20mg',
      category: 'GI',
      quantity: 200,
      minStock: 80,
      unit: 'capsules',
      expiry: '2026-05-22',
      location: 'Shelf E2',
      manufacturer: 'GI Solutions',
      batchNumber: 'BATCH-2024-005',
      schedule: 'IV'
    },
    {
      id: 'MED-006',
      name: 'Diphenhydramine 25mg',
      category: 'Antihistamine',
      quantity: 15,
      minStock: 60,
      unit: 'tablets',
      expiry: '2024-09-15',
      location: 'Shelf F1',
      manufacturer: 'Allergy Relief',
      batchNumber: 'BATCH-2023-006',
      schedule: 'IV'
    },
    {
      id: 'MED-007',
      name: 'Cetirizine 10mg',
      category: 'Antihistamine',
      quantity: 95,
      minStock: 50,
      unit: 'tablets',
      expiry: '2025-10-30',
      location: 'Shelf F2',
      manufacturer: 'Allergy Relief',
      batchNumber: 'BATCH-2024-007',
      schedule: 'V'
    },
    {
      id: 'MED-008',
      name: 'Acetaminophen 500mg',
      category: 'Pain Relief',
      quantity: 5,
      minStock: 100,
      unit: 'tablets',
      expiry: '2025-07-11',
      location: 'Shelf B3',
      manufacturer: 'Pain Relief Inc',
      batchNumber: 'BATCH-2024-008',
      schedule: 'IV'
    }
  ];

  useEffect(() => {
    // Extract user info from JWT token
    const token = localStorage.getItem("token");
    if (token) {
      const decoded = decodeJwt(token);
      if (decoded) {
        setUserInfo(decoded);
        console.log('User hospital context:', decoded.hospitalId);
      }
    }
    
    fetchInventory();
  }, []);

  const fetchInventory = async () => {
    setLoading(true);
    setError('');
    try {
      const token = localStorage.getItem('token');
      if (!token) {
        throw new Error('No authentication token found');
      }

      const decoded = decodeJwt(token);
      if (!decoded || !decoded.hospitalId) {
        throw new Error('Cannot determine hospital context');
      }

      // Fetch pharmacy inventory for the specific hospital
      const response = await fetch(
        `${BACKEND_URL}/api/pharmacy/inventory?hospitalId=${decoded.hospitalId}`,
        {
          headers: { 'Authorization': `Bearer ${token}` }
        }
      );
      
      if (response.ok) {
        const data = await response.json();
        console.log('Fetched pharmacy inventory from backend:', data);
        
        // Extract the data array from the response object
        const inventoryArray = data.data || data || [];
        
        // Transform backend data to match expected format
        const transformedData = Array.isArray(inventoryArray) ? inventoryArray.map(inv => ({
          id: inv._id || inv.medicationId,
          name: inv.medicationName || inv.name,
          category: inv.category || 'General',
          quantity: inv.quantity || 0,
          minStock: inv.minStockLevel || 10,
          unit: inv.unit || 'units',
          expiry: inv.expiryDate || '2026-12-31',
          location: inv.location || 'Pharmacy Main',
          manufacturer: inv.manufacturer || 'Unknown',
          batchNumber: inv.batchNumber || 'N/A',
          unitPrice: inv.unitPrice || 0,
          status: inv.status || 'in-stock',
          schedule: inv.schedule || 'Unscheduled'
        })) : [];
        
        console.log(`Transformed ${transformedData.length} medications for display`);
        setInventory(transformedData);
      } else {
        setError('Failed to fetch pharmacy inventory');
        console.log('Using mock pharmacy inventory data as fallback');
        setInventory(mockInventory);
      }
    } catch (err) {
      console.error('Error fetching pharmacy inventory:', err);
      setError(`Error loading pharmacy inventory: ${err.message}`);
      setInventory(mockInventory);
    }
    setLoading(false);
  };

  const getStockStatus = (quantity, minStock) => {
    if (quantity === 0) return { status: 'out', label: 'Out of Stock', color: '#ef4444' };
    if (quantity < minStock) return { status: 'low', label: 'Low Stock', color: '#f59e0b' };
    return { status: 'ok', label: 'In Stock', color: '#10b981' };
  };

  const isExpiringSoon = (expiry) => {
    const expiryDate = new Date(expiry);
    const today = new Date();
    const daysUntilExpiry = Math.floor((expiryDate - today) / (1000 * 60 * 60 * 24));
    return daysUntilExpiry <= 30 && daysUntilExpiry >= 0;
  };

  const isExpired = (expiry) => {
    const expiryDate = new Date(expiry);
    return expiryDate < new Date();
  };

  const filteredInventory = inventory.filter(med => {
    const matchesSearch = med.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         med.category.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         med.id.toLowerCase().includes(searchTerm.toLowerCase());
    
    let matchesStatus = true;
    if (filterStatus === 'low') {
      matchesStatus = med.quantity < med.minStock;
    } else if (filterStatus === 'expired') {
      matchesStatus = isExpired(med.expiry);
    } else if (filterStatus === 'expiring') {
      matchesStatus = isExpiringSoon(med.expiry) && !isExpired(med.expiry);
    }

    let matchesSchedule = true;
    if (filterSchedule !== 'all') {
      matchesSchedule = med.schedule === filterSchedule;
    }

    return matchesSearch && matchesStatus && matchesSchedule;
  });

  const handleHandoverMedicine = async () => {
    if (!selectedMed || !handoverPatient || handoverQuantity < 1) {
      setError('Please select a patient and quantity');
      setTimeout(() => setError(''), 3000);
      return;
    }

    try {
      const token = localStorage.getItem('token');
      const decoded = decodeJwt(token);
      
      const response = await fetch(`${BACKEND_URL}/api/pharmacy/handover`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          hospitalId: decoded?.hospitalId,
          medicineId: selectedMed.id,
          patientId: handoverPatient,
          quantity: handoverQuantity,
          timestamp: new Date()
        })
      });

      if (response.ok) {
        setSuccessMessage(`Handed over ${handoverQuantity} of ${selectedMed.name} to patient ${handoverPatient}`);
        setTimeout(() => setSuccessMessage(''), 3000);
        setShowHandoverModal(false);
        setHandoverPatient('');
        setHandoverQuantity(1);
        
        // Update inventory
        setInventory(inv => inv.map(med => 
          med.id === selectedMed.id 
            ? { ...med, quantity: med.quantity - handoverQuantity }
            : med
        ));
      }
    } catch (err) {
      setError('Failed to process handover: ' + err.message);
      setTimeout(() => setError(''), 3000);
    }
  };

  // Dispense Medication Handler
  const handleDispenseMedication = async () => {
    if (!selectedMed || !dispenseForm.pharmacistName || !dispenseForm.patientId || !dispenseForm.prescribedByName) {
      setError('Please fill in all required fields');
      setTimeout(() => setError(''), 3000);
      return;
    }

    try {
      const token = localStorage.getItem('token');
      const decoded = decodeJwt(token);
      
      const response = await fetch(`${BACKEND_URL}/api/pharmacy/dispense`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          hospitalId: decoded?.hospitalId,
          medicationId: selectedMed.id,
          medicationName: selectedMed.name,
          patientId: dispenseForm.patientId,
          pharmacistName: dispenseForm.pharmacistName,
          pharmacistNotes: dispenseForm.pharmacistNotes,
          prescribedByName: dispenseForm.prescribedByName,
          prescribedByRole: dispenseForm.prescribedByRole,
          frequency: dispenseForm.frequency,
          scheduledTimes: dispenseForm.scheduledTimes,
          duration: dispenseForm.duration,
          dosagePerTime: dispenseForm.dosagePerTime,
          instructions: dispenseForm.instructions,
          patientConditionBefore: dispenseForm.patientConditionBefore,
          scheduleLevel: dispenseForm.scheduleLevel
        })
      });

      if (response.ok) {
        const result = await response.json();
        console.log('[DISPENSE SUCCESS]', result.data);
        setSuccessMessage(`✓ Medication dispensed successfully for patient ${dispenseForm.patientId}. Schedule Level: ${result.data.scheduleLevel}`);
        setTimeout(() => setSuccessMessage(''), 5000);
        setShowDispenseModal(false);
        setDispenseForm({
          pharmacistName: '',
          pharmacistNotes: '',
          frequency: 'once daily',
          scheduledTimes: ['08:00'],
          duration: '7 days',
          dosagePerTime: '1',
          instructions: '',
          prescribedByName: '',
          prescribedByRole: 'PHYSICIAN',
          patientId: '',
          patientConditionBefore: '',
          scheduleLevel: 'OTC'
        });
        // Refresh inventory
        await fetchInventory();
      } else {
        const result = await response.json();
        setError(`Failed to dispense: ${result.message || 'Unknown error'}`);
        setTimeout(() => setError(''), 3000);
      }
    } catch (err) {
      setError('Failed to dispense medication: ' + err.message);
      setTimeout(() => setError(''), 3000);
    }
  };

  // Fetch Dispensing History for a Medication
  const fetchDispensingHistory = async () => {
    if (!selectedMed) return;
    
    setHistoryLoading(true);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${BACKEND_URL}/api/pharmacy/dispensing-history/${selectedMed.id}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const result = await response.json();
        setDispensingHistory(result.data || []);
        setShowDispensingHistoryModal(true);
        console.log('[DISPENSING HISTORY]', result.data);
      } else {
        setError('Failed to fetch dispensing history');
        setTimeout(() => setError(''), 3000);
      }
    } catch (err) {
      setError('Error fetching dispensing history: ' + err.message);
      setTimeout(() => setError(''), 3000);
    }
    setHistoryLoading(false);
  };

  // CSV Upload Handler
  const handleCSVUpload = async () => {
    if (!csvFile) {
      setError('Please select a CSV file');
      setTimeout(() => setError(''), 3000);
      return;
    }

    setCSVUploadLoading(true);
    try {
      const token = localStorage.getItem('token');
      const formData = new FormData();
      formData.append('file', csvFile);

      const response = await fetch(`${BACKEND_URL}/api/pharmacy/upload-csv`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        setCSVUploadResult(result.data);
        setSuccessMessage(`✓ CSV processed: ${result.data.summary.successCount} successful, ${result.data.summary.failureCount} failed`);
        setTimeout(() => setSuccessMessage(''), 5000);
        setCSVFile(null);
        setTimeout(() => setShowCSVUploadModal(false), 2000);
        await fetchInventory(); // Refresh inventory
        console.log('[CSV UPLOAD SUCCESS]', result.data);
      } else {
        const result = await response.json();
        setError(`Failed to upload CSV: ${result.message || 'Unknown error'}`);
        setTimeout(() => setError(''), 3000);
      }
    } catch (err) {
      setError('Error uploading CSV: ' + err.message);
      setTimeout(() => setError(''), 3000);
    }
    setCSVUploadLoading(false);
  };

  // CSV Export Handler
  const handleCSVExport = async () => {
    setCSVExportLoading(true);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${BACKEND_URL}/api/pharmacy/export-csv`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        // Get the CSV content
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `inventory-${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
        
        setSuccessMessage('✓ Inventory exported successfully');
        setTimeout(() => setSuccessMessage(''), 3000);
        console.log('[CSV EXPORT SUCCESS] File downloaded');
      } else {
        const result = await response.json();
        setError(`Failed to export CSV: ${result.message || 'Unknown error'}`);
        setTimeout(() => setError(''), 3000);
      }
    } catch (err) {
      setError('Error exporting CSV: ' + err.message);
      setTimeout(() => setError(''), 3000);
    }
    setCSVExportLoading(false);
  };

  // Refill Medication Handler
  const handleRefillMedication = async () => {
    if (!selectedMed || !refillForm.quantity || !refillForm.expiryDate) {
      setError('Please fill in quantity and expiry date');
      setTimeout(() => setError(''), 3000);
      return;
    }

    try {
      const token = localStorage.getItem('token');
      const decoded = decodeJwt(token);
      
      const response = await fetch(`${BACKEND_URL}/api/pharmacy/refill`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          hospitalId: decoded?.hospitalId,
          medicationId: selectedMed.id,
          quantity: parseInt(refillForm.quantity),
          batchNumber: refillForm.batchNumber,
          expiryDate: refillForm.expiryDate,
          manufacturer: refillForm.manufacturer || selectedMed.manufacturer
        })
      });

      if (response.ok) {
        setSuccessMessage(`✓ Refilled ${refillForm.quantity} units of ${selectedMed.name}`);
        setTimeout(() => setSuccessMessage(''), 3000);
        setShowRefillModal(false);
        setRefillForm({
          quantity: '',
          batchNumber: '',
          expiryDate: '',
          manufacturer: ''
        });
        fetchInventory(); // Refresh inventory
      } else {
        setError('Failed to refill medication');
        setTimeout(() => setError(''), 3000);
      }
    } catch (err) {
      setError('Failed to refill: ' + err.message);
      setTimeout(() => setError(''), 3000);
    }
  };

  // Add Medication Handler
  const handleAddMedication = async () => {
    if (!newMedForm.name || !newMedForm.stock || !newMedForm.unitPrice) {
      setError('Please fill in name, stock, and price');
      setTimeout(() => setError(''), 3000);
      return;
    }

    try {
      const token = localStorage.getItem('token');
      const decoded = decodeJwt(token);
      
      const response = await fetch(`${BACKEND_URL}/api/pharmacy/add-medication`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          hospitalId: decoded?.hospitalId,
          medicationName: newMedForm.name,
          strength: newMedForm.strength,
          form: newMedForm.form,
          route: newMedForm.route,
          quantity: parseInt(newMedForm.stock),
          unitPrice: parseFloat(newMedForm.unitPrice),
          minStockLevel: parseInt(newMedForm.minStockLevel) || 20,
          maxStockLevel: parseInt(newMedForm.maxStockLevel) || 200,
          schedule: newMedForm.schedule,
          category: 'General',
          manufacturer: 'New Stock'
        })
      });

      if (response.ok) {
        setSuccessMessage(`✓ Added new medication: ${newMedForm.name}`);
        setTimeout(() => setSuccessMessage(''), 3000);
        setShowAddMedModal(false);
        setNewMedForm({
          name: '',
          strength: '',
          form: 'Tablet',
          route: 'PO',
          stock: '',
          unitPrice: '',
          minStockLevel: '',
          maxStockLevel: '',
          schedule: 'V'
        });
        fetchInventory(); // Refresh inventory
      } else {
        setError('Failed to add medication');
        setTimeout(() => setError(''), 3000);
      }
    } catch (err) {
      setError('Failed to add medication: ' + err.message);
      setTimeout(() => setError(''), 3000);
    }
  };

  // Record Administration Handler
  const handleRecordAdministration = async () => {
    if (!selectedMed || !adminForm.staffName || !adminForm.administeredTime || !adminForm.dosageGiven) {
      setError('Please fill in staff name, time, and dosage');
      setTimeout(() => setError(''), 3000);
      return;
    }

    try {
      const token = localStorage.getItem('token');
      const decoded = decodeJwt(token);
      
      const response = await fetch(`${BACKEND_URL}/api/pharmacy/administration`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          hospitalId: decoded?.hospitalId,
          medicationId: selectedMed.id,
          staffName: adminForm.staffName,
          staffRole: adminForm.staffRole,
          administeredTime: adminForm.administeredTime,
          dosageGiven: adminForm.dosageGiven,
          routeUsed: adminForm.routeUsed,
          patientConditionAfter: adminForm.patientConditionAfter,
          sideEffectsObserved: adminForm.sideEffectsObserved
        })
      });

      if (response.ok) {
        setSuccessMessage(`✓ Administration recorded for ${adminForm.staffName}`);
        setTimeout(() => setSuccessMessage(''), 3000);
        setShowAdminModal(false);
        setAdminForm({
          staffName: '',
          staffRole: 'NURSE',
          administeredTime: '',
          dosageGiven: '',
          routeUsed: 'PO',
          patientConditionAfter: '',
          sideEffectsObserved: ''
        });
      } else {
        setError('Failed to record administration');
        setTimeout(() => setError(''), 3000);
      }
    } catch (err) {
      setError('Failed to record administration: ' + err.message);
      setTimeout(() => setError(''), 3000);
    }
  };

  return (
    <SidebarLayout>
      <div style={{ padding: '32px', backgroundColor: '#f0f4f8', minHeight: '100vh' }}>
        
        {/* HEADER SECTION */}
        <div style={{
          marginBottom: '32px',
          backgroundColor: '#fff',
          borderRadius: '12px',
          padding: '24px',
          border: '1px solid #e2e8f0',
          boxShadow: '0 1px 3px rgba(0,0,0,0.08)'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                <div style={{
                  width: '40px',
                  height: '40px',
                  backgroundColor: '#ecfdf5',
                  borderRadius: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <FaPills size={24} style={{ color: '#059669' }} />
                </div>
                <div>
                  <h1 style={{ margin: 0, fontSize: '28px', fontWeight: '700', color: '#0f172a' }}>
                    Pharmaceutical Inventory Management
                  </h1>
                  <p style={{ margin: '4px 0 0 0', color: '#64748b', fontSize: '13px', fontWeight: '500' }}>
                    Centralized medication tracking, dispensing, and audit compliance
                  </p>
                </div>
              </div>
            </div>
            
            {/* Quick Stats */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              gap: '16px',
              minWidth: '300px'
            }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '12px', color: '#64748b', fontWeight: '500', marginBottom: '4px' }}>
                  Total Items
                </div>
                <div style={{ fontSize: '24px', fontWeight: '700', color: '#059669' }}>
                  {inventory.length}
                </div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '12px', color: '#64748b', fontWeight: '500', marginBottom: '4px' }}>
                  Low Stock
                </div>
                <div style={{ fontSize: '24px', fontWeight: '700', color: '#f59e0b' }}>
                  {inventory.filter(m => m.quantity < m.minStock).length}
                </div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '12px', color: '#64748b', fontWeight: '500', marginBottom: '4px' }}>
                  Total Value
                </div>
                <div style={{ fontSize: '24px', fontWeight: '700', color: '#0284c7' }}>
                  R {(inventory.reduce((sum, m) => sum + (m.quantity * (m.unitPrice || 0)), 0)).toLocaleString('en-ZA', {maximumFractionDigits: 2})}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* STATUS MESSAGES */}
        {error && (
          <div style={{
            backgroundColor: '#fef2f2',
            color: '#7f1d1d',
            padding: '14px 16px',
            borderRadius: '8px',
            marginBottom: '16px',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            border: '1px solid #fecaca',
            fontSize: '13px',
            fontWeight: '500'
          }}>
            <FiAlertCircle size={18} style={{ flexShrink: 0 }} />
            {error}
          </div>
        )}

        {successMessage && (
          <div style={{
            backgroundColor: '#f0fdf4',
            color: '#166534',
            padding: '14px 16px',
            borderRadius: '8px',
            marginBottom: '16px',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            border: '1px solid #bbf7d0',
            fontSize: '13px',
            fontWeight: '500'
          }}>
            <div style={{ color: '#059669', fontWeight: '700' }}>✓</div>
            {successMessage}
          </div>
        )}

        {/* TOOLBAR - ORGANIZED SECTIONS */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr auto',
          gap: '16px',
          marginBottom: '24px',
          alignItems: 'start'
        }}>
          
          {/* LEFT: Search & Filters */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'auto 1fr 1fr 1fr',
            gap: '12px',
            alignItems: 'center'
          }}>
            {/* Search Input */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              backgroundColor: '#fff',
              border: '1px solid #e2e8f0',
              borderRadius: '8px',
              padding: '10px 12px',
              gridColumn: '1 / -1'
            }}>
              <FiSearch style={{ color: '#94a3b8', marginRight: '8px', flexShrink: 0 }} size={18} />
              <input
                type="text"
                placeholder="Search by medication name or batch..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                style={{
                  flex: 1,
                  border: 'none',
                  outline: 'none',
                  fontSize: '14px',
                  color: '#0f172a'
                }}
              />
            </div>

            {/* Filter: Status */}
            <div>
              <label style={{ display: 'block', fontSize: '11px', color: '#64748b', fontWeight: '600', marginBottom: '5px', textTransform: 'uppercase' }}>
                Status
              </label>
              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                style={{
                  width: '100%',
                  padding: '9px 11px',
                  border: '1px solid #e2e8f0',
                  borderRadius: '6px',
                  fontSize: '13px',
                  color: '#0f172a',
                  backgroundColor: '#fff',
                  cursor: 'pointer',
                  fontWeight: '500'
                }}
              >
                <option value="all">All Items</option>
                <option value="low">Low Stock</option>
                <option value="expired">Expired</option>
                <option value="expiring">Expiring Soon</option>
              </select>
            </div>

            {/* Filter: Schedule */}
            <div>
              <label style={{ display: 'block', fontSize: '11px', color: '#64748b', fontWeight: '600', marginBottom: '5px', textTransform: 'uppercase' }}>
                Schedule
              </label>
              <select
                value={filterSchedule}
                onChange={(e) => setFilterSchedule(e.target.value)}
                style={{
                  width: '100%',
                  padding: '9px 11px',
                  border: '1px solid #e2e8f0',
                  borderRadius: '6px',
                  fontSize: '13px',
                  color: '#0f172a',
                  backgroundColor: '#fff',
                  cursor: 'pointer',
                  fontWeight: '500'
                }}
              >
                <option value="all">All Schedules</option>
                <option value="I">Schedule I</option>
                <option value="II">Schedule II</option>
                <option value="III">Schedule III</option>
                <option value="IV">Schedule IV</option>
                <option value="V">Schedule V</option>
                <option value="OTC">OTC</option>
              </select>
            </div>

            {/* Filter: Category */}
            <div>
              <label style={{ display: 'block', fontSize: '11px', color: '#64748b', fontWeight: '600', marginBottom: '5px', textTransform: 'uppercase' }}>
                Records
              </label>
              <button
                onClick={fetchInventory}
                style={{
                  width: '100%',
                  padding: '9px 11px',
                  backgroundColor: '#f1f5f9',
                  border: '1px solid #cbd5e1',
                  borderRadius: '6px',
                  fontSize: '13px',
                  fontWeight: '600',
                  color: '#475569',
                  cursor: 'pointer',
                  transition: 'all 150ms ease',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '6px'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = '#e2e8f0';
                  e.currentTarget.style.borderColor = '#94a3b8';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = '#f1f5f9';
                  e.currentTarget.style.borderColor = '#cbd5e1';
                }}
              >
                <FiRefreshCw size={14} />
                Refresh
              </button>
            </div>
          </div>

          {/* RIGHT: Action Buttons */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: '10px',
            minWidth: '340px'
          }}>
            {/* Add Medication Button */}
            <button
              onClick={() => setShowAddMedModal(true)}
              style={{
                padding: '10px 14px',
                backgroundColor: '#059669',
                color: '#fff',
                border: 'none',
                borderRadius: '6px',
                fontSize: '13px',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 150ms ease',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '6px'
              }}
              onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#047857'; }}
              onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#059669'; }}
            >
              <FiPlus size={16} />
              Add Med
            </button>

            {/* Upload CSV */}
            <button
              onClick={() => setShowCSVUploadModal(true)}
              style={{
                padding: '10px 14px',
                backgroundColor: '#0284c7',
                color: '#fff',
                border: 'none',
                borderRadius: '6px',
                fontSize: '13px',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 150ms ease',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '6px'
              }}
              onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#0369a1'; }}
              onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#0284c7'; }}
            >
              <FiPlus size={16} />
              Bulk Upload
            </button>

            {/* Export CSV */}
            <button
              onClick={handleCSVExport}
              disabled={csvExportLoading}
              style={{
                padding: '10px 14px',
                backgroundColor: csvExportLoading ? '#cbd5e1' : '#7c3aed',
                color: '#fff',
                border: 'none',
                borderRadius: '6px',
                fontSize: '13px',
                fontWeight: '600',
                cursor: csvExportLoading ? 'not-allowed' : 'pointer',
                transition: 'all 150ms ease',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '6px',
                opacity: csvExportLoading ? 0.7 : 1
              }}
              onMouseEnter={(e) => {
                if (!csvExportLoading) e.currentTarget.style.backgroundColor = '#6d28d9';
              }}
              onMouseLeave={(e) => {
                if (!csvExportLoading) e.currentTarget.style.backgroundColor = '#7c3aed';
              }}
            >
              <FiMinus size={16} />
              {csvExportLoading ? 'Exporting...' : 'Export'}
            </button>
          </div>
        </div>

          {/* Schedule Filter */}
          <select
            value={filterSchedule}
            onChange={(e) => setFilterSchedule(e.target.value)}
            style={{
              padding: '8px 12px',
              border: '1px solid #e2e8f0',
              borderRadius: '8px',
              fontSize: '14px',
              backgroundColor: '#fff',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '4px'
            }}
          >
            <option value="all">All Schedules</option>
            <option value="I">Schedule I (Highest Risk)</option>
            <option value="II">Schedule II (High Risk)</option>
            <option value="III">Schedule III (Moderate Risk)</option>
            <option value="IV">Schedule IV (Lower Risk)</option>
            <option value="V">Schedule V (Lowest Risk)</option>
          </select>

        {/* INVENTORY TABLE */}
        {loading ? (
          <div style={{ textAlign: 'center', padding: '48px', color: '#94a3b8' }}>
            Loading inventory...
          </div>
        ) : (
          <div style={{ backgroundColor: '#fff', borderRadius: '12px', border: '1px solid #e2e8f0', overflow: 'hidden' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ backgroundColor: '#f8fafc', borderBottom: '1px solid #e2e8f0' }}>
                  <th style={{ padding: '12px 16px', textAlign: 'left', fontSize: '12px', fontWeight: '600', color: '#64748b' }}>Medication</th>
                  <th style={{ padding: '12px 16px', textAlign: 'left', fontSize: '12px', fontWeight: '600', color: '#64748b' }}>Category</th>
                  <th style={{ padding: '12px 16px', textAlign: 'center', fontSize: '12px', fontWeight: '600', color: '#64748b' }}>Quantity</th>
                  <th style={{ padding: '12px 16px', textAlign: 'center', fontSize: '12px', fontWeight: '600', color: '#64748b' }}>Status</th>
                  <th style={{ padding: '12px 16px', textAlign: 'center', fontSize: '12px', fontWeight: '600', color: '#64748b' }}>Schedule</th>
                  <th style={{ padding: '12px 16px', textAlign: 'center', fontSize: '12px', fontWeight: '600', color: '#64748b' }}>Expiry</th>
                  <th style={{ padding: '12px 16px', textAlign: 'center', fontSize: '12px', fontWeight: '600', color: '#64748b' }}>Location</th>
                  <th style={{ padding: '12px 16px', textAlign: 'center', fontSize: '12px', fontWeight: '600', color: '#64748b' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredInventory.length === 0 ? (
                  <tr>
                    <td colSpan="7" style={{ textAlign: 'center', padding: '32px', color: '#94a3b8' }}>
                      No medications found
                    </td>
                  </tr>
                ) : (
                  filteredInventory.map(med => {
                    const stockStatus = getStockStatus(med.quantity, med.minStock);
                    const expired = isExpired(med.expiry);
                    const expiringSoon = isExpiringSoon(med.expiry);

                    return (
                      <tr key={med.id} style={{ borderBottom: '1px solid #e2e8f0' }}>
                        <td style={{ padding: '12px 16px' }}>
                          <div style={{ fontWeight: '600', color: '#0f172a' }}>{med.name}</div>
                          <div style={{ fontSize: '12px', color: '#94a3b8' }}>{med.id}</div>
                        </td>
                        <td style={{ padding: '12px 16px', color: '#64748b' }}>{med.category}</td>
                        <td style={{ padding: '12px 16px', textAlign: 'center' }}>
                          <div style={{ fontWeight: '600', fontSize: '16px', color: '#0f172a' }}>
                            {med.quantity}
                          </div>
                          <div style={{ fontSize: '12px', color: '#94a3b8' }}>of {med.minStock} {med.unit}</div>
                        </td>
                        <td style={{ padding: '12px 16px', textAlign: 'center' }}>
                          <span style={{
                            backgroundColor: stockStatus.color + '20',
                            color: stockStatus.color,
                            padding: '4px 8px',
                            borderRadius: '4px',
                            fontSize: '12px',
                            fontWeight: '600'
                          }}>
                            {stockStatus.label}
                          </span>
                        </td>
                        <td style={{ padding: '12px 16px', textAlign: 'center', color: '#64748b', fontSize: '12px' }}>
                          {(() => {
                            let color = '#10b981';
                            let bgColor = '#ecfdf5';
                            if (med.schedule === 'I') {
                              color = '#ef4444';
                              bgColor = '#fee2e2';
                            } else if (med.schedule === 'II') {
                              color = '#dc2626';
                              bgColor = '#fecaca';
                            } else if (med.schedule === 'III') {
                              color = '#f97316';
                              bgColor = '#fed7aa';
                            } else if (med.schedule === 'IV') {
                              color = '#eab308';
                              bgColor = '#fef3c7';
                            }
                            return (
                              <span style={{
                                backgroundColor: bgColor,
                                color: color,
                                padding: '4px 8px',
                                borderRadius: '4px',
                                fontWeight: '600',
                                fontSize: '12px'
                              }}>
                                Schedule {med.schedule}
                              </span>
                            );
                          })()}
                        </td>
                        <td style={{ padding: '12px 16px', textAlign: 'center' }}>
                          <div style={{
                            color: expired ? '#ef4444' : expiringSoon ? '#f59e0b' : '#10b981',
                            fontWeight: '600',
                            fontSize: '12px'
                          }}>
                            {new Date(med.expiry).toLocaleDateString()}
                          </div>
                          {expired && <div style={{ fontSize: '11px', color: '#ef4444' }}>EXPIRED</div>}
                          {expiringSoon && !expired && <div style={{ fontSize: '11px', color: '#f59e0b' }}>Expiring</div>}
                        </td>
                        <td style={{ padding: '12px 16px', textAlign: 'center', color: '#64748b' }}>
                          {med.location}
                        </td>
                        <td style={{ padding: '12px 16px', textAlign: 'center' }}>
                          <div style={{ display: 'flex', gap: '8px', justifyContent: 'center', flexWrap: 'wrap' }}>
                            <button
                              onClick={() => {
                                setSelectedMed(med);
                                setShowDetailModal(true);
                              }}
                              style={{
                                padding: '8px 14px',
                                backgroundColor: '#ecfdf5',
                                color: '#10b981',
                                border: '1px solid #d1fae5',
                                borderRadius: '8px',
                                cursor: 'pointer',
                                transition: 'all 200ms ease',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '6px',
                                fontSize: '13px',
                                fontWeight: '600',
                                whiteSpace: 'nowrap'
                              }}
                              onMouseEnter={(e) => {
                                e.currentTarget.style.backgroundColor = '#d1fae5';
                                e.currentTarget.style.transform = 'translateY(-2px)';
                                e.currentTarget.style.boxShadow = '0 4px 12px rgba(16, 185, 129, 0.15)';
                              }}
                              onMouseLeave={(e) => {
                                e.currentTarget.style.backgroundColor = '#ecfdf5';
                                e.currentTarget.style.transform = 'translateY(0)';
                                e.currentTarget.style.boxShadow = 'none';
                              }}
                            >
                              <FiEye size={16} />
                              View
                            </button>
                            <button
                              onClick={() => {
                                setSelectedMed(med);
                                setShowDispenseModal(true);
                              }}
                              style={{
                                padding: '8px 14px',
                                backgroundColor: '#dbeafe',
                                color: '#0284c7',
                                border: '1px solid #bfdbfe',
                                borderRadius: '8px',
                                cursor: 'pointer',
                                transition: 'all 200ms ease',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '6px',
                                fontSize: '13px',
                                fontWeight: '600',
                                whiteSpace: 'nowrap'
                              }}
                              onMouseEnter={(e) => {
                                e.currentTarget.style.backgroundColor = '#93c5fd';
                                e.currentTarget.style.transform = 'translateY(-2px)';
                                e.currentTarget.style.boxShadow = '0 4px 12px rgba(2, 132, 199, 0.15)';
                              }}
                              onMouseLeave={(e) => {
                                e.currentTarget.style.backgroundColor = '#dbeafe';
                                e.currentTarget.style.transform = 'translateY(0)';
                                e.currentTarget.style.boxShadow = 'none';
                              }}
                            >
                              <FiTrendingUp size={16} />
                              Dispense
                            </button>
                            <button
                              onClick={() => {
                                setSelectedMed(med);
                                setShowRefillModal(true);
                              }}
                              style={{
                                padding: '8px 14px',
                                backgroundColor: '#fef3c7',
                                color: '#92400e',
                                border: '1px solid #fcd34d',
                                borderRadius: '8px',
                                cursor: 'pointer',
                                transition: 'all 200ms ease',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '6px',
                                fontSize: '13px',
                                fontWeight: '600',
                                whiteSpace: 'nowrap'
                              }}
                              onMouseEnter={(e) => {
                                e.currentTarget.style.backgroundColor = '#fcd34d';
                                e.currentTarget.style.transform = 'translateY(-2px)';
                                e.currentTarget.style.boxShadow = '0 4px 12px rgba(217, 119, 6, 0.15)';
                              }}
                              onMouseLeave={(e) => {
                                e.currentTarget.style.backgroundColor = '#fef3c7';
                                e.currentTarget.style.transform = 'translateY(0)';
                                e.currentTarget.style.boxShadow = 'none';
                              }}
                            >
                              <FiPlus size={16} />
                              Refill
                            </button>
                            <button
                              onClick={() => {
                                setSelectedMed(med);
                                fetchDispensingHistory();
                              }}
                              style={{
                                padding: '8px 14px',
                                backgroundColor: '#f3e8ff',
                                color: '#7c3aed',
                                border: '1px solid #e9d5ff',
                                borderRadius: '8px',
                                cursor: 'pointer',
                                transition: 'all 200ms ease',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '6px',
                                fontSize: '13px',
                                fontWeight: '600',
                                whiteSpace: 'nowrap'
                              }}
                              onMouseEnter={(e) => {
                                e.currentTarget.style.backgroundColor = '#e9d5ff';
                                e.currentTarget.style.transform = 'translateY(-2px)';
                                e.currentTarget.style.boxShadow = '0 4px 12px rgba(124, 58, 237, 0.15)';
                              }}
                              onMouseLeave={(e) => {
                                e.currentTarget.style.backgroundColor = '#f3e8ff';
                                e.currentTarget.style.transform = 'translateY(0)';
                                e.currentTarget.style.boxShadow = 'none';
                              }}
                            >
                              <FiRefreshCw size={16} />
                              History
                            </button>
                            <button
                              onClick={() => {
                                setSelectedMed(med);
                                setShowAdminModal(true);
                              }}
                              style={{
                                padding: '8px 14px',
                                backgroundColor: '#fce7f3',
                                color: '#be185d',
                                border: '1px solid #fbcfe8',
                                borderRadius: '8px',
                                cursor: 'pointer',
                                transition: 'all 200ms ease',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '6px',
                                fontSize: '13px',
                                fontWeight: '600',
                                whiteSpace: 'nowrap'
                              }}
                              onMouseEnter={(e) => {
                                e.currentTarget.style.backgroundColor = '#fbcfe8';
                                e.currentTarget.style.transform = 'translateY(-2px)';
                                e.currentTarget.style.boxShadow = '0 4px 12px rgba(190, 24, 93, 0.15)';
                              }}
                              onMouseLeave={(e) => {
                                e.currentTarget.style.backgroundColor = '#fce7f3';
                                e.currentTarget.style.transform = 'translateY(0)';
                                e.currentTarget.style.boxShadow = 'none';
                              }}
                            >
                              <FiSettings size={16} />
                              Admin
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        )}

        {/* Detail Modal */}
        {showDetailModal && selectedMed && (
          <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000
          }} onClick={() => setShowDetailModal(false)}>
            <div style={{
              backgroundColor: '#fff',
              borderRadius: '12px',
              padding: '32px',
              maxWidth: '500px',
              width: '90%',
              maxHeight: '90vh',
              overflowY: 'auto',
              boxShadow: '0 20px 25px rgba(0,0,0,0.15)'
            }} onClick={(e) => e.stopPropagation()}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h2 style={{ margin: 0, fontSize: '24px', fontWeight: '700', color: '#0f172a' }}>
                  {selectedMed.name}
                </h2>
                <button
                  onClick={() => setShowDetailModal(false)}
                  style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: '24px', color: '#94a3b8' }}
                >
                  ✕
                </button>
              </div>

              <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '16px',
                marginBottom: '20px'
              }}>
                <div style={{ backgroundColor: '#f8fafc', padding: '12px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>Medication ID</div>
                  <div style={{ fontWeight: '600', color: '#0f172a' }}>{selectedMed.id}</div>
                </div>
                <div style={{ backgroundColor: '#f8fafc', padding: '12px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>Category</div>
                  <div style={{ fontWeight: '600', color: '#0f172a' }}>{selectedMed.category}</div>
                </div>
                <div style={{ backgroundColor: '#f8fafc', padding: '12px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>Quantity</div>
                  <div style={{ fontWeight: '600', color: '#0f172a' }}>{selectedMed.quantity} {selectedMed.unit}</div>
                </div>
                <div style={{ backgroundColor: '#f8fafc', padding: '12px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>Min Stock</div>
                  <div style={{ fontWeight: '600', color: '#0f172a' }}>{selectedMed.minStock} {selectedMed.unit}</div>
                </div>
                <div style={{ backgroundColor: '#f8fafc', padding: '12px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>Expiry Date</div>
                  <div style={{ fontWeight: '600', color: isExpired(selectedMed.expiry) ? '#ef4444' : '#0f172a' }}>
                    {new Date(selectedMed.expiry).toLocaleDateString()}
                  </div>
                </div>
                <div style={{ backgroundColor: '#f8fafc', padding: '12px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>Location</div>
                  <div style={{ fontWeight: '600', color: '#0f172a' }}>{selectedMed.location}</div>
                </div>
                <div style={{ backgroundColor: '#f8fafc', padding: '12px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>Manufacturer</div>
                  <div style={{ fontWeight: '600', color: '#0f172a' }}>{selectedMed.manufacturer}</div>
                </div>
                <div style={{ backgroundColor: '#f8fafc', padding: '12px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>Batch Number</div>
                  <div style={{ fontWeight: '600', color: '#0f172a' }}>{selectedMed.batchNumber}</div>
                </div>
              </div>

              <button
                onClick={() => {
                  setShowDetailModal(false);
                  setShowHandoverModal(true);
                }}
                style={{
                  width: '100%',
                  padding: '12px 16px',
                  backgroundColor: '#10b981',
                  color: '#fff',
                  border: 'none',
                  borderRadius: '8px',
                  fontSize: '14px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'all 200ms ease'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = '#059669';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = '#10b981';
                }}
              >
                Hand Over to Patient
              </button>
            </div>
          </div>
        )}

        {/* Handover Modal */}
        {showHandoverModal && selectedMed && (
          <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1001
          }} onClick={() => setShowHandoverModal(false)}>
            <div style={{
              backgroundColor: '#fff',
              borderRadius: '12px',
              padding: '32px',
              maxWidth: '500px',
              width: '90%',
              boxShadow: '0 20px 25px rgba(0,0,0,0.15)'
            }} onClick={(e) => e.stopPropagation()}>
              <h2 style={{ margin: '0 0 20px 0', fontSize: '24px', fontWeight: '700', color: '#0f172a' }}>
                Hand Over Medication
              </h2>

              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                  Medication
                </label>
                <div style={{
                  backgroundColor: '#f8fafc',
                  padding: '12px 16px',
                  borderRadius: '8px',
                  border: '1px solid #e2e8f0'
                }}>
                  {selectedMed.name}
                </div>
              </div>

              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                  Patient ID
                </label>
                <input
                  type="text"
                  value={handoverPatient}
                  onChange={(e) => setHandoverPatient(e.target.value)}
                  placeholder="Enter patient ID"
                  style={{
                    width: '100%',
                    padding: '12px 16px',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                    fontSize: '14px',
                    boxSizing: 'border-box',
                    outline: 'none'
                  }}
                  onFocus={(e) => {
                    e.target.style.borderColor = '#10b981';
                  }}
                  onBlur={(e) => {
                    e.target.style.borderColor = '#e2e8f0';
                  }}
                />
              </div>

              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                  Quantity Handed Over ({selectedMed.unit})
                </label>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <button
                    onClick={() => setHandoverQuantity(Math.max(1, handoverQuantity - 1))}
                    style={{
                      padding: '8px 12px',
                      backgroundColor: '#f1f5f9',
                      border: '1px solid #e2e8f0',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      fontSize: '18px',
                      color: '#64748b'
                    }}
                  >
                    −
                  </button>
                  <input
                    type="number"
                    value={handoverQuantity}
                    onChange={(e) => setHandoverQuantity(Math.max(1, parseInt(e.target.value) || 1))}
                    min="1"
                    max={selectedMed.quantity}
                    style={{
                      flex: 1,
                      padding: '8px 12px',
                      border: '1px solid #e2e8f0',
                      borderRadius: '6px',
                      fontSize: '14px',
                      textAlign: 'center',
                      outline: 'none'
                    }}
                  />
                  <button
                    onClick={() => setHandoverQuantity(Math.min(selectedMed.quantity, handoverQuantity + 1))}
                    style={{
                      padding: '8px 12px',
                      backgroundColor: '#f1f5f9',
                      border: '1px solid #e2e8f0',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      fontSize: '18px',
                      color: '#64748b'
                    }}
                  >
                    +
                  </button>
                </div>
                <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '8px' }}>
                  Available: {selectedMed.quantity} {selectedMed.unit}
                </div>
              </div>

              <div style={{ display: 'flex', gap: '12px' }}>
                <button
                  onClick={() => setShowHandoverModal(false)}
                  style={{
                    flex: 1,
                    padding: '12px 16px',
                    backgroundColor: '#f1f5f9',
                    color: '#64748b',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontWeight: '600',
                    cursor: 'pointer',
                    transition: 'all 200ms ease'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = '#e2e8f0';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = '#f1f5f9';
                  }}
                >
                  Cancel
                </button>
                <button
                  onClick={handleHandoverMedicine}
                  style={{
                    flex: 1,
                    padding: '12px 16px',
                    backgroundColor: '#10b981',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontWeight: '600',
                    cursor: 'pointer',
                    transition: 'all 200ms ease'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = '#059669';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = '#10b981';
                  }}
                >
                  Confirm Handover
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Dispense Modal */}
        {showDispenseModal && selectedMed && (
          <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1001,
            overflowY: 'auto'
          }} onClick={() => setShowDispenseModal(false)}>
            <div style={{
              backgroundColor: '#fff',
              borderRadius: '12px',
              padding: '32px',
              maxWidth: '600px',
              width: '90%',
              margin: '20px auto',
              boxShadow: '0 20px 25px rgba(0,0,0,0.15)'
            }} onClick={(e) => e.stopPropagation()}>
              <h2 style={{ margin: '0 0 20px 0', fontSize: '24px', fontWeight: '700', color: '#0f172a' }}>
                Dispense Medication
              </h2>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                    Medication
                  </label>
                  <div style={{
                    backgroundColor: '#f8fafc',
                    padding: '12px 16px',
                    borderRadius: '8px',
                    border: '1px solid #e2e8f0',
                    fontSize: '14px',
                    fontWeight: '500'
                  }}>
                    {selectedMed.name}
                  </div>
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                    Dose/Time
                  </label>
                  <input
                    type="text"
                    value={dispenseForm.dosagePerTime}
                    onChange={(e) => setDispenseForm({...dispenseForm, dosagePerTime: e.target.value})}
                    placeholder="e.g., 1 tablet"
                    style={{
                      width: '100%',
                      padding: '12px 16px',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                      fontSize: '14px',
                      boxSizing: 'border-box',
                      outline: 'none'
                    }}
                    onFocus={(e) => { e.target.style.borderColor = '#10b981'; }}
                    onBlur={(e) => { e.target.style.borderColor = '#e2e8f0'; }}
                  />
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                    Pharmacist Name *
                  </label>
                  <input
                    type="text"
                    value={dispenseForm.pharmacistName}
                    onChange={(e) => setDispenseForm({...dispenseForm, pharmacistName: e.target.value})}
                    placeholder="Full name"
                    style={{
                      width: '100%',
                      padding: '12px 16px',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                      fontSize: '14px',
                      boxSizing: 'border-box',
                      outline: 'none'
                    }}
                    onFocus={(e) => { e.target.style.borderColor = '#10b981'; }}
                    onBlur={(e) => { e.target.style.borderColor = '#e2e8f0'; }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                    Patient ID *
                  </label>
                  <input
                    type="text"
                    value={dispenseForm.patientId}
                    onChange={(e) => setDispenseForm({...dispenseForm, patientId: e.target.value})}
                    placeholder="Patient ID"
                    style={{
                      width: '100%',
                      padding: '12px 16px',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                      fontSize: '14px',
                      boxSizing: 'border-box',
                      outline: 'none'
                    }}
                    onFocus={(e) => { e.target.style.borderColor = '#10b981'; }}
                    onBlur={(e) => { e.target.style.borderColor = '#e2e8f0'; }}
                  />
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                    Prescribed By *
                  </label>
                  <input
                    type="text"
                    value={dispenseForm.prescribedByName}
                    onChange={(e) => setDispenseForm({...dispenseForm, prescribedByName: e.target.value})}
                    placeholder="Doctor/Nurse name"
                    style={{
                      width: '100%',
                      padding: '12px 16px',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                      fontSize: '14px',
                      boxSizing: 'border-box',
                      outline: 'none'
                    }}
                    onFocus={(e) => { e.target.style.borderColor = '#10b981'; }}
                    onBlur={(e) => { e.target.style.borderColor = '#e2e8f0'; }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                    Prescriber Role
                  </label>
                  <select
                    value={dispenseForm.prescribedByRole}
                    onChange={(e) => setDispenseForm({...dispenseForm, prescribedByRole: e.target.value})}
                    style={{
                      width: '100%',
                      padding: '12px 16px',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                      fontSize: '14px',
                      backgroundColor: '#fff',
                      outline: 'none'
                    }}
                  >
                    <option value="PHYSICIAN">Physician</option>
                    <option value="NURSE">Nurse</option>
                  </select>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                    Frequency
                  </label>
                  <input
                    type="text"
                    value={dispenseForm.frequency}
                    onChange={(e) => setDispenseForm({...dispenseForm, frequency: e.target.value})}
                    placeholder="e.g., Once daily"
                    style={{
                      width: '100%',
                      padding: '12px 16px',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                      fontSize: '14px',
                      boxSizing: 'border-box',
                      outline: 'none'
                    }}
                    onFocus={(e) => { e.target.style.borderColor = '#10b981'; }}
                    onBlur={(e) => { e.target.style.borderColor = '#e2e8f0'; }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                    Duration
                  </label>
                  <input
                    type="text"
                    value={dispenseForm.duration}
                    onChange={(e) => setDispenseForm({...dispenseForm, duration: e.target.value})}
                    placeholder="e.g., 7 days"
                    style={{
                      width: '100%',
                      padding: '12px 16px',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                      fontSize: '14px',
                      boxSizing: 'border-box',
                      outline: 'none'
                    }}
                    onFocus={(e) => { e.target.style.borderColor = '#10b981'; }}
                    onBlur={(e) => { e.target.style.borderColor = '#e2e8f0'; }}
                  />
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                    Schedule Level (DEA) *
                  </label>
                  <select
                    value={dispenseForm.scheduleLevel}
                    onChange={(e) => {
                      setDispenseForm({...dispenseForm, scheduleLevel: e.target.value});
                      console.log('Schedule Level changed to:', e.target.value);
                    }}
                    style={{
                      width: '100%',
                      padding: '12px 16px',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                      fontSize: '14px',
                      backgroundColor: '#fff',
                      outline: 'none',
                      fontWeight: '600'
                    }}
                  >
                    <option value="OTC">OTC (Over The Counter)</option>
                    <option value="V">Schedule V (Lowest Risk)</option>
                    <option value="IV">Schedule IV</option>
                    <option value="III">Schedule III</option>
                    <option value="II">Schedule II (High Risk - Requires Documentation)</option>
                    <option value="I">Schedule I (Highest Risk - Restricted)</option>
                  </select>
                  <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '4px' }}>
                    Select appropriate DEA schedule for this dispensing
                  </div>
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                    Schedule Level Info
                  </label>
                  <div style={{
                    backgroundColor: dispenseForm.scheduleLevel === 'I' || dispenseForm.scheduleLevel === 'II' ? '#fee2e2' : '#ecfdf5',
                    padding: '12px 16px',
                    borderRadius: '8px',
                    border: '1px solid' + (dispenseForm.scheduleLevel === 'I' || dispenseForm.scheduleLevel === 'II' ? '#fecaca' : '#d1fae5'),
                    fontSize: '12px',
                    fontWeight: '500',
                    color: dispenseForm.scheduleLevel === 'I' || dispenseForm.scheduleLevel === 'II' ? '#dc2626' : '#059669'
                  }}>
                    {dispenseForm.scheduleLevel === 'OTC' && 'Over-the-counter medication - no special restrictions'}
                    {dispenseForm.scheduleLevel === 'V' && 'Low abuse potential - minimal special handling'}
                    {dispenseForm.scheduleLevel === 'IV' && 'Low to moderate abuse potential - standard documentation'}
                    {dispenseForm.scheduleLevel === 'III' && 'Moderate abuse potential - enhanced documentation required'}
                    {dispenseForm.scheduleLevel === 'II' && '⚠️ High abuse potential - requires strict documentation and witnesses'}
                    {dispenseForm.scheduleLevel === 'I' && '🚫 Extreme risk - highly restricted - requires special authorization'}
                  </div>
                </div>
              </div>

              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                  Patient Condition Before
                </label>
                <textarea
                  value={dispenseForm.patientConditionBefore}
                  onChange={(e) => setDispenseForm({...dispenseForm, patientConditionBefore: e.target.value})}
                  placeholder="Describe patient's condition"
                  style={{
                    width: '100%',
                    padding: '12px 16px',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                    fontSize: '14px',
                    boxSizing: 'border-box',
                    minHeight: '80px',
                    outline: 'none',
                    resize: 'vertical'
                  }}
                  onFocus={(e) => { e.target.style.borderColor = '#10b981'; }}
                  onBlur={(e) => { e.target.style.borderColor = '#e2e8f0'; }}
                />
              </div>

              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                  Pharmacist Notes
                </label>
                <textarea
                  value={dispenseForm.pharmacistNotes}
                  onChange={(e) => setDispenseForm({...dispenseForm, pharmacistNotes: e.target.value})}
                  placeholder="Any special instructions or notes"
                  style={{
                    width: '100%',
                    padding: '12px 16px',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                    fontSize: '14px',
                    boxSizing: 'border-box',
                    minHeight: '80px',
                    outline: 'none',
                    resize: 'vertical'
                  }}
                  onFocus={(e) => { e.target.style.borderColor = '#10b981'; }}
                  onBlur={(e) => { e.target.style.borderColor = '#e2e8f0'; }}
                />
              </div>

              <div style={{ display: 'flex', gap: '12px' }}>
                <button
                  onClick={() => {
                    setShowDispenseModal(false);
                    setDispenseForm({
                      pharmacistName: '',
                      pharmacistNotes: '',
                      frequency: 'once daily',
                      scheduledTimes: ['08:00'],
                      duration: '7 days',
                      dosagePerTime: '1',
                      instructions: '',
                      prescribedByName: '',
                      prescribedByRole: 'PHYSICIAN',
                      patientId: '',
                      patientConditionBefore: ''
                    });
                  }}
                  style={{
                    flex: 1,
                    padding: '12px 16px',
                    backgroundColor: '#f1f5f9',
                    color: '#64748b',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontWeight: '600',
                    cursor: 'pointer',
                    transition: 'all 200ms ease'
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#e2e8f0'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#f1f5f9'; }}
                >
                  Cancel
                </button>
                <button
                  onClick={handleDispenseMedication}
                  style={{
                    flex: 1,
                    padding: '12px 16px',
                    backgroundColor: '#10b981',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontWeight: '600',
                    cursor: 'pointer',
                    transition: 'all 200ms ease'
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#059669'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#10b981'; }}
                >
                  ✓ Dispense & Create Schedule
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Refill Modal */}
        {showRefillModal && selectedMed && (
          <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1001
          }} onClick={() => setShowRefillModal(false)}>
            <div style={{
              backgroundColor: '#fff',
              borderRadius: '12px',
              padding: '32px',
              maxWidth: '500px',
              width: '90%',
              boxShadow: '0 20px 25px rgba(0,0,0,0.15)'
            }} onClick={(e) => e.stopPropagation()}>
              <h2 style={{ margin: '0 0 20px 0', fontSize: '24px', fontWeight: '700', color: '#0f172a' }}>
                Refill Medication
              </h2>

              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                  Medication
                </label>
                <div style={{
                  backgroundColor: '#f8fafc',
                  padding: '12px 16px',
                  borderRadius: '8px',
                  border: '1px solid #e2e8f0',
                  fontSize: '14px',
                  fontWeight: '500'
                }}>
                  {selectedMed.name}
                </div>
              </div>

              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                  Quantity to Add *
                </label>
                <input
                  type="number"
                  value={refillForm.quantity}
                  onChange={(e) => setRefillForm({...refillForm, quantity: e.target.value})}
                  placeholder="Enter quantity"
                  min="1"
                  style={{
                    width: '100%',
                    padding: '12px 16px',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                    fontSize: '14px',
                    boxSizing: 'border-box',
                    outline: 'none'
                  }}
                  onFocus={(e) => { e.target.style.borderColor = '#10b981'; }}
                  onBlur={(e) => { e.target.style.borderColor = '#e2e8f0'; }}
                />
              </div>

              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                  Batch Number
                </label>
                <input
                  type="text"
                  value={refillForm.batchNumber}
                  onChange={(e) => setRefillForm({...refillForm, batchNumber: e.target.value})}
                  placeholder="e.g., BATCH-2024-001"
                  style={{
                    width: '100%',
                    padding: '12px 16px',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                    fontSize: '14px',
                    boxSizing: 'border-box',
                    outline: 'none'
                  }}
                  onFocus={(e) => { e.target.style.borderColor = '#10b981'; }}
                  onBlur={(e) => { e.target.style.borderColor = '#e2e8f0'; }}
                />
              </div>

              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                  Expiry Date *
                </label>
                <input
                  type="date"
                  value={refillForm.expiryDate}
                  onChange={(e) => setRefillForm({...refillForm, expiryDate: e.target.value})}
                  style={{
                    width: '100%',
                    padding: '12px 16px',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                    fontSize: '14px',
                    boxSizing: 'border-box',
                    outline: 'none'
                  }}
                  onFocus={(e) => { e.target.style.borderColor = '#10b981'; }}
                  onBlur={(e) => { e.target.style.borderColor = '#e2e8f0'; }}
                />
              </div>

              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                  Manufacturer
                </label>
                <input
                  type="text"
                  value={refillForm.manufacturer}
                  onChange={(e) => setRefillForm({...refillForm, manufacturer: e.target.value})}
                  placeholder="Leave blank to keep current"
                  style={{
                    width: '100%',
                    padding: '12px 16px',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                    fontSize: '14px',
                    boxSizing: 'border-box',
                    outline: 'none'
                  }}
                  onFocus={(e) => { e.target.style.borderColor = '#10b981'; }}
                  onBlur={(e) => { e.target.style.borderColor = '#e2e8f0'; }}
                />
              </div>

              <div style={{ display: 'flex', gap: '12px' }}>
                <button
                  onClick={() => {
                    setShowRefillModal(false);
                    setRefillForm({
                      quantity: '',
                      batchNumber: '',
                      expiryDate: '',
                      manufacturer: ''
                    });
                  }}
                  style={{
                    flex: 1,
                    padding: '12px 16px',
                    backgroundColor: '#f1f5f9',
                    color: '#64748b',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontWeight: '600',
                    cursor: 'pointer',
                    transition: 'all 200ms ease'
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#e2e8f0'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#f1f5f9'; }}
                >
                  Cancel
                </button>
                <button
                  onClick={handleRefillMedication}
                  style={{
                    flex: 1,
                    padding: '12px 16px',
                    backgroundColor: '#f59e0b',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontWeight: '600',
                    cursor: 'pointer',
                    transition: 'all 200ms ease'
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#d97706'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#f59e0b'; }}
                >
                  ✓ Complete Refill
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Add Medication Modal */}
        {showAddMedModal && (
          <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1001,
            overflowY: 'auto'
          }} onClick={() => setShowAddMedModal(false)}>
            <div style={{
              backgroundColor: '#fff',
              borderRadius: '12px',
              padding: '32px',
              maxWidth: '500px',
              width: '90%',
              margin: '20px auto',
              boxShadow: '0 20px 25px rgba(0,0,0,0.15)'
            }} onClick={(e) => e.stopPropagation()}>
              <h2 style={{ margin: '0 0 20px 0', fontSize: '24px', fontWeight: '700', color: '#0f172a' }}>
                Add New Medication
              </h2>

              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                  Medication Name *
                </label>
                <input
                  type="text"
                  value={newMedForm.name}
                  onChange={(e) => setNewMedForm({...newMedForm, name: e.target.value})}
                  placeholder="e.g., Amoxicillin"
                  style={{
                    width: '100%',
                    padding: '12px 16px',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                    fontSize: '14px',
                    boxSizing: 'border-box',
                    outline: 'none'
                  }}
                  onFocus={(e) => { e.target.style.borderColor = '#10b981'; }}
                  onBlur={(e) => { e.target.style.borderColor = '#e2e8f0'; }}
                />
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                    Strength
                  </label>
                  <input
                    type="text"
                    value={newMedForm.strength}
                    onChange={(e) => setNewMedForm({...newMedForm, strength: e.target.value})}
                    placeholder="e.g., 500mg"
                    style={{
                      width: '100%',
                      padding: '12px 16px',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                      fontSize: '14px',
                      boxSizing: 'border-box',
                      outline: 'none'
                    }}
                    onFocus={(e) => { e.target.style.borderColor = '#10b981'; }}
                    onBlur={(e) => { e.target.style.borderColor = '#e2e8f0'; }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                    Form
                  </label>
                  <select
                    value={newMedForm.form}
                    onChange={(e) => setNewMedForm({...newMedForm, form: e.target.value})}
                    style={{
                      width: '100%',
                      padding: '12px 16px',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                      fontSize: '14px',
                      backgroundColor: '#fff',
                      outline: 'none'
                    }}
                  >
                    <option value="Tablet">Tablet</option>
                    <option value="Capsule">Capsule</option>
                    <option value="Syrup">Syrup</option>
                    <option value="Injection">Injection</option>
                    <option value="Ointment">Ointment</option>
                  </select>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                    Route
                  </label>
                  <select
                    value={newMedForm.route}
                    onChange={(e) => setNewMedForm({...newMedForm, route: e.target.value})}
                    style={{
                      width: '100%',
                      padding: '12px 16px',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                      fontSize: '14px',
                      backgroundColor: '#fff',
                      outline: 'none'
                    }}
                  >
                    <option value="PO">Oral (PO)</option>
                    <option value="IV">Intravenous (IV)</option>
                    <option value="IM">Intramuscular (IM)</option>
                    <option value="SC">Subcutaneous (SC)</option>
                    <option value="Topical">Topical</option>
                  </select>
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                    Current Stock *
                  </label>
                  <input
                    type="number"
                    value={newMedForm.stock}
                    onChange={(e) => setNewMedForm({...newMedForm, stock: e.target.value})}
                    placeholder="Quantity"
                    min="0"
                    style={{
                      width: '100%',
                      padding: '12px 16px',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                      fontSize: '14px',
                      boxSizing: 'border-box',
                      outline: 'none'
                    }}
                    onFocus={(e) => { e.target.style.borderColor = '#10b981'; }}
                    onBlur={(e) => { e.target.style.borderColor = '#e2e8f0'; }}
                  />
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                    Unit Price (ZAR) *
                  </label>
                  <input
                    type="number"
                    value={newMedForm.unitPrice}
                    onChange={(e) => setNewMedForm({...newMedForm, unitPrice: e.target.value})}
                    placeholder="Price per unit in Rands"
                    step="0.01"
                    min="0"
                    style={{
                      width: '100%',
                      padding: '12px 16px',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                      fontSize: '14px',
                      boxSizing: 'border-box',
                      outline: 'none'
                    }}
                    onFocus={(e) => { e.target.style.borderColor = '#10b981'; }}
                    onBlur={(e) => { e.target.style.borderColor = '#e2e8f0'; }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                    Min Stock Level
                  </label>
                  <input
                    type="number"
                    value={newMedForm.minStockLevel}
                    onChange={(e) => setNewMedForm({...newMedForm, minStockLevel: e.target.value})}
                    placeholder="e.g., 20"
                    min="0"
                    style={{
                      width: '100%',
                      padding: '12px 16px',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                      fontSize: '14px',
                      boxSizing: 'border-box',
                      outline: 'none'
                    }}
                    onFocus={(e) => { e.target.style.borderColor = '#10b981'; }}
                    onBlur={(e) => { e.target.style.borderColor = '#e2e8f0'; }}
                  />
                </div>
              </div>

              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                  Max Stock Level
                </label>
                <input
                  type="number"
                  value={newMedForm.maxStockLevel}
                  onChange={(e) => setNewMedForm({...newMedForm, maxStockLevel: e.target.value})}
                  placeholder="e.g., 200"
                  min="0"
                  style={{
                    width: '100%',
                    padding: '12px 16px',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                    fontSize: '14px',
                    boxSizing: 'border-box',
                    outline: 'none'
                  }}
                  onFocus={(e) => { e.target.style.borderColor = '#10b981'; }}
                  onBlur={(e) => { e.target.style.borderColor = '#e2e8f0'; }}
                />
              </div>

              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                  Drug Schedule (DEA/FDA Classification)
                </label>
                <select
                  value={newMedForm.schedule}
                  onChange={(e) => setNewMedForm({...newMedForm, schedule: e.target.value})}
                  style={{
                    width: '100%',
                    padding: '12px 16px',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                    fontSize: '14px',
                    backgroundColor: '#fff',
                    outline: 'none'
                  }}
                >
                  <option value="I">Schedule I - Highest Risk (No accepted medical use)</option>
                  <option value="II">Schedule II - High Risk (e.g., morphine, cocaine)</option>
                  <option value="III">Schedule III - Moderate Risk (e.g., codeine, ketamine)</option>
                  <option value="IV">Schedule IV - Lower Risk (e.g., benzodiazepines)</option>
                  <option value="V">Schedule V - Lowest Risk (e.g., cough syrups)</option>
                </select>
              </div>

              <div style={{ display: 'flex', gap: '12px' }}>
                <button
                  onClick={() => {
                    setShowAddMedModal(false);
                    setNewMedForm({
                      name: '',
                      strength: '',
                      form: 'Tablet',
                      route: 'PO',
                      stock: '',
                      unitPrice: '',
                      minStockLevel: '',
                      maxStockLevel: '',
                      schedule: 'V'
                    });
                  }}
                  style={{
                    flex: 1,
                    padding: '12px 16px',
                    backgroundColor: '#f1f5f9',
                    color: '#64748b',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontWeight: '600',
                    cursor: 'pointer',
                    transition: 'all 200ms ease'
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#e2e8f0'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#f1f5f9'; }}
                >
                  Cancel
                </button>
                <button
                  onClick={handleAddMedication}
                  style={{
                    flex: 1,
                    padding: '12px 16px',
                    backgroundColor: '#10b981',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontWeight: '600',
                    cursor: 'pointer',
                    transition: 'all 200ms ease'
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#059669'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#10b981'; }}
                >
                  + Add to Inventory
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Record Administration Modal */}
        {showAdminModal && selectedMed && (
          <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1001,
            overflowY: 'auto'
          }} onClick={() => setShowAdminModal(false)}>
            <div style={{
              backgroundColor: '#fff',
              borderRadius: '12px',
              padding: '32px',
              maxWidth: '500px',
              width: '90%',
              margin: '20px auto',
              boxShadow: '0 20px 25px rgba(0,0,0,0.15)'
            }} onClick={(e) => e.stopPropagation()}>
              <h2 style={{ margin: '0 0 20px 0', fontSize: '24px', fontWeight: '700', color: '#0f172a' }}>
                Record Medication Administration
              </h2>

              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                  Medication
                </label>
                <div style={{
                  backgroundColor: '#f8fafc',
                  padding: '12px 16px',
                  borderRadius: '8px',
                  border: '1px solid #e2e8f0',
                  fontSize: '14px',
                  fontWeight: '500'
                }}>
                  {selectedMed.name}
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                    Staff Name *
                  </label>
                  <input
                    type="text"
                    value={adminForm.staffName}
                    onChange={(e) => setAdminForm({...adminForm, staffName: e.target.value})}
                    placeholder="Who gave the medication"
                    style={{
                      width: '100%',
                      padding: '12px 16px',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                      fontSize: '14px',
                      boxSizing: 'border-box',
                      outline: 'none'
                    }}
                    onFocus={(e) => { e.target.style.borderColor = '#10b981'; }}
                    onBlur={(e) => { e.target.style.borderColor = '#e2e8f0'; }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                    Staff Role
                  </label>
                  <select
                    value={adminForm.staffRole}
                    onChange={(e) => setAdminForm({...adminForm, staffRole: e.target.value})}
                    style={{
                      width: '100%',
                      padding: '12px 16px',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                      fontSize: '14px',
                      backgroundColor: '#fff',
                      outline: 'none'
                    }}
                  >
                    <option value="NURSE">Nurse</option>
                    <option value="PHYSICIAN">Physician</option>
                    <option value="PHARMACIST">Pharmacist</option>
                  </select>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                    Time Administered *
                  </label>
                  <input
                    type="datetime-local"
                    value={adminForm.administeredTime}
                    onChange={(e) => setAdminForm({...adminForm, administeredTime: e.target.value})}
                    style={{
                      width: '100%',
                      padding: '12px 16px',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                      fontSize: '14px',
                      boxSizing: 'border-box',
                      outline: 'none'
                    }}
                    onFocus={(e) => { e.target.style.borderColor = '#10b981'; }}
                    onBlur={(e) => { e.target.style.borderColor = '#e2e8f0'; }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                    Dosage Given *
                  </label>
                  <input
                    type="text"
                    value={adminForm.dosageGiven}
                    onChange={(e) => setAdminForm({...adminForm, dosageGiven: e.target.value})}
                    placeholder="e.g., 1 tablet"
                    style={{
                      width: '100%',
                      padding: '12px 16px',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                      fontSize: '14px',
                      boxSizing: 'border-box',
                      outline: 'none'
                    }}
                    onFocus={(e) => { e.target.style.borderColor = '#10b981'; }}
                    onBlur={(e) => { e.target.style.borderColor = '#e2e8f0'; }}
                  />
                </div>
              </div>

              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                  Route Used
                </label>
                <select
                  value={adminForm.routeUsed}
                  onChange={(e) => setAdminForm({...adminForm, routeUsed: e.target.value})}
                  style={{
                    width: '100%',
                    padding: '12px 16px',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                    fontSize: '14px',
                    backgroundColor: '#fff',
                    outline: 'none'
                  }}
                >
                  <option value="PO">Oral (PO)</option>
                  <option value="IV">Intravenous (IV)</option>
                  <option value="IM">Intramuscular (IM)</option>
                  <option value="SC">Subcutaneous (SC)</option>
                  <option value="Topical">Topical</option>
                </select>
              </div>

              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                  Patient Condition After
                </label>
                <textarea
                  value={adminForm.patientConditionAfter}
                  onChange={(e) => setAdminForm({...adminForm, patientConditionAfter: e.target.value})}
                  placeholder="Patient condition after administration"
                  style={{
                    width: '100%',
                    padding: '12px 16px',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                    fontSize: '14px',
                    boxSizing: 'border-box',
                    minHeight: '80px',
                    outline: 'none',
                    resize: 'vertical'
                  }}
                  onFocus={(e) => { e.target.style.borderColor = '#10b981'; }}
                  onBlur={(e) => { e.target.style.borderColor = '#e2e8f0'; }}
                />
              </div>

              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                  Side Effects Observed
                </label>
                <textarea
                  value={adminForm.sideEffectsObserved}
                  onChange={(e) => setAdminForm({...adminForm, sideEffectsObserved: e.target.value})}
                  placeholder="Any side effects or adverse reactions"
                  style={{
                    width: '100%',
                    padding: '12px 16px',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                    fontSize: '14px',
                    boxSizing: 'border-box',
                    minHeight: '80px',
                    outline: 'none',
                    resize: 'vertical'
                  }}
                  onFocus={(e) => { e.target.style.borderColor = '#10b981'; }}
                  onBlur={(e) => { e.target.style.borderColor = '#e2e8f0'; }}
                />
              </div>

              <div style={{ display: 'flex', gap: '12px' }}>
                <button
                  onClick={() => {
                    setShowAdminModal(false);
                    setAdminForm({
                      staffName: '',
                      staffRole: 'NURSE',
                      administeredTime: '',
                      dosageGiven: '',
                      routeUsed: 'PO',
                      patientConditionAfter: '',
                      sideEffectsObserved: ''
                    });
                  }}
                  style={{
                    flex: 1,
                    padding: '12px 16px',
                    backgroundColor: '#f1f5f9',
                    color: '#64748b',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontWeight: '600',
                    cursor: 'pointer',
                    transition: 'all 200ms ease'
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#e2e8f0'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#f1f5f9'; }}
                >
                  Cancel
                </button>
                <button
                  onClick={handleRecordAdministration}
                  style={{
                    flex: 1,
                    padding: '12px 16px',
                    backgroundColor: '#ec4899',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontWeight: '600',
                    cursor: 'pointer',
                    transition: 'all 200ms ease'
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#be185d'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#ec4899'; }}
                >
                  ✓ Record Administration
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Dispensing History Modal */}
        {showDispensingHistoryModal && (
          <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1002,
            overflowY: 'auto'
          }} onClick={() => setShowDispensingHistoryModal(false)}>
            <div style={{
              backgroundColor: '#fff',
              borderRadius: '12px',
              padding: '32px',
              maxWidth: '900px',
              width: '95%',
              margin: '20px auto',
              maxHeight: '90vh',
              overflowY: 'auto',
              boxShadow: '0 20px 25px rgba(0,0,0,0.15)'
            }} onClick={(e) => e.stopPropagation()}>
              <h2 style={{ margin: '0 0 20px 0', fontSize: '24px', fontWeight: '700', color: '#0f172a' }}>
                Dispensing History & Audit Trail
              </h2>

              {selectedMed && (
                <div style={{ marginBottom: '24px', padding: '16px', backgroundColor: '#f8fafc', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                  <h3 style={{ margin: '0 0 12px 0', fontSize: '16px', fontWeight: '600', color: '#0f172a' }}>
                    {selectedMed.name} {selectedMed.strength && `(${selectedMed.strength})`}
                  </h3>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
                    <div>
                      <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>Current Stock</div>
                      <div style={{ fontSize: '18px', fontWeight: '700', color: '#0f172a' }}>
                        {dispensingHistory.medication?.currentStock || 'N/A'}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>Total Dispensed</div>
                      <div style={{ fontSize: '18px', fontWeight: '700', color: '#10b981' }}>
                        {dispensingHistory.statistics?.totalDispensed || 0}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>Dispense Count</div>
                      <div style={{ fontSize: '18px', fontWeight: '700', color: '#0284c7' }}>
                        {dispensingHistory.statistics?.totalRecords || 0}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>Avg per Dispense</div>
                      <div style={{ fontSize: '18px', fontWeight: '700', color: '#f59e0b' }}>
                        {dispensingHistory.statistics?.averageQuantityPerDispense || 'N/A'}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {historyLoading ? (
                <div style={{ textAlign: 'center', padding: '32px', color: '#94a3b8' }}>
                  Loading dispensing history...
                </div>
              ) : dispensingHistory.dispensingHistory && dispensingHistory.dispensingHistory.length > 0 ? (
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                    <thead>
                      <tr style={{ backgroundColor: '#f1f5f9', borderBottom: '2px solid #e2e8f0' }}>
                        <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#0f172a' }}>Date/Time</th>
                        <th style={{ padding: '12px', textAlign: 'center', fontWeight: '600', color: '#0f172a' }}>Quantity</th>
                        <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#0f172a' }}>Dispensed By</th>
                        <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#0f172a' }}>Patient</th>
                        <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#0f172a' }}>Batch</th>
                        <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#0f172a' }}>Prescribed By</th>
                      </tr>
                    </thead>
                    <tbody>
                      {dispensingHistory.dispensingHistory.map((record, idx) => (
                        <tr key={idx} style={{ borderBottom: '1px solid #e2e8f0', backgroundColor: idx % 2 === 0 ? '#fff' : '#f8fafc' }}>
                          <td style={{ padding: '12px', color: '#0f172a' }}>
                            {new Date(record.dispensedAt).toLocaleString()}
                          </td>
                          <td style={{ padding: '12px', textAlign: 'center', fontWeight: '600', color: '#10b981' }}>
                            {record.quantityDispensed}
                          </td>
                          <td style={{ padding: '12px', color: '#64748b' }}>
                            {record.dispensedBy || 'N/A'}
                          </td>
                          <td style={{ padding: '12px', color: '#64748b' }}>
                            {record.dispensedToPatient || 'N/A'}
                          </td>
                          <td style={{ padding: '12px', color: '#64748b', fontSize: '12px' }}>
                            {record.batchUsed || 'N/A'}
                          </td>
                          <td style={{ padding: '12px', color: '#64748b' }}>
                            {record.prescribedBy || 'N/A'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: '32px', color: '#94a3b8' }}>
                  No dispensing history available for this medication
                </div>
              )}

              <div style={{ display: 'flex', gap: '12px', marginTop: '24px', justifyContent: 'flex-end' }}>
                <button
                  onClick={() => setShowDispensingHistoryModal(false)}
                  style={{
                    flex: 1,
                    padding: '12px 16px',
                    backgroundColor: '#e2e8f0',
                    color: '#0f172a',
                    border: 'none',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontWeight: '600',
                    cursor: 'pointer',
                    transition: 'all 200ms ease'
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#cbd5e1'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#e2e8f0'; }}
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        )}

        {/* CSV Upload Modal */}
        {showCSVUploadModal && (
          <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1003,
            overflowY: 'auto'
          }} onClick={() => setShowCSVUploadModal(false)}>
            <div style={{
              backgroundColor: '#fff',
              borderRadius: '12px',
              padding: '32px',
              maxWidth: '700px',
              width: '95%',
              margin: '20px auto',
              boxShadow: '0 20px 25px rgba(0,0,0,0.15)'
            }} onClick={(e) => e.stopPropagation()}>
              <h2 style={{ margin: '0 0 20px 0', fontSize: '24px', fontWeight: '700', color: '#0f172a' }}>
                Upload CSV to Refill Medications
              </h2>

              <div style={{ marginBottom: '24px' }}>
                <p style={{ color: '#64748b', marginBottom: '16px' }}>
                  Upload a CSV file to bulk refill existing medications or add new ones. Use the template format below:
                </p>

                <div style={{
                  backgroundColor: '#f8fafc',
                  padding: '12px 16px',
                  borderRadius: '8px',
                  border: '1px solid #e2e8f0',
                  fontSize: '12px',
                  fontFamily: 'monospace',
                  color: '#64748b',
                  overflowX: 'auto'
                }}>
                  <div>medicationName,strength,form,route,quantity,unitPrice,minStockLevel,batchNumber</div>
                  <div>Amoxicillin,500mg,Tablet,PO,100,2.50,20,BATCH-2024-001</div>
                  <div>Ibuprofen,200mg,Tablet,PO,50,1.25,25,BATCH-2024-002</div>
                </div>

                <ul style={{ marginTop: '12px', fontSize: '12px', color: '#64748b', paddingLeft: '20px' }}>
                  <li><strong>medicationName</strong> - Name of medication (required)</li>
                  <li><strong>quantity</strong> - Units to add/refill (required)</li>
                  <li><strong>batchNumber</strong> - Batch/Lot number (required)</li>
                  <li><strong>unitPrice</strong> - Price per unit (optional)</li>
                  <li><strong>minStockLevel</strong> - Minimum stock threshold (optional, default 10)</li>
                  <li><strong>strength, form, route</strong> - For new medications (optional)</li>
                </ul>
              </div>

              {/* Drag & Drop Area */}
              <div
                onClick={() => document.getElementById('csvFileInput').click()}
                onDragOver={(e) => {
                  e.preventDefault();
                  e.currentTarget.style.backgroundColor = '#ecfdf5';
                  e.currentTarget.style.borderColor = '#10b981';
                }}
                onDragLeave={(e) => {
                  e.currentTarget.style.backgroundColor = '#f8fafc';
                  e.currentTarget.style.borderColor = '#e2e8f0';
                }}
                onDrop={(e) => {
                  e.preventDefault();
                  e.currentTarget.style.backgroundColor = '#f8fafc';
                  e.currentTarget.style.borderColor = '#e2e8f0';
                  const file = e.dataTransfer.files[0];
                  if (file && file.name.endsWith('.csv')) {
                    setCSVFile(file);
                  }
                }}
                style={{
                  border: '2px dashed #e2e8f0',
                  borderRadius: '8px',
                  padding: '32px',
                  textAlign: 'center',
                  backgroundColor: '#f8fafc',
                  cursor: 'pointer',
                  transition: 'all 200ms ease',
                  marginBottom: '20px'
                }}
              >
                <div style={{ fontSize: '32px', marginBottom: '12px' }}>📄</div>
                <div style={{ fontWeight: '600', color: '#0f172a', marginBottom: '4px' }}>
                  {csvFile ? csvFile.name : 'Drag CSV file here or click to browse'}
                </div>
                <div style={{ fontSize: '12px', color: '#64748b' }}>
                  {csvFile ? '✓ File selected' : 'Max file size: 5MB'}
                </div>
              </div>

              <input
                id="csvFileInput"
                type="file"
                accept=".csv"
                onChange={(e) => {
                  if (e.target.files[0]) {
                    setCSVFile(e.target.files[0]);
                  }
                }}
                style={{ display: 'none' }}
              />

              <div style={{ display: 'flex', gap: '12px' }}>
                <button
                  onClick={() => {
                    setShowCSVUploadModal(false);
                    setCSVFile(null);
                    setCSVUploadResult(null);
                  }}
                  style={{
                    flex: 1,
                    padding: '12px 16px',
                    backgroundColor: '#e2e8f0',
                    color: '#0f172a',
                    border: 'none',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontWeight: '600',
                    cursor: 'pointer',
                    transition: 'all 200ms ease'
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#cbd5e1'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#e2e8f0'; }}
                >
                  Cancel
                </button>
                <button
                  onClick={handleCSVUpload}
                  disabled={!csvFile || csvUploadLoading}
                  style={{
                    flex: 1,
                    padding: '12px 16px',
                    backgroundColor: csvFile && !csvUploadLoading ? '#10b981' : '#cbd5e1',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontWeight: '600',
                    cursor: csvFile && !csvUploadLoading ? 'pointer' : 'not-allowed',
                    transition: 'all 200ms ease'
                  }}
                  onMouseEnter={(e) => {
                    if (csvFile && !csvUploadLoading) {
                      e.currentTarget.style.backgroundColor = '#059669';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (csvFile && !csvUploadLoading) {
                      e.currentTarget.style.backgroundColor = '#10b981';
                    }
                  }}
                >
                  {csvUploadLoading ? '⏳ Uploading...' : '📤 Upload & Process'}
                </button>
              </div>

              {csvUploadResult && (
                <div style={{ marginTop: '20px', padding: '16px', backgroundColor: '#f8fafc', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                  <h3 style={{ margin: '0 0 12px 0', fontSize: '14px', fontWeight: '600', color: '#0f172a' }}>
                    Upload Results
                  </h3>
                  <div style={{ fontSize: '13px', color: '#64748b', lineHeight: '1.6' }}>
                    <div>✓ <strong>{csvUploadResult.summary.successCount}</strong> medications processed successfully</div>
                    {csvUploadResult.summary.failureCount > 0 && (
                      <div>✗ <strong>{csvUploadResult.summary.failureCount}</strong> records failed</div>
                    )}
                    <div>📦 <strong>{csvUploadResult.summary.totalStockAdded}</strong> total units added</div>
                    <div>💰 <strong>R {csvUploadResult.summary.totalValue.toLocaleString('en-ZA', {maximumFractionDigits: 2})}</strong> total value</div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </SidebarLayout>
  );
};

export default Pharmacy;
