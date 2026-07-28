import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { FiChevronLeft, FiClock, FiSearch, FiArrowRight, FiFilter } from 'react-icons/fi';
import SidebarLayout from '../SidebarLayout';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

const formatDateTime = (value) => {
  if (!value) return 'N/A';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'N/A';
  return date.toLocaleString(undefined, {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
};

const formatDate = (value) => {
  if (!value) return 'N/A';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'N/A';
  return date.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
};

const statusStyle = (status) => {
  const normalized = String(status || '').toLowerCase();
  if (normalized === 'dispensed') return { background: '#d1fae5', color: '#047857' };
  if (normalized === 'accepted') return { background: '#fef3c7', color: '#92400e' };
  if (normalized === 'cancelled') return { background: '#fee2e2', color: '#991b1b' };
  if (normalized === 'submitted' || normalized === 'issued' || normalized === 'pending') return { background: '#e0f2fe', color: '#0c4a6e' };
  return { background: '#e2e8f0', color: '#0f172a' };
};

const valueOrNA = (value) => (value === null || value === undefined || value === '' ? 'N/A' : value);

const OrderedMeds = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const query = useMemo(() => new URLSearchParams(location.search), [location.search]);

  const [prescriptions, setPrescriptions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedPrescriptionId, setSelectedPrescriptionId] = useState(null);
  const [patientIdFilter, setPatientIdFilter] = useState(query.get('patientId') || '');
  const [startDate, setStartDate] = useState(query.get('startDate') || '');
  const [endDate, setEndDate] = useState(query.get('endDate') || '');

  useEffect(() => {
    setPatientIdFilter(query.get('patientId') || '');
  }, [query]);

  const loadOrderedMeds = async () => {
    setLoading(true);
    setError('');
    try {
      const token = localStorage.getItem('token');
      const params = new URLSearchParams();
      if (patientIdFilter) params.set('patientId', patientIdFilter.trim());
      if (startDate) params.set('startDate', startDate);
      if (endDate) params.set('endDate', endDate);
      const response = await fetch(`${BACKEND_URL}/api/prescriptions/ordered-meds?${params.toString()}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        cache: 'no-store'
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || 'Failed to load ordered medications');
      setPrescriptions(Array.isArray(data.prescriptions) ? data.prescriptions : []);
      setSelectedPrescriptionId((prevId) => {
        if (prevId && data.prescriptions?.some((p) => String(p._id) === String(prevId))) return prevId;
        return data.prescriptions?.[0]?._id || null;
      });
    } catch (err) {
      setError(err.message || 'Unable to load ordered meds');
      setPrescriptions([]);
      setSelectedPrescriptionId(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadOrderedMeds();
  }, [patientIdFilter, startDate, endDate]);

  const selectedPrescription = prescriptions.find((p) => String(p._id) === String(selectedPrescriptionId)) || prescriptions[0] || null;
  const listItems = prescriptions.map((prescription) => {
    const primaryMedication = prescription.items?.[0] || {};
    return (
      <button
        key={prescription._id}
        type="button"
        onClick={() => setSelectedPrescriptionId(prescription._id)}
        style={{
          width: '100%',
          padding: 16,
          textAlign: 'left',
          borderRadius: 20,
          border: selectedPrescriptionId === prescription._id ? '1px solid #2563eb' : '1px solid #e2e8f0',
          background: selectedPrescriptionId === prescription._id ? '#eff6ff' : '#ffffff',
          cursor: 'pointer',
          marginBottom: 12,
          display: 'grid',
          gridTemplateColumns: '1fr auto',
          gap: 16,
          alignItems: 'center'
        }}
      >
        <div style={{ display: 'grid', gap: 6 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 14, fontWeight: 700, color: '#0f172a' }}>{valueOrNA(prescription.patient_details?.name || prescription.patient_name || 'Patient')}</span>
            <span style={{ fontSize: 12, color: '#64748b' }}>({valueOrNA(prescription.patient_details?.id_number || prescription.patientId)})</span>
          </div>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
            <span style={{ fontSize: 12, color: '#475569' }}>{primaryMedication.medication_name || primaryMedication.medicationName || 'Medication order'}</span>
            <span style={{ fontSize: 12, color: '#64748b' }}>{prescription.items?.length > 1 ? `${prescription.items.length} items` : '1 item'}</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12, color: '#0f172a' }}><FiClock size={12} /> {formatDateTime(prescription.date_issued || prescription.dateIssued)}</span>
            <span style={{ padding: '4px 10px', borderRadius: 999, fontSize: 11, fontWeight: 700, ...statusStyle(prescription.status) }}>{valueOrNA(prescription.status)}</span>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
          <FiArrowRight size={16} color="#0f172a" />
        </div>
      </button>
    );
  });

  const handleOpenReport = (id) => {
    if (!id) return;
    navigate(`/prescription/${id}`);
  };

  const displayedPrescription = selectedPrescription || null;

  return (
    <SidebarLayout>
      <div style={{ maxWidth: 1200, width: '100%', marginBottom: 40 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, marginBottom: 24, flexWrap: 'wrap' }}>
          <div style={{ minWidth: 280, flex: '1 1 320px' }}>
            <h1 style={{ margin: 0, fontSize: 34, fontWeight: 800, color: '#0f172a' }}>Ordered Meds</h1>
            <p style={{ margin: '10px 0 0', color: '#475569', fontSize: 15, maxWidth: 760 }}>Review physician medication orders and pharmacy status in a timeline-inspired list. Filter by patient, order date, and status before opening the prescription report.</p>
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center' }}>
            <button
              type="button"
              onClick={() => navigate('/dashboard')}
              style={{ borderRadius: 999, border: '1px solid #e2e8f0', background: '#fff', color: '#0f172a', padding: '10px 16px', cursor: 'pointer', fontWeight: 700 }}
            ><FiChevronLeft size={16} style={{ marginRight: 6 }} /> Back to Dashboard</button>
          </div>
        </div>

        <div style={{ display: 'grid', gap: 16, marginBottom: 28, gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
          <div style={{ padding: 16, borderRadius: 18, border: '1px solid #e2e8f0', background: '#fff', boxShadow: '0 8px 24px rgba(15,23,42,0.04)' }}>
            <label style={{ display: 'block', marginBottom: 8, color: '#475569', fontSize: 13, fontWeight: 700 }}>Patient ID / MRN</label>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <FiSearch size={16} color="#64748b" />
              <input
                value={patientIdFilter}
                onChange={(e) => setPatientIdFilter(e.target.value)}
                placeholder="Enter patient ID"
                style={{ flex: 1, border: 'none', outline: 'none', fontSize: 14, color: '#0f172a' }}
              />
            </div>
          </div>
          <div style={{ padding: 16, borderRadius: 18, border: '1px solid #e2e8f0', background: '#fff', boxShadow: '0 8px 24px rgba(15,23,42,0.04)' }}>
            <label style={{ display: 'block', marginBottom: 8, color: '#475569', fontSize: 13, fontWeight: 700 }}>Start date</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              style={{ width: '100%', border: 'none', outline: 'none', fontSize: 14, color: '#0f172a' }}
            />
          </div>
          <div style={{ padding: 16, borderRadius: 18, border: '1px solid #e2e8f0', background: '#fff', boxShadow: '0 8px 24px rgba(15,23,42,0.04)' }}>
            <label style={{ display: 'block', marginBottom: 8, color: '#475569', fontSize: 13, fontWeight: 700 }}>End date</label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              style={{ width: '100%', border: 'none', outline: 'none', fontSize: 14, color: '#0f172a' }}
            />
          </div>
          <div style={{ padding: 16, borderRadius: 18, border: '1px solid #e2e8f0', background: '#fff', boxShadow: '0 8px 24px rgba(15,23,42,0.04)' }}>
            <label style={{ display: 'block', marginBottom: 8, color: '#475569', fontSize: 13, fontWeight: 700 }}>Status</label>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
              {['PENDING', 'ISSUED', 'SUBMITTED', 'ACCEPTED', 'DISPENSED', 'CANCELLED'].map((status) => (
                <button
                  key={status}
                  type="button"
                  onClick={() => {
                    const next = status === 'PENDING' ? 'pending' : status.toLowerCase();
                    if (next === 'pending') setEndDate(endDate);
                  }}
                  style={{
                    borderRadius: 999,
                    border: '1px solid #e2e8f0',
                    padding: '6px 12px',
                    background: '#f8fafc',
                    color: '#334155',
                    fontSize: 12,
                    cursor: 'not-allowed'
                  }}
                  disabled
                >
                  {status}
                </button>
              ))}
            </div>
            <div style={{ marginTop: 10, color: '#94a3b8', fontSize: 12 }}>Status filters are not yet enabled in this view.</div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 20 }}>
          <div style={{ minHeight: 480, padding: 20, borderRadius: 24, border: '1px solid #e2e8f0', background: '#ffffff', boxShadow: '0 16px 42px rgba(15,23,42,0.06)' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 18 }}>
              <div>
                <div style={{ fontSize: 14, fontWeight: 700, color: '#0f172a' }}>Medication order history</div>
                <div style={{ fontSize: 12, color: '#64748b' }}>{prescriptions.length} order{prescriptions.length === 1 ? '' : 's'} found</div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#475569', fontSize: 13 }}>
                <FiFilter size={14} /> Date-range & patient
              </div>
            </div>
            {loading ? (
              <div style={{ color: '#64748b', padding: 24 }}>Loading ordered medications...</div>
            ) : error ? (
              <div style={{ color: '#b91c1c', padding: 24 }}>{error}</div>
            ) : prescriptions.length === 0 ? (
              <div style={{ color: '#475569', padding: 24 }}>No ordered meds match the current filters. Adjust the patient ID or date range to refresh.</div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column' }}>{listItems}</div>
            )}
          </div>

          <div style={{ minHeight: 480, padding: 20, borderRadius: 24, border: '1px solid #e2e8f0', background: '#f8fafc', boxShadow: '0 16px 42px rgba(15,23,42,0.06)' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 18 }}>
              <div>
                <div style={{ fontSize: 14, fontWeight: 700, color: '#0f172a' }}>Selected order details</div>
                <div style={{ fontSize: 12, color: '#64748b' }}>Tap any order from the timeline list to inspect it.</div>
              </div>
              <button
                type="button"
                onClick={() => handleOpenReport(selectedPrescription?._id)}
                disabled={!selectedPrescription}
                style={{
                  borderRadius: 999,
                  border: 'none',
                  background: selectedPrescription ? '#0f172a' : '#cbd5e1',
                  color: '#fff',
                  padding: '10px 16px',
                  cursor: selectedPrescription ? 'pointer' : 'not-allowed',
                  fontWeight: 700,
                  fontSize: 13
                }}
              >
                Open prescription
              </button>
            </div>

            {selectedPrescription ? (
              <div style={{ display: 'grid', gap: 16 }}>
                <div style={{ display: 'grid', gap: 10 }}>
                  <span style={{ fontSize: 12, color: '#475569', fontWeight: 700 }}>Patient</span>
                  <div style={{ fontSize: 16, fontWeight: 700, color: '#0f172a' }}>{valueOrNA(selectedPrescription.patient_details?.name || selectedPrescription.patient_name)}</div>
                  <div style={{ fontSize: 13, color: '#64748b' }}>ID: {valueOrNA(selectedPrescription.patient_details?.id_number || selectedPrescription.patientId)}</div>
                </div>
                <div style={{ display: 'grid', gap: 9, padding: 16, borderRadius: 18, background: '#fff', border: '1px solid #e2e8f0' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, alignItems: 'center' }}>
                    <div>
                      <div style={{ fontSize: 12, color: '#475569', fontWeight: 700 }}>Order created</div>
                      <div style={{ fontSize: 14, color: '#0f172a' }}>{formatDateTime(selectedPrescription.date_issued || selectedPrescription.dateIssued)}</div>
                    </div>
                    <span style={{ padding: '6px 12px', borderRadius: 999, fontSize: 12, fontWeight: 700, ...statusStyle(selectedPrescription.status) }}>{selectedPrescription.status}</span>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                    <div style={{ display: 'grid', gap: 4 }}>
                      <span style={{ fontSize: 12, color: '#475569' }}>Encounter</span>
                      <span style={{ fontSize: 13, color: '#0f172a' }}>{valueOrNA(selectedPrescription.encounterId)}</span>
                    </div>
                    <div style={{ display: 'grid', gap: 4 }}>
                      <span style={{ fontSize: 12, color: '#475569' }}>Valid until</span>
                      <span style={{ fontSize: 13, color: '#0f172a' }}>{formatDate(selectedPrescription.valid_until || selectedPrescription.expiryDate)}</span>
                    </div>
                  </div>
                </div>

                <div style={{ display: 'grid', gap: 10, padding: 16, borderRadius: 18, background: '#fff', border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: 13, fontWeight: 700, color: '#0f172a' }}>Medications</div>
                  {(selectedPrescription.items || []).map((item, index) => (
                    <div key={item._id || index} style={{ display: 'grid', gap: 4, padding: 12, borderRadius: 14, background: '#f8fafc' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, alignItems: 'center' }}>
                        <span style={{ fontSize: 14, fontWeight: 700, color: '#0f172a' }}>{valueOrNA(item.medication_name)}</span>
                        <span style={{ fontSize: 12, color: '#64748b' }}>{valueOrNA(item.status)}</span>
                      </div>
                      <div style={{ display: 'grid', gap: 4, fontSize: 12, color: '#475569' }}>
                        <div>Dosage: {valueOrNA(item.dosage || item.strength)}</div>
                        <div>Frequency: {valueOrNA(item.frequency)}</div>
                        <div>Route: {valueOrNA(item.route)}</div>
                        <div>Quantity: {valueOrNA(item.quantity)}</div>
                        <div>Indication: {valueOrNA(item.indication)}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div style={{ color: '#475569', padding: 24 }}>Select an order from the left panel to inspect the physician pharmacy handoff and open the detailed prescription report.</div>
            )}
          </div>
        </div>
      </div>
    </SidebarLayout>
  );
};

export default OrderedMeds;
