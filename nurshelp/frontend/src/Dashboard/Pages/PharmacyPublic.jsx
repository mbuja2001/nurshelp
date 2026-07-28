import React, { useEffect, useState } from 'react';
import SidebarLayout from '../SidebarLayout';
import { FaPills } from 'react-icons/fa';

const PharmacyPublic = () => {
  const [inventory, setInventory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [requesting, setRequesting] = useState(false);
  const [selectedMed, setSelectedMed] = useState(null);
  const token = localStorage.getItem('token');

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';
        const res = await fetch(`${BACKEND_URL}/api/pharmacy/inventory`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        if (!res.ok) throw new Error('Failed to load inventory');
        const data = await res.json();
        setInventory(Array.isArray(data) ? data : (data.data || []));
      } catch (err) {
        setError(err.message || 'Error');
      } finally { setLoading(false); }
    };
    load();
  }, [token]);

  const sendRequest = async (med) => {
    setRequesting(true);
    try {
      const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';
      const res = await fetch(`${BACKEND_URL}/api/pharmacy/requests`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ medicationId: med.id || med._id, medicationName: med.name || med.medicationName, quantity: 1 })
      });
      if (!res.ok) throw new Error('Failed to create request');
      setSelectedMed(null);
      alert('Request sent to pharmacy');
    } catch (err) {
      alert('Error sending request: ' + (err.message || 'unknown'));
    } finally { setRequesting(false); }
  };

  return (
    <SidebarLayout>
      <div style={{ padding: 24 }}>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 12 }}>
          <div style={{ width: 44, height: 44, borderRadius: 8, background: '#ecfdf5', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <FaPills size={20} style={{ color: '#059669' }} />
          </div>
          <div>
            <h2 style={{ margin: 0 }}>Pharmacy Inventory</h2>
            <div style={{ fontSize: 12, color: '#64748b' }}>Browse stock and request items from the pharmacy</div>
          </div>
        </div>

        {loading ? (<div>Loading inventory…</div>) : null}
        {error && (<div style={{ color: 'red' }}>{error}</div>)}

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 12 }}>
          {inventory.map(m => (
            <div key={m.id || m._id} style={{ background: '#fff', padding: 12, borderRadius: 8, border: '1px solid #e5e7eb' }}>
              <div style={{ fontWeight: 700 }}>{m.name || m.medicationName}</div>
              <div style={{ fontSize: 12, color: '#64748b' }}>Stock: {m.quantity ?? m.stock ?? 0}</div>
              <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
                <button onClick={() => setSelectedMed(m)} style={{ padding: '8px 10px', borderRadius: 8, border: 'none', background: '#059669', color: '#fff' }}>Request</button>
                <button onClick={() => navigator.clipboard.writeText(m.batchNumber || '')} style={{ padding: '8px 10px', borderRadius: 8, border: '1px solid #e2e8f0', background: '#fff' }}>Copy batch</button>
              </div>
            </div>
          ))}
        </div>

        {selectedMed && (
          <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.4)', display: 'flex', alignItems: 'center', justifyContent: 'center' }} onClick={() => setSelectedMed(null)}>
            <div style={{ background: '#fff', padding: 20, borderRadius: 8, width: 420 }} onClick={e => e.stopPropagation()}>
              <h3 style={{ marginTop: 0 }}>{selectedMed.name || selectedMed.medicationName}</h3>
              <div style={{ marginBottom: 12 }}>Request this medication from pharmacy. Quantity:</div>
              <div style={{ display: 'flex', gap: 8 }}>
                <input defaultValue={1} type="number" min={1} style={{ width: 80, padding: 8, borderRadius: 6, border: '1px solid #e2e8f0' }} id="pharm-req-qty" />
                <button onClick={() => sendRequest(selectedMed)} disabled={requesting} style={{ padding: '8px 12px', background: '#059669', color: '#fff', border: 'none', borderRadius: 6 }}>Send Request</button>
                <button onClick={() => setSelectedMed(null)} style={{ padding: '8px 12px', background: '#f3f4f6', border: 'none', borderRadius: 6 }}>Cancel</button>
              </div>
            </div>
          </div>
        )}
      </div>
    </SidebarLayout>
  );
};

export default PharmacyPublic;
