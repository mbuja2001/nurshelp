// src/Dashboard/styles.jsx
export const styles = {
  patientBg: { height: '100vh', backgroundColor: '#0f172a', color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center' },
  hero: { maxWidth: '800px' },
  loginCard: { backgroundColor: '#1e293b', padding: '40px', borderRadius: '15px', width: '380px', boxShadow: '0 25px 50px -12px rgba(15, 23, 42, 0.8)' },
  formGroup: { marginBottom: '16px', display: 'flex', flexDirection: 'column', gap: '6px' },
  formLabel: { 
    fontSize: '13px', 
    fontWeight: '600', 
    color: '#38bdf8',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
    display: 'flex',
    alignItems: 'center',
    gap: '4px'
  },
  loginInput: { 
    width: '100%', 
    padding: '12px 14px', 
    borderRadius: '8px', 
    border: '2px solid #334155',
    boxSizing: 'border-box',
    backgroundColor: '#0f172a',
    color: '#e2e8f0',
    fontSize: '14px',
    transition: 'all 0.3s ease',
    '::placeholder': { color: '#64748b' }
  },
  loginInputFocus: {
    borderColor: '#38bdf8',
    boxShadow: '0 0 0 3px rgba(56, 189, 248, 0.1)',
    backgroundColor: '#1e293b'
  },
  header: { padding: '1.2200748px 8%', backgroundColor: '#0f172a', color: '#fff', display: 'flex', justifyContent: 'space-between' },
  statGrid: { display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '20px' },
  statCard: { border: '1px solid #e2e8f0', borderRadius: '15px', padding: '20px', textAlign: 'center' },
  queueBox: { marginTop: '30px', border: '1px solid #e2e8f0', padding: '20px', borderRadius: '15px' },
  actionBox: { marginTop: '30px', display: 'flex', justifyContent: 'flex-end' },
  btnWhite: { padding: '15px 30px', borderRadius: '8px', backgroundColor: '#fff', color: '#0f172a', fontWeight: 'bold', cursor: 'pointer', border:'none' },
  btnOutline: { padding: '15px 30px', borderRadius: '8px', border: '1px solid #fff', backgroundColor: 'transparent', color: '#fff', cursor: 'pointer' },
  btnLarge: { backgroundColor: '#0f172a', color: 'white', padding: '15px 30px', borderRadius: '10px', fontWeight: 'bold', cursor: 'pointer', border: 'none' },
  resetBtn: { backgroundColor: '#ef4444', color: 'white', padding: '5px 10px', border: 'none', borderRadius: '5px' },
  btnLogout: { backgroundColor: '#fff', color: '#0f172a', padding: '5px 10px', borderRadius: '5px', border: 'none' },
  table: { width: '100%', borderCollapse: 'collapse', textAlign: 'left' },
  tableHeadRow: { borderBottom: '2px solid #e2e8f0', height: '40px' },
  tableRow: { borderBottom: '1px solid #f1f5f9', height: '50px' },
  textArea: { width: '100%', height: '150px', padding: '10px', boxSizing: 'border-box', borderRadius: '8px' },
  micArea: { border: '2px dashed #cbd5e1', padding: '20px', borderRadius: '10px', textAlign: 'center' },
  macRecorder: { 
    backgroundColor: '#1e293b', 
    width: '300px', 
    padding: '40px 20px', 
    borderRadius: '40px', 
    display: 'flex', 
    flexDirection: 'column', 
    alignItems: 'center',
    boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.3)',
    border: '1px solid #334155'
  },
  waveGroup: { display: 'flex', gap: '8px', marginBottom: '40px', alignItems: 'center', justifyContent: 'center', height: '60px' },
  waveBar: { width: '6px', backgroundColor: '#38bdf8', borderRadius: '10px' },
  stopBtn: { width: '60px', height: '60px', backgroundColor: '#ef4444', border: 'none', borderRadius: '50%', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }
};
