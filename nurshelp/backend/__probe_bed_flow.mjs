import fetch from 'node-fetch';

const API_BASE = 'http://localhost:5001/api';
const payload = {
  hospitalName: 'ProbeBedFlow',
  managerName: 'Admin',
  managerSurname: 'Tester',
  managerEmail: 'probe-bed-flow@test.local',
  managerPassword: 'Admin@123456',
  department: 'All'
};

const reg = await fetch(`${API_BASE}/hospital/register`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload)
});
const regJson = await reg.json();
console.log('REGISTER_JSON', JSON.stringify(regJson, null, 2));

if (!regJson.success || !regJson.token) {
  console.log('ABORT: registration failed');
  process.exit(1);
}

const token = regJson.token;
const beds = [
  { bed_id: 'PROBE-BED-01', ward_unit: 'ICU', equipment_baseline: ['High-Acuity Monitor', 'Mechanical Ventilator'] },
  { bed_id: 'PROBE-BED-02', ward_unit: 'Medical Ward', equipment_baseline: ['Oxygen Delivery'] }
];

const resp = await fetch(`${API_BASE}/beds/batch`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${token}`
  },
  body: JSON.stringify({ beds })
});
const text = await resp.text();
console.log('BATCH_STATUS', resp.status);
console.log('BATCH_TEXT', text);
