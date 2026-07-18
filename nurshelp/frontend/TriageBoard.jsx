// src/Dashboard/TriageBoard.jsx
import React, { useEffect, useState } from "react";
const BACKEND_URL = "http://localhost:5001";

export default function TriageBoard() {
  const [patients, setPatients] = useState([]);

  useEffect(() => {
    let mounted = true;
    const fetchBoard = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/api/queue`);
        const data = await res.json();
        if (mounted) setPatients(data);
      } catch (err) {
        console.error("queue fetch error", err);
      }
    };
    fetchBoard();
    const iv = setInterval(fetchBoard, 5000);
    return () => { mounted = false; clearInterval(iv); };
  }, []);

  const colorFor = (satsLevel) => {
    // SATS Color Mapping: Red, Orange, Yellow, Green
    if (satsLevel === 'red' || satsLevel === 1) return "#ef4444"; // Critical/High Risk
    if (satsLevel === 'orange' || satsLevel === 2) return "#f97316"; // Very Urgent
    if (satsLevel === 'yellow' || satsLevel === 3) return "#eab308"; // Urgent
    if (satsLevel === 'green' || satsLevel === 4) return "#22c55e"; // Routine
    return "#6c757d"; // Default gray if unknown
  };

  return (
    <div style={{ padding: 20 }}>
      <h3>Triage Queue</h3>
      <div style={{ display: "grid", gap: 10 }}>
        {patients.map(p => (
          <div key={p.id || p._id} style={{ background: colorFor(p.triage?.satsLevel || 'green'), color: "white", padding: 12, borderRadius: 6 }}>
            <div style={{ fontWeight: "bold" }}>{p.patientName}</div>
            <div>SATS: {p.triage?.satsLevel?.toUpperCase() || 'GREEN'}</div>
            <div style={{ opacity: 0.9 }}>{p.triage?.specialty || 'General'}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
