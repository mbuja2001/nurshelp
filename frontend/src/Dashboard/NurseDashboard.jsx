import React, { useEffect } from "react";
import { styles } from "./styles";

const NurseDashboard = ({
  navigate = () => {},
  setView = () => {},
  username = "Staff",
  patients = [],
  bedsAvailable = 0,
  sortedPatients = [],
  toggleAssigned = () => {},
  getWaitTime = () => "N/A",
  handleLogout = () => {}
}) => {

  useEffect(() => {
    if (!username) {
      setView("login");
      navigate("/login");
    }
  }, [username, navigate, setView]);

  const handleNewIntake = () => navigate("/summary/new");

  const severityColor = (sev) => {
    if (sev === 3) return "#ef4444";
    if (sev === 2) return "#f59e0b";
    return "#10b981";
  };

  const pendingPatients = sortedPatients.filter(p => (p.isWaiting === true) || (p.status === "pending"));
  const confirmedPatients = sortedPatients.filter(p => p.status === "confirmed");

  return (
    <div style={{ fontFamily: "sans-serif", backgroundColor: "#fff", minHeight: "100vh" }}>
      <header style={styles.header}>
        <h2 style={{ margin: 0 }}>Nurse Helpdesk</h2>
        <div style={{ display: "flex", gap: "15px", alignItems: "center" }}>
          <span>Nurse {username}</span>
          <button onClick={() => { localStorage.removeItem("token"); handleLogout(); }} style={styles.btnLogout}>Log out</button>
        </div>
      </header>

      <main style={{ padding: "40px 8%" }}>
        <div style={styles.statGrid}>
          {[
            ["Active", patients.length],
            ["Pending", pendingPatients.length],
            ["Beds", bedsAvailable]
          ].map(([label, count]) => (
            <div key={label} style={styles.statCard}>
              <p>{label}</p>
              <h1 style={{ fontSize: "40px" }}>{count}</h1>
            </div>
          ))}
        </div>

        {/* Pending / Waiting */}
        <div style={styles.queueBox}>
          <h3>Pending / Waiting Patients</h3>
          <table style={styles.table}>
            <thead>
              <tr style={styles.tableHeadRow}>
                <th>ID</th>
                <th>Wait</th>
                <th>Severity</th>
                <th>Status</th>
                <th style={{ textAlign: "right" }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {pendingPatients.map(p => (
                <tr key={p._id || p.id} style={styles.tableRow}>
                  <td>#{String(p.id || (p._id || "")).padStart(3, "0")}</td>
                  <td>{getWaitTime(p)}</td>
                  <td style={{ fontWeight: "bold", color: severityColor(p.severity || p.triage?.severity) }}>
                    {(p.severity || p.triage?.severity) === 3 ? "High" : ((p.severity || p.triage?.severity) === 2 ? "Med" : "Low")}
                  </td>
                  <td>
                    <input type="checkbox" checked={!!p.assigned} onChange={() => toggleAssigned(p._id || p.id)} />
                    {p.assigned ? " Assigned" : ` ${p.status || (p.isWaiting ? "waiting" : "pending")}`}
                  </td>
                  <td style={{ textAlign: "right" }}>
                    <button onClick={() => navigate(`/summary/${p._id || p.id}`)} style={{ ...styles.btnLarge, padding: "10px", fontSize: "12px", width: "auto" }}>
                      Summary
                    </button>
                  </td>
                </tr>
              ))}
              {pendingPatients.length === 0 && (
                <tr><td colSpan={5} style={{ textAlign: "center", padding: 12, color: "#777" }}>No waiting patients</td></tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Confirmed */}
        {confirmedPatients.length > 0 && (
          <div style={{ ...styles.queueBox, marginTop: 40 }}>
            <h3>Confirmed Patient Reports</h3>
            <table style={styles.table}>
              <thead>
                <tr style={styles.tableHeadRow}>
                  <th>ID</th>
                  <th>Patient</th>
                  <th>Severity</th>
                  <th>Nurse Notes</th>
                  <th style={{ textAlign: "right" }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {confirmedPatients.map(p => (
                  <tr key={p._id || p.id} style={styles.tableRow}>
                    <td>#{String(p.id || (p._id || "")).padStart(3, "0")}</td>
                    <td>{p.patient?.name ?? "N/A"}</td>
                    <td style={{ fontWeight: "bold", color: severityColor(p.severity || (p.triage?.severity)) }}>
                      {(p.severity || p.triage?.severity) === 3 ? "High" : ((p.severity || p.triage?.severity) === 2 ? "Med" : "Low")}
                    </td>
                    <td>{p.nurseNotes ?? ""}</td>
                    <td style={{ textAlign: "right" }}>
                      <button onClick={() => navigate(`/summary/${p._id || p.id}`)} style={{ ...styles.btnLarge, padding: "10px", fontSize: "12px", width: "auto" }}>
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <div style={styles.actionBox}>
          <button onClick={handleNewIntake} style={styles.btnLarge}>New Intake</button>
        </div>
      </main>
    </div>
  );
};

export default NurseDashboard;