import React, { useEffect, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { FaBed, FaSync, FaClock, FaTools, FaCheckCircle } from "react-icons/fa";
import SidebarLayout from "./SidebarLayout";
import { useUserFromToken } from "../hooks/useUserFromToken";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

/**
 * BedManagementPage - Display beds organized by wards with live status monitoring
 * 
 * Features:
 * - Ward-based organization
 * - Real-time bed status (Available, Occupied, Cleaning, Maintenance)
 * - Occupancy rate tracking
 * - Patient information for occupied beds
 * - Status duration tracking
 * - Color-coded status indicators
 */
const BedManagementPage = ({ navigate, handleLogout }) => {
  const [wards, setWards] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedWard, setSelectedWard] = useState(null);
  const [filter, setFilter] = useState("all"); // all, available, occupied, cleaning, maintenance
  const [lastUpdated, setLastUpdated] = useState(null);
  const pollRef = useRef(null);
  const POLL_INTERVAL = 10000; // Poll every 10 seconds

  const { userName } = useUserFromToken();

  // Fetch ward data with bed status
  const fetchWardsData = async () => {
    try {
      setLoading(true);
      const token = localStorage.getItem("token");
      if (!token) throw new Error("Authentication required to load wards");
      const response = await fetch(`${BACKEND_URL}/api/beds/wards`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      if (!response.ok) throw new Error("Failed to fetch wards");
      
      const data = await response.json();
      if (data.success && data.data) {
        setWards(data.data);
        setLastUpdated(new Date());
      }
    } catch (error) {
      console.error("Error fetching ward data:", error);
    } finally {
      setLoading(false);
    }
  };

  // Initialize and setup polling
  useEffect(() => {
    fetchWardsData();
    
    pollRef.current = setInterval(() => {
      fetchWardsData();
    }, POLL_INTERVAL);

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // Status color mapping
  const getStatusColor = (status) => {
    switch (status) {
      case "AVAILABLE":
        return { bg: "#d1fae5", border: "#10b981", text: "#065f46", icon: "✓" };
      case "OCCUPIED":
        return { bg: "#fef3c7", border: "#f59e0b", text: "#78350f", icon: "👤" };
      case "CLEANING":
        return { bg: "#dbeafe", border: "#3b82f6", text: "#1e3a8a", icon: "🧹" };
      case "MAINTENANCE":
        return { bg: "#fee2e2", border: "#ef4444", text: "#7f1d1d", icon: "🔧" };
      default:
        return { bg: "#f3f4f6", border: "#9ca3af", text: "#374151", icon: "?" };
    }
  };

  // Format duration
  const formatDuration = (startTime) => {
    if (!startTime) return "N/A";
    const diff = Math.floor((new Date() - new Date(startTime)) / 60000);
    if (diff < 60) return `${diff}m`;
    const hours = Math.floor(diff / 60);
    return `${hours}h ${diff % 60}m`;
  };

  // Filter beds based on selected filter
  const filterBeds = (beds) => {
    if (filter === "all") return beds;
    return beds.filter(b => b.status === filter);
  };

  // Determine which wards to show
  const displayWards = selectedWard 
    ? wards.filter(w => w._id === selectedWard) 
    : wards;

  return (
    <SidebarLayout username={userName} handleLogout={handleLogout}>
      <div style={{ padding: "20px", maxWidth: "1400px", margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: "30px" }}>
          <h1 style={{ fontSize: "28px", fontWeight: "700", marginBottom: "8px" }}>
            <FaBed style={{ marginRight: "10px" }} />
            Bed Management System
          </h1>
          <p style={{ color: "#666", marginBottom: "15px" }}>
            {wards.length} wards • Real-time bed status monitoring
          </p>

          {/* Controls */}
          <div style={{ display: "flex", gap: "15px", alignItems: "center", flexWrap: "wrap" }}>
            {/* Ward selector */}
            <select
              value={selectedWard || ""}
              onChange={(e) => setSelectedWard(e.target.value || null)}
              style={{
                padding: "8px 12px",
                borderRadius: "6px",
                border: "1px solid #ddd",
                fontSize: "14px",
                cursor: "pointer"
              }}
            >
              <option value="">All Wards</option>
              {wards.map(w => (
                <option key={w._id} value={w._id}>
                  {w.wardName || w.department} ({w.totalBeds} beds)
                </option>
              ))}
            </select>

            {/* Status filter */}
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              style={{
                padding: "8px 12px",
                borderRadius: "6px",
                border: "1px solid #ddd",
                fontSize: "14px",
                cursor: "pointer"
              }}
            >
              <option value="all">All Status</option>
              <option value="AVAILABLE">Available Only</option>
              <option value="OCCUPIED">Occupied Only</option>
              <option value="CLEANING">Cleaning Only</option>
              <option value="MAINTENANCE">Maintenance Only</option>
            </select>

            {/* Refresh button */}
            <button
              onClick={fetchWardsData}
              disabled={loading}
              style={{
                padding: "8px 16px",
                borderRadius: "6px",
                border: "none",
                background: "#3b82f6",
                color: "white",
                cursor: loading ? "not-allowed" : "pointer",
                display: "flex",
                alignItems: "center",
                gap: "6px",
                opacity: loading ? 0.6 : 1
              }}
              title="Refresh bed status"
            >
              <FaSync style={{ animation: loading ? "spin 1s linear infinite" : "none" }} />
              Refresh
            </button>

            {/* Last updated */}
            {lastUpdated && (
              <div style={{ fontSize: "12px", color: "#666" }}>
                Updated: {lastUpdated.toLocaleTimeString()}
              </div>
            )}
          </div>
        </div>

        {/* Add CSS for loading animation */}
        <style>{`
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
        `}</style>

        {/* Ward Cards */}
        <div style={{ display: "grid", gap: "20px" }}>
          {displayWards.map(ward => (
            <div
              key={ward._id}
              style={{
                border: "1px solid #e5e7eb",
                borderRadius: "8px",
                overflow: "hidden",
                background: "white",
                boxShadow: "0 1px 3px rgba(0,0,0,0.1)"
              }}
            >
              {/* Ward Header */}
              <div style={{
                padding: "16px",
                background: "#f9fafb",
                borderBottom: "1px solid #e5e7eb",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center"
              }}>
                <div>
                  <h2 style={{ fontSize: "18px", fontWeight: "700", margin: "0 0 6px 0" }}>
                    {ward.wardName || ward.department}
                  </h2>
                  <div style={{ display: "flex", gap: "20px", fontSize: "13px", color: "#666" }}>
                    <span>Total Beds: <strong>{ward.totalBeds}</strong></span>
                    <span>Occupancy: <strong>{ward.occupancyRate}%</strong></span>
                  </div>
                </div>

                {/* Status Summary */}
                <div style={{
                  display: "flex",
                  gap: "20px",
                  padding: "10px 15px",
                  background: "white",
                  borderRadius: "6px",
                  border: "1px solid #e5e7eb"
                }}>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ fontSize: "20px", fontWeight: "700", color: "#10b981" }}>
                      {ward.currentStatus.available}
                    </div>
                    <div style={{ fontSize: "11px", color: "#666" }}>Available</div>
                  </div>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ fontSize: "20px", fontWeight: "700", color: "#f59e0b" }}>
                      {ward.currentStatus.occupied}
                    </div>
                    <div style={{ fontSize: "11px", color: "#666" }}>Occupied</div>
                  </div>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ fontSize: "20px", fontWeight: "700", color: "#3b82f6" }}>
                      {ward.currentStatus.cleaning}
                    </div>
                    <div style={{ fontSize: "11px", color: "#666" }}>Cleaning</div>
                  </div>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ fontSize: "20px", fontWeight: "700", color: "#ef4444" }}>
                      {ward.currentStatus.maintenance}
                    </div>
                    <div style={{ fontSize: "11px", color: "#666" }}>Maintenance</div>
                  </div>
                </div>
              </div>

              {/* Occupancy Bar */}
              <div style={{
                height: "6px",
                background: "#e5e7eb",
                overflow: "hidden"
              }}>
                <div style={{
                  height: "100%",
                  background: `hsl(${Math.max(0, 100 - ward.occupancyRate)}, 100%, 50%)`,
                  width: `${ward.occupancyRate}%`,
                  transition: "width 0.3s ease"
                }} />
              </div>

              {/* Beds Grid */}
              <div style={{
                padding: "16px",
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
                gap: "12px"
              }}>
                {filterBeds(ward.bedsLayout || []).map(bed => {
                  const statusColor = getStatusColor(bed.status);
                  return (
                    <div
                      key={bed._id}
                      style={{
                        border: `2px solid ${statusColor.border}`,
                        background: statusColor.bg,
                        borderRadius: "6px",
                        padding: "12px",
                        cursor: "pointer",
                        transition: "all 0.2s",
                        position: "relative"
                      }}
                      onMouseOver={(e) => {
                        e.currentTarget.style.boxShadow = "0 4px 12px rgba(0,0,0,0.15)";
                        e.currentTarget.style.transform = "translateY(-2px)";
                      }}
                      onMouseOut={(e) => {
                        e.currentTarget.style.boxShadow = "none";
                        e.currentTarget.style.transform = "translateY(0)";
                      }}
                    >
                      {/* Status Badge */}
                      <div style={{
                        position: "absolute",
                        top: "8px",
                        right: "8px",
                        fontSize: "20px"
                      }}>
                        {statusColor.icon}
                      </div>

                      {/* Bed Info */}
                      <div style={{ fontSize: "14px", fontWeight: "700", color: statusColor.text, marginBottom: "6px" }}>
                        Bed {bed.bedId}
                      </div>

                      {/* Status */}
                      <div style={{
                        fontSize: "12px",
                        color: statusColor.text,
                        marginBottom: "6px",
                        fontWeight: "600"
                      }}>
                        {bed.status}
                      </div>

                      {/* Bed Type */}
                      <div style={{
                        fontSize: "11px",
                        color: statusColor.text,
                        marginBottom: "6px"
                      }}>
                        {bed.bedType || "Standard"}
                      </div>

                      {/* Duration */}
                      {(bed.status === "OCCUPIED" || bed.status === "CLEANING") && bed.occupancyStartTime && (
                        <div style={{
                          fontSize: "11px",
                          color: statusColor.text,
                          marginTop: "6px",
                          paddingTop: "6px",
                          borderTop: `1px solid ${statusColor.border}`,
                          display: "flex",
                          alignItems: "center",
                          gap: "4px"
                        }}>
                          <FaClock size={10} />
                          {formatDuration(bed.occupancyStartTime)}
                        </div>
                      )}

                      {/* Expected Ready Time */}
                      {bed.status === "CLEANING" && bed.estimatedReadyTime && (
                        <div style={{
                          fontSize: "10px",
                          color: statusColor.text,
                          marginTop: "4px",
                          fontStyle: "italic"
                        }}>
                          Ready: {new Date(bed.estimatedReadyTime).toLocaleTimeString()}
                        </div>
                      )}

                      {/* Acuity Level */}
                      {bed.acuityLevel && (
                        <div style={{
                          fontSize: "10px",
                          color: statusColor.text,
                          marginTop: "4px",
                          background: "rgba(0,0,0,0.05)",
                          padding: "2px 4px",
                          borderRadius: "3px",
                          display: "inline-block"
                        }}>
                          {bed.acuityLevel}
                        </div>
                      )}
                    </div>
                  );
                })}

                {/* No beds message */}
                {(!ward.bedsLayout || ward.bedsLayout.length === 0) && (
                  <div style={{ gridColumn: "1/-1", textAlign: "center", padding: "20px", color: "#999" }}>
                    No beds found
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Loading State */}
        {loading && (
          <div style={{
            position: "fixed",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            background: "white",
            padding: "30px",
            borderRadius: "8px",
            boxShadow: "0 4px 6px rgba(0,0,0,0.1)",
            textAlign: "center"
          }}>
            <FaSync style={{ fontSize: "30px", marginBottom: "10px", animation: "spin 1s linear infinite" }} />
            <p>Loading bed status...</p>
          </div>
        )}

        {/* Empty State */}
        {!loading && displayWards.length === 0 && (
          <div style={{
            textAlign: "center",
            padding: "40px",
            color: "#999"
          }}>
            <FaBed style={{ fontSize: "40px", marginBottom: "10px", opacity: 0.5 }} />
            <p>No wards found</p>
          </div>
        )}
      </div>
    </SidebarLayout>
  );
};

export default BedManagementPage;
