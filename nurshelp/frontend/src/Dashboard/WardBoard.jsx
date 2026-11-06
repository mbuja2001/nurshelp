// src/Dashboard/WardBoard.jsx
import React, { useState, useEffect } from 'react';

const BACKEND_URL = "http://localhost:5001";

const statusColors = {
  AVAILABLE: { bg: '#d4edda', border: '#28a745', text: '#155724', icon: '🟢' },
  OCCUPIED: { bg: '#f8d7da', border: '#dc3545', text: '#721c24', icon: '🔴' },
  CLEANING: { bg: '#fff3cd', border: '#ffc107', text: '#856404', icon: '🟡' },
  MAINTENANCE: { bg: '#d1ecf1', border: '#17a2b8', text: '#0c5460', icon: '🔵' }
};

export default function WardBoard({ token }) {
  const [wards, setWards] = useState([]);
  const [selectedWard, setSelectedWard] = useState(null);
  const [wardDetails, setWardDetails] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedBed, setSelectedBed] = useState(null);
  const [statusFilter, setStatusFilter] = useState("ALL");

  // Fetch all wards on mount
  useEffect(() => {
    fetchWards();
    const interval = setInterval(fetchWards, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, []);

  // Fetch selected ward details
  useEffect(() => {
    if (selectedWard) {
      fetchWardDetails(selectedWard);
      const interval = setInterval(() => fetchWardDetails(selectedWard), 5000);
      return () => clearInterval(interval);
    }
  }, [selectedWard]);

  const fetchWards = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/beds/wards`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
      if (data.success) {
        setWards(data.data);
        setError(null);
      }
    } catch (err) {
      setError(err.message);
    }
  };

  const fetchWardDetails = async (wardId) => {
    try {
      setLoading(true);
      const res = await fetch(`${BACKEND_URL}/api/beds/ward/${wardId}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
      if (data.success) {
        setWardDetails(data.data);
        setError(null);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleBedStatusChange = async (bedId, newStatus) => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/beds/${bedId}/status`, {
        method: 'PATCH',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ status: newStatus })
      });
      const data = await res.json();
      if (data.success) {
        fetchWardDetails(selectedWard);
        setSelectedBed(null);
      } else {
        setError(data.error);
      }
    } catch (err) {
      setError(err.message);
    }
  };

  const getFilteredBeds = () => {
    if (!wardDetails) return [];
    if (statusFilter === "ALL") return wardDetails.beds;
    return wardDetails.beds.filter(b => b.status === statusFilter);
  };

  const filteredBeds = getFilteredBeds();

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Ward Management & Live Bed Status</h1>

      {error && (
        <div style={styles.error}>
          ⚠️ {error}
        </div>
      )}

      <div style={styles.mainLayout}>
        {/* Ward Selector */}
        <div style={styles.wardList}>
          <h3 style={styles.sectionTitle}>Wards</h3>
          <div style={styles.wardCards}>
            {wards.map(ward => (
              <div
                key={ward._id}
                onClick={() => setSelectedWard(ward._id)}
                style={{
                  ...styles.wardCard,
                  ...(selectedWard === ward._id ? styles.wardCardActive : {}),
                  cursor: 'pointer'
                }}
              >
                <div style={styles.wardCardTitle}>{ward.wardName}</div>
                <div style={styles.wardCardContent}>
                  <div><span style={styles.label}>Total Beds:</span> {ward.totalBeds}</div>
                  <div style={styles.statusSummary}>
                    <span style={{ color: '#28a745' }}>🟢 {ward.currentStatus?.available || 0}</span>
                    <span style={{ color: '#dc3545' }}>🔴 {ward.currentStatus?.occupied || 0}</span>
                    <span style={{ color: '#ffc107' }}>🟡 {ward.currentStatus?.cleaning || 0}</span>
                  </div>
                  <div style={styles.occupancyBar}>
                    <div
                      style={{
                        ...styles.occupancyFill,
                        width: `${ward.occupancyRate}%`,
                        backgroundColor: ward.occupancyRate > 80 ? '#dc3545' : '#28a745'
                      }}
                    />
                  </div>
                  <div style={styles.occupancyRate}>{ward.occupancyRate}% occupied</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Bed Details */}
        {selectedWard && wardDetails && (
          <div style={styles.bedPanel}>
            <h2 style={styles.wardTitle}>{wardDetails.ward.wardName}</h2>

            {/* Status Filter */}
            <div style={styles.filterBar}>
              <button
                onClick={() => setStatusFilter("ALL")}
                style={{
                  ...styles.filterBtn,
                  ...(statusFilter === "ALL" ? styles.filterBtnActive : {})
                }}
              >
                All ({wardDetails.beds.length})
              </button>
              <button
                onClick={() => setStatusFilter("AVAILABLE")}
                style={{
                  ...styles.filterBtn,
                  ...(statusFilter === "AVAILABLE" ? styles.filterBtnActive : {})
                }}
              >
                🟢 Available ({wardDetails.statusCounts.available})
              </button>
              <button
                onClick={() => setStatusFilter("OCCUPIED")}
                style={{
                  ...styles.filterBtn,
                  ...(statusFilter === "OCCUPIED" ? styles.filterBtnActive : {})
                }}
              >
                🔴 Occupied ({wardDetails.statusCounts.occupied})
              </button>
              <button
                onClick={() => setStatusFilter("CLEANING")}
                style={{
                  ...styles.filterBtn,
                  ...(statusFilter === "CLEANING" ? styles.filterBtnActive : {})
                }}
              >
                🟡 Cleaning ({wardDetails.statusCounts.cleaning})
              </button>
            </div>

            {/* Bed Grid */}
            <div style={styles.bedGrid}>
              {filteredBeds.map(bed => {
                const color = statusColors[bed.status];
                const isSelected = selectedBed?._id === bed._id;

                return (
                  <div
                    key={bed._id}
                    onClick={() => setSelectedBed(isSelected ? null : bed)}
                    style={{
                      ...styles.bedCard,
                      backgroundColor: color.bg,
                      borderColor: color.border,
                      borderWidth: isSelected ? '3px' : '2px',
                      cursor: 'pointer',
                      boxShadow: isSelected ? `0 0 10px ${color.border}` : 'none'
                    }}
                  >
                    <div style={styles.bedHeader}>
                      <div style={styles.bedId}>{color.icon} {bed.bedId}</div>
                      <div style={styles.bedStatus}>{bed.status}</div>
                    </div>

                    <div style={styles.bedInfo}>
                      <div style={styles.bedDetail}>
                        <span style={styles.label}>Type:</span> {bed.bedType}
                      </div>
                      {bed.acuityLevel && (
                        <div style={styles.bedDetail}>
                          <span style={styles.label}>Acuity:</span> {bed.acuityLevel}
                        </div>
                      )}
                      {bed.status === "OCCUPIED" && bed.occupancyStartTime && (
                        <div style={styles.bedDetail}>
                          <span style={styles.label}>Since:</span>{' '}
                          {new Date(bed.occupancyStartTime).toLocaleTimeString()}
                        </div>
                      )}
                      {bed.status === "CLEANING" && bed.estimatedReadyTime && (
                        <div style={styles.bedDetail}>
                          <span style={styles.label}>Ready:</span>{' '}
                          {new Date(bed.estimatedReadyTime).toLocaleTimeString()}
                        </div>
                      )}
                    </div>

                    {/* Quick Status Update */}
                    {isSelected && (
                      <div style={styles.bedActions}>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleBedStatusChange(bed._id, "AVAILABLE");
                          }}
                          style={styles.actionBtn}
                        >
                          ✅ Available
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleBedStatusChange(bed._id, "CLEANING");
                          }}
                          style={styles.actionBtn}
                        >
                          🧹 Cleaning
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleBedStatusChange(bed._id, "MAINTENANCE");
                          }}
                          style={styles.actionBtn}
                        >
                          🔧 Maintenance
                        </button>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {filteredBeds.length === 0 && (
              <div style={styles.emptyState}>
                No beds with status "{statusFilter}"
              </div>
            )}
          </div>
        )}

        {!selectedWard && (
          <div style={styles.emptyState}>
            Select a ward to view and manage beds
          </div>
        )}
      </div>
    </div>
  );
}

const styles = {
  container: {
    padding: '20px',
    backgroundColor: '#f5f5f5',
    minHeight: '100vh'
  },
  title: {
    textAlign: 'center',
    color: '#333',
    marginBottom: '20px'
  },
  error: {
    backgroundColor: '#f8d7da',
    color: '#721c24',
    padding: '12px',
    borderRadius: '4px',
    marginBottom: '16px',
    border: '1px solid #f5c6cb'
  },
  mainLayout: {
    display: 'grid',
    gridTemplateColumns: '300px 1fr',
    gap: '20px'
  },
  wardList: {
    backgroundColor: 'white',
    padding: '16px',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
  },
  sectionTitle: {
    marginTop: 0,
    marginBottom: '12px',
    color: '#333'
  },
  wardCards: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px'
  },
  wardCard: {
    padding: '12px',
    border: '1px solid #ddd',
    borderRadius: '6px',
    backgroundColor: '#f9f9f9',
    transition: 'all 0.2s ease'
  },
  wardCardActive: {
    backgroundColor: '#e3f2fd',
    borderColor: '#2196F3',
    borderWidth: '2px'
  },
  wardCardTitle: {
    fontWeight: 'bold',
    marginBottom: '8px',
    color: '#333'
  },
  wardCardContent: {
    fontSize: '12px',
    color: '#666'
  },
  label: {
    fontWeight: 'bold',
    marginRight: '4px'
  },
  statusSummary: {
    display: 'flex',
    gap: '8px',
    marginTop: '4px',
    fontSize: '11px'
  },
  occupancyBar: {
    width: '100%',
    height: '6px',
    backgroundColor: '#eee',
    borderRadius: '3px',
    marginTop: '4px',
    overflow: 'hidden'
  },
  occupancyFill: {
    height: '100%',
    transition: 'width 0.3s ease'
  },
  occupancyRate: {
    marginTop: '4px',
    fontSize: '11px',
    fontWeight: 'bold'
  },
  bedPanel: {
    backgroundColor: 'white',
    padding: '20px',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
  },
  wardTitle: {
    marginTop: 0,
    marginBottom: '16px',
    color: '#333'
  },
  filterBar: {
    display: 'flex',
    gap: '8px',
    marginBottom: '20px',
    flexWrap: 'wrap'
  },
  filterBtn: {
    padding: '8px 12px',
    border: '1px solid #ddd',
    borderRadius: '4px',
    backgroundColor: 'white',
    cursor: 'pointer',
    fontSize: '12px',
    transition: 'all 0.2s ease'
  },
  filterBtnActive: {
    backgroundColor: '#2196F3',
    color: 'white',
    borderColor: '#2196F3'
  },
  bedGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
    gap: '12px'
  },
  bedCard: {
    padding: '12px',
    borderRadius: '6px',
    transition: 'all 0.2s ease',
    minHeight: '140px',
    display: 'flex',
    flexDirection: 'column'
  },
  bedHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    marginBottom: '8px'
  },
  bedId: {
    fontWeight: 'bold',
    fontSize: '14px'
  },
  bedStatus: {
    fontSize: '11px',
    fontWeight: 'bold'
  },
  bedInfo: {
    flex: 1,
    fontSize: '12px',
    marginBottom: '8px'
  },
  bedDetail: {
    marginBottom: '4px',
    wordBreak: 'break-word'
  },
  bedActions: {
    display: 'flex',
    gap: '4px',
    flexWrap: 'wrap'
  },
  actionBtn: {
    padding: '4px 8px',
    fontSize: '11px',
    border: 'none',
    borderRadius: '3px',
    backgroundColor: '#2196F3',
    color: 'white',
    cursor: 'pointer',
    flex: 1,
    minWidth: '60px'
  },
  emptyState: {
    padding: '40px 20px',
    textAlign: 'center',
    color: '#999',
    fontSize: '16px'
  }
};
