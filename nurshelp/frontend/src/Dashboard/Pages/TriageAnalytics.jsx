import React, { useState, useEffect } from 'react';
import { FiBarChart2, FiPieChart, FiTrendingUp, FiAlertTriangle, FiCheckCircle, FiClock, FiUsers, FiDownload } from 'react-icons/fi';
import { FaHeartbeat } from 'react-icons/fa';
import SidebarLayout from '../SidebarLayout';
import { useUserFromToken } from '../../hooks/useUserFromToken';

const TriageAnalytics = () => {
  const { userName } = useUserFromToken();
  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

  // ============ STATE ============
  const [analytics, setAnalytics] = useState({
    waitTimeVsSeverity: [
      { severity: 'Low', allocated: 120, actual: 95, count: 12 },
      { severity: 'Medium', allocated: 60, actual: 72, count: 18 },
      { severity: 'High', allocated: 15, actual: 28, count: 8 },
      { severity: 'Critical', allocated: 5, actual: 8, count: 3 },
    ],
    diagnoses: [
      { name: 'Chest Pain / Acute Coronary Syndrome', count: 12, percentage: 18 },
      { name: 'Abdominal Pain / Appendicitis', count: 9, percentage: 14 },
      { name: 'Respiratory Issues / Pneumonia', count: 8, percentage: 12 },
      { name: 'Hypertension Crisis', count: 7, percentage: 11 },
      { name: 'Trauma / Fractures', count: 6, percentage: 9 },
      { name: 'Sepsis / Infections', count: 5, percentage: 8 },
      { name: 'Other', count: 19, percentage: 28 },
    ],
    mistriage: [
      { caseId: 'ENC-001', severity: 'High', aiSuggested: 'Chest Pain / ACS', nurseApproved: 'Chest Pain / ACS', physicianAttended: 'Stable Angina', status: 'mistriage', discrepancy: 'Risk Level' },
      { caseId: 'ENC-004', severity: 'Medium', aiSuggested: 'Abdominal Pain', nurseApproved: 'Appendicitis', physicianAttended: 'Appendicitis', status: 'correct', discrepancy: null },
      { caseId: 'ENC-007', severity: 'High', aiSuggested: 'Sepsis', nurseApproved: 'Sepsis', physicianAttended: 'UTI', status: 'mistriage', discrepancy: 'Diagnosis Severity' },
      { caseId: 'ENC-010', severity: 'Critical', aiSuggested: 'Stroke', nurseApproved: 'TIA', physicianAttended: 'Migraine', status: 'mistriage', discrepancy: 'Critical Misdiagnosis' },
      { caseId: 'ENC-013', severity: 'Medium', aiSuggested: 'Respiratory', nurseApproved: 'Respiratory', physicianAttended: 'Respiratory', status: 'correct', discrepancy: null },
    ],
    staffHours: [
      { name: 'Sarah Johnson (ED Nurse)', shift: 12, break: 1, actual: 11 },
      { name: 'Michael Chen (ED Nurse)', shift: 8, break: 0.5, actual: 7.5 },
      { name: 'Dr. Emily Rodriguez (Attending)', shift: 12, break: 1, actual: 11 },
      { name: 'Dr. James Park (Resident)', shift: 10, break: 0.5, actual: 9.5 },
      { name: 'Lisa Wong (Triage Nurse)', shift: 12, break: 1, actual: 11 },
    ],
    dailyMetrics: {
      totalPatients: 65,
      triageAccuracy: 87.7,
      mistriageRate: 12.3,
      avgWaitTime: 45,
      staffEfficiency: 92.1,
      totalStaffHours: 50,
    }
  });

  const [loading, setLoading] = useState(false);
  const [dateRange, setDateRange] = useState('today');

  // ============ FETCH ANALYTICS ============
  useEffect(() => {
    const fetchAnalytics = async () => {
      setLoading(true);
      try {
        const response = await fetch(`${BACKEND_URL}/api/analytics/triage?range=${dateRange}`, {
          headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` },
        });
        if (response.ok) {
          const data = await response.json();
          setAnalytics(data);
        }
      } catch (error) {
        console.error('Error fetching analytics:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchAnalytics();
  }, [dateRange]);

  // ============ CHART COMPONENTS ============
  const SimpleBarChart = ({ data, label, maxValue }) => {
    return (
      <div style={{ display: 'grid', gap: 12 }}>
        {data.map((item, idx) => (
          <div key={idx}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
              <span style={{ fontSize: 13, fontWeight: 600, color: '#0f172a' }}>{item.severity || item.label}</span>
              <span style={{ fontSize: 12, color: '#6b7280', fontWeight: 500 }}>{item.actual}m / {item.allocated}m</span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              {/* Allocated */}
              <div style={{
                height: 24,
                backgroundColor: '#e5e7eb',
                borderRadius: 4,
                overflow: 'hidden',
                position: 'relative',
              }}>
                <div
                  style={{
                    width: `${(item.allocated / maxValue) * 100}%`,
                    height: '100%',
                    backgroundColor: '#0369a1',
                    transition: 'width 0.3s ease',
                  }}
                />
              </div>
              {/* Actual */}
              <div style={{
                height: 24,
                backgroundColor: '#e5e7eb',
                borderRadius: 4,
                overflow: 'hidden',
                position: 'relative',
              }}>
                <div
                  style={{
                    width: `${(item.actual / maxValue) * 100}%`,
                    height: '100%',
                    backgroundColor: item.actual > item.allocated ? '#ef4444' : '#10b981',
                    transition: 'width 0.3s ease',
                  }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  };

  const SimplePieChart = ({ data, title }) => {
    const total = data.reduce((sum, item) => sum + item.count, 0);
    let currentAngle = 0;
    const colors = ['#0369a1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#6b7280'];

    const slices = data.map((item, idx) => {
      const sliceAngle = (item.count / total) * 360;
      const path = describePiePath(100, 100, 80, currentAngle, currentAngle + sliceAngle);
      const startAngle = currentAngle;
      currentAngle += sliceAngle;

      const labelAngle = startAngle + sliceAngle / 2;
      const labelRadius = 60;
      const labelX = 100 + labelRadius * Math.cos((labelAngle - 90) * Math.PI / 180);
      const labelY = 100 + labelRadius * Math.sin((labelAngle - 90) * Math.PI / 180);

      return {
        path,
        color: colors[idx % colors.length],
        label: `${item.percentage}%`,
        labelX,
        labelY,
      };
    });

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, alignItems: 'center' }}>
        <svg width={220} height={220} viewBox="0 0 220 220">
          {slices.map((slice, idx) => (
            <g key={idx}>
              <path d={slice.path} fill={slice.color} stroke="white" strokeWidth={2} />
              <text
                x={slice.labelX}
                y={slice.labelY}
                textAnchor="middle"
                dominantBaseline="middle"
                style={{ fontSize: 12, fontWeight: 700, fill: '#0f172a' }}
              >
                {slice.label}
              </text>
            </g>
          ))}
        </svg>
        <div style={{ display: 'grid', gap: 8 }}>
          {data.slice(0, 5).map((item, idx) => (
            <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
              <div
                style={{
                  width: 12,
                  height: 12,
                  borderRadius: 2,
                  backgroundColor: colors[idx % colors.length],
                }}
              />
              <span style={{ flex: 1, color: '#0f172a' }}>{item.name}</span>
              <span style={{ fontWeight: 600, color: '#6b7280' }}>{item.count}</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const describePiePath = (cx, cy, radius, startAngle, endAngle) => {
    const start = polarToCartesian(cx, cy, radius, endAngle);
    const end = polarToCartesian(cx, cy, radius, startAngle);
    const largeArc = endAngle - startAngle <= 180 ? '0' : '1';
    return [
      'M',
      cx,
      cy,
      'L',
      start.x,
      start.y,
      'A',
      radius,
      radius,
      0,
      largeArc,
      0,
      end.x,
      end.y,
      'Z',
    ].join(' ');
  };

  const polarToCartesian = (centerX, centerY, radius, angleInDegrees) => {
    const angleInRadians = (angleInDegrees - 90) * Math.PI / 180.0;
    return {
      x: centerX + radius * Math.cos(angleInRadians),
      y: centerY + radius * Math.sin(angleInRadians),
    };
  };

  // ============ METRICS CARD ============
  const MetricCard = ({ icon: Icon, label, value, unit, trend, trendColor }) => (
    <div style={{
      padding: 20,
      backgroundColor: 'white',
      borderRadius: 12,
      border: '1px solid #e5e7eb',
      boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
      display: 'grid',
      gap: 12,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <p style={{ margin: 0, fontSize: 12, fontWeight: 600, color: '#6b7280', textTransform: 'uppercase', letterSpacing: 0.5 }}>{label}</p>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 4, marginTop: 8 }}>
            <span style={{ fontSize: 28, fontWeight: 700, color: '#0f172a' }}>{value}</span>
            {unit && <span style={{ fontSize: 14, color: '#6b7280', fontWeight: 500 }}>{unit}</span>}
          </div>
        </div>
        <Icon size={24} color="#0369a1" />
      </div>
      {trend && (
        <div style={{
          fontSize: 12,
          color: trendColor || '#10b981',
          fontWeight: 600,
          display: 'flex',
          alignItems: 'center',
          gap: 4,
        }}>
          <FiTrendingUp size={14} />
          {trend}
        </div>
      )}
    </div>
  );

  // ============ RENDER ============
  return (
    <SidebarLayout username={userName} handleLogout={() => {}}>
    <div style={{ maxWidth: 1400 }}>
      {/* Header */}
      <div style={{ marginBottom: 32 }}>
        <h1 style={{ margin: '0 0 8px 0', fontSize: 28, fontWeight: 700, color: '#0f172a' }}>
          <FiBarChart2 size={32} style={{ marginRight: 12, verticalAlign: 'middle', color: '#0369a1' }} />
          Triage Analytics Dashboard
        </h1>
        <p style={{ margin: 0, fontSize: 14, color: '#6b7280' }}>Clinical trial performance metrics and quality analysis</p>
      </div>

      {/* Date Range Filter */}
      <div style={{ marginBottom: 24, display: 'flex', gap: 8 }}>
        {['today', 'week', 'month'].map(range => (
          <button
            key={range}
            onClick={() => setDateRange(range)}
            style={{
              padding: '8px 16px',
              backgroundColor: dateRange === range ? '#0f172a' : '#f3f4f6',
              color: dateRange === range ? 'white' : '#374151',
              border: 'none',
              borderRadius: 6,
              cursor: 'pointer',
              fontSize: 13,
              fontWeight: 600,
              transition: 'all 0.2s',
            }}
          >
            {range.charAt(0).toUpperCase() + range.slice(1)}
          </button>
        ))}
        <button
          style={{
            marginLeft: 'auto',
            padding: '8px 16px',
            backgroundColor: '#0369a1',
            color: 'white',
            border: 'none',
            borderRadius: 6,
            cursor: 'pointer',
            fontSize: 13,
            fontWeight: 600,
            display: 'flex',
            alignItems: 'center',
            gap: 6,
          }}
        >
          <FiDownload size={14} />
          Export Report
        </button>
      </div>

      {/* Key Metrics */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
        gap: 16,
        marginBottom: 32,
      }}>
        <MetricCard
          icon={FiUsers}
          label="Total Patients"
          value={analytics.dailyMetrics.totalPatients}
          trend="↑ 8% from yesterday"
        />
        <MetricCard
          icon={FiCheckCircle}
          label="Triage Accuracy"
          value={analytics.dailyMetrics.triageAccuracy}
          unit="%"
          trend="↑ 2.3 points"
          trendColor="#10b981"
        />
        <MetricCard
          icon={FiAlertTriangle}
          label="Mistriage Rate"
          value={analytics.dailyMetrics.mistriageRate}
          unit="%"
          trend="↓ 1.1 points"
          trendColor="#10b981"
        />
        <MetricCard
          icon={FiClock}
          label="Avg Wait Time"
          value={analytics.dailyMetrics.avgWaitTime}
          unit="min"
          trend="↓ 5 min improvement"
          trendColor="#10b981"
        />
        <MetricCard
          icon={FiUsers}
          label="Staff Efficiency"
          value={analytics.dailyMetrics.staffEfficiency}
          unit="%"
          trend="Optimal"
          trendColor="#10b981"
        />
        <MetricCard
          icon={FiClock}
          label="Total Staff Hours"
          value={analytics.dailyMetrics.totalStaffHours}
          unit="hrs"
        />
      </div>

      {/* Charts Section */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))', gap: 24, marginBottom: 32 }}>
        {/* Wait Time vs Severity */}
        <div style={{
          padding: 24,
          backgroundColor: 'white',
          borderRadius: 12,
          border: '1px solid #e5e7eb',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
        }}>
          <h2 style={{ margin: '0 0 20px 0', fontSize: 18, fontWeight: 700, color: '#0f172a', display: 'flex', alignItems: 'center', gap: 8 }}>
            <FiBarChart2 size={20} color="#0369a1" />
            Wait Time vs Allocated Time by Severity
          </h2>
          <SimpleBarChart data={analytics.waitTimeVsSeverity} maxValue={130} />
          <div style={{ marginTop: 16, padding: '12px 0', borderTop: '1px solid #e5e7eb', fontSize: 12, color: '#6b7280' }}>
            <div style={{ display: 'flex', gap: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <div style={{ width: 12, height: 12, borderRadius: 2, backgroundColor: '#0369a1' }} />
                Allocated Time
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <div style={{ width: 12, height: 12, borderRadius: 2, backgroundColor: '#10b981' }} />
                Actual Time (On Target)
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <div style={{ width: 12, height: 12, borderRadius: 2, backgroundColor: '#ef4444' }} />
                Actual Time (Overdue)
              </div>
            </div>
          </div>
        </div>

        {/* Top Diagnoses */}
        <div style={{
          padding: 24,
          backgroundColor: 'white',
          borderRadius: 12,
          border: '1px solid #e5e7eb',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
        }}>
          <h2 style={{ margin: '0 0 20px 0', fontSize: 18, fontWeight: 700, color: '#0f172a', display: 'flex', alignItems: 'center', gap: 8 }}>
            <FiPieChart size={20} color="#0369a1" />
            Most Common Diagnoses
          </h2>
          <SimplePieChart data={analytics.diagnoses} />
        </div>
      </div>

      {/* Staff Hours Section */}
      <div style={{
        padding: 24,
        backgroundColor: 'white',
        borderRadius: 12,
        border: '1px solid #e5e7eb',
        boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
        marginBottom: 32,
      }}>
        <h2 style={{ margin: '0 0 20px 0', fontSize: 18, fontWeight: 700, color: '#0f172a', display: 'flex', alignItems: 'center', gap: 8 }}>
          <FiUsers size={20} color="#0369a1" />
          Staff Hours Tracking
        </h2>
        <div style={{ overflowX: 'auto' }}>
          <table style={{
            width: '100%',
            borderCollapse: 'collapse',
            fontSize: 13,
          }}>
            <thead>
              <tr style={{
                borderBottom: '2px solid #e5e7eb',
              }}>
                <th style={{ textAlign: 'left', padding: '12px 0', fontWeight: 700, color: '#0f172a' }}>Staff Member</th>
                <th style={{ textAlign: 'center', padding: '12px 0', fontWeight: 700, color: '#0f172a' }}>Shift Hours</th>
                <th style={{ textAlign: 'center', padding: '12px 0', fontWeight: 700, color: '#0f172a' }}>Break (hrs)</th>
                <th style={{ textAlign: 'center', padding: '12px 0', fontWeight: 700, color: '#0f172a' }}>Actual Worked</th>
                <th style={{ textAlign: 'center', padding: '12px 0', fontWeight: 700, color: '#0f172a' }}>Efficiency</th>
              </tr>
            </thead>
            <tbody>
              {analytics.staffHours.map((staff, idx) => {
                const efficiency = ((staff.actual / (staff.shift - staff.break)) * 100).toFixed(1);
                return (
                  <tr key={idx} style={{ borderBottom: '1px solid #e5e7eb' }}>
                    <td style={{ padding: '12px 0', color: '#0f172a', fontWeight: 500 }}>{staff.name}</td>
                    <td style={{ textAlign: 'center', padding: '12px 0', color: '#6b7280' }}>{staff.shift}h</td>
                    <td style={{ textAlign: 'center', padding: '12px 0', color: '#6b7280' }}>{staff.break}h</td>
                    <td style={{ textAlign: 'center', padding: '12px 0', color: '#0f172a', fontWeight: 600 }}>{staff.actual}h</td>
                    <td style={{ textAlign: 'center', padding: '12px 0' }}>
                      <span style={{
                        display: 'inline-block',
                        padding: '4px 8px',
                        borderRadius: 4,
                        backgroundColor: efficiency >= 95 ? '#dcfce7' : efficiency >= 85 ? '#fef3c7' : '#fee2e2',
                        color: efficiency >= 95 ? '#166534' : efficiency >= 85 ? '#92400e' : '#991b1b',
                        fontWeight: 600,
                      }}>
                        {efficiency}%
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Mistriage Analysis Section */}
      <div style={{
        padding: 24,
        backgroundColor: 'white',
        borderRadius: 12,
        border: '1px solid #e5e7eb',
        boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
      }}>
        <h2 style={{ margin: '0 0 20px 0', fontSize: 18, fontWeight: 700, color: '#0f172a', display: 'flex', alignItems: 'center', gap: 8 }}>
          <FiAlertTriangle size={20} color="#ef4444" />
          Mistriage Analysis & Report Discrepancies
        </h2>
        <div style={{ display: 'grid', gap: 16 }}>
          {analytics.mistriage.map((case_, idx) => (
            <div
              key={idx}
              style={{
                padding: 16,
                borderRadius: 8,
                border: `1px solid ${case_.status === 'mistriage' ? '#fecaca' : '#dcfce7'}`,
                backgroundColor: case_.status === 'mistriage' ? '#fef2f2' : '#f0fdf4',
              }}
            >
              <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr auto', gap: 16, alignItems: 'start' }}>
                <div>
                  <p style={{ margin: '0 0 4px 0', fontSize: 12, fontWeight: 600, color: '#6b7280', textTransform: 'uppercase' }}>Case ID</p>
                  <p style={{ margin: 0, fontSize: 14, fontWeight: 700, color: '#0f172a' }}>{case_.caseId}</p>
                </div>

                <div>
                  <p style={{ margin: '0 0 8px 0', fontSize: 12, fontWeight: 600, color: '#6b7280' }}>Report Chain</p>
                  <div style={{ display: 'grid', gap: 6 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12, fontSize: 13 }}>
                      <span style={{ fontWeight: 600, color: '#f59e0b', minWidth: 100 }}>AI Suggested:</span>
                      <span style={{ color: '#0f172a' }}>{case_.aiSuggested}</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12, fontSize: 13 }}>
                      <span style={{ fontWeight: 600, color: '#0369a1', minWidth: 100 }}>Nurse Approved:</span>
                      <span style={{ color: '#0f172a' }}>{case_.nurseApproved}</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12, fontSize: 13 }}>
                      <span style={{ fontWeight: 600, color: '#10b981', minWidth: 100 }}>Physician Attended:</span>
                      <span style={{ color: '#0f172a' }}>{case_.physicianAttended}</span>
                    </div>
                  </div>
                </div>

                <div style={{ textAlign: 'right', minWidth: 160 }}>
                  {case_.status === 'mistriage' ? (
                    <div>
                      <span style={{
                        display: 'inline-block',
                        padding: '6px 12px',
                        borderRadius: 6,
                        backgroundColor: '#fee2e2',
                        color: '#991b1b',
                        fontWeight: 700,
                        fontSize: 12,
                        marginBottom: 8,
                      }}>
                        ⚠ MISTRIAGE
                      </span>
                      <p style={{ margin: '8px 0 0 0', fontSize: 12, color: '#991b1b', fontWeight: 600 }}>{case_.discrepancy}</p>
                    </div>
                  ) : (
                    <span style={{
                      display: 'inline-block',
                      padding: '6px 12px',
                      borderRadius: 6,
                      backgroundColor: '#dcfce7',
                      color: '#166534',
                      fontWeight: 700,
                      fontSize: 12,
                    }}>
                      ✓ Correct
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Mistriage Summary */}
        <div style={{
          marginTop: 20,
          padding: 16,
          backgroundColor: '#f3f4f6',
          borderRadius: 8,
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: 12,
        }}>
          <div>
            <p style={{ margin: '0 0 4px 0', fontSize: 12, fontWeight: 600, color: '#6b7280', textTransform: 'uppercase' }}>Total Cases Reviewed</p>
            <p style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#0f172a' }}>{analytics.mistriage.length}</p>
          </div>
          <div>
            <p style={{ margin: '0 0 4px 0', fontSize: 12, fontWeight: 600, color: '#6b7280', textTransform: 'uppercase' }}>Correct Diagnoses</p>
            <p style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#10b981' }}>
              {analytics.mistriage.filter(c => c.status === 'correct').length} / {analytics.mistriage.length}
            </p>
          </div>
          <div>
            <p style={{ margin: '0 0 4px 0', fontSize: 12, fontWeight: 600, color: '#6b7280', textTransform: 'uppercase' }}>Mistriage Instances</p>
            <p style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#ef4444' }}>
              {analytics.mistriage.filter(c => c.status === 'mistriage').length} / {analytics.mistriage.length}
            </p>
          </div>
          <div>
            <p style={{ margin: '0 0 4px 0', fontSize: 12, fontWeight: 600, color: '#6b7280', textTransform: 'uppercase' }}>Accuracy Rate</p>
            <p style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#0f172a' }}>
              {((analytics.mistriage.filter(c => c.status === 'correct').length / analytics.mistriage.length) * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      </div>
    </div>
    </SidebarLayout>
  );
};

export default TriageAnalytics;
