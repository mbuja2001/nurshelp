import React, { useEffect, useMemo, useRef, useState } from 'react';
import { FiBarChart2, FiPieChart, FiAlertTriangle, FiCheckCircle, FiClock, FiUsers, FiDownload } from 'react-icons/fi';
import SidebarLayout from '../SidebarLayout';
import { useUserFromToken } from '../../hooks/useUserFromToken';
import './TriageAnalytics.css';
import {
  getBreachPriorityOrder,
  getMetricConfidenceState,
  getVisualizationDataState,
  summarizeStaffWorkload,
} from '../../utils/analyticsMetrics';

const inferUrgency = (value) => {
  const normalized = String(value || '').toLowerCase();
  if (['critical', 'red'].some((token) => normalized.includes(token))) return 'Critical';
  if (['high', 'orange'].some((token) => normalized.includes(token))) return 'High';
  if (['medium', 'yellow', 'amber'].some((token) => normalized.includes(token))) return 'Medium';
  if (['low', 'green'].some((token) => normalized.includes(token))) return 'Low';
  return null;
};

const formatValueWithN = (value, unit, sampleSize, minimumN) => {
  const numeric = value === null || value === undefined || value === '' ? NaN : Number(value);
  const state = Number.isFinite(numeric)
    ? getMetricConfidenceState(numeric, sampleSize, minimumN || 30)
    : 'INSUFFICIENT';

  if (state === 'INSUFFICIENT') {
    return {
      mainText: 'Insufficient data',
      subtitle: `n=${sampleSize || 0}, minimum n=${minimumN || 30}`,
      state,
    };
  }

  const displayValue = unit === '%' ? `${Number.isFinite(numeric) ? numeric.toFixed(1) : '0.0'}%` : `${Number.isFinite(numeric) ? numeric.toFixed(1) : '0.0'}`;

  return {
    mainText: displayValue,
    subtitle: `n=${sampleSize || 0}${minimumN ? `, minimum n=${minimumN}` : ''}`,
    state,
  };
};

const AnimatedNumber = ({ value, decimals = 0, suffix = '', prefix = '', duration = 800 }) => {
  const [displayValue, setDisplayValue] = useState(0);
  const animationStartRef = useRef(0);

  useEffect(() => {
    const numericValue = Number(value) || 0;
    let animationFrame;

    const animate = (timestamp) => {
      if (!animationStartRef.current) {
        animationStartRef.current = timestamp;
      }
      const elapsed = timestamp - animationStartRef.current;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplayValue(numericValue * eased);

      if (progress < 1) {
        animationFrame = requestAnimationFrame(animate);
      }
    };

    animationFrame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationFrame);
  }, [value, duration]);

  return <>{prefix}{Number(displayValue).toFixed(decimals)}{suffix}</>;
};

const TriageAnalytics = ({ embedded = false, mode = 'specialist' }) => {
  const { userName, userRole } = useUserFromToken();
  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';
  const canViewCaseDetails = ['CLINICAL_LEAD', 'QA', 'PHYSICIAN', 'CLINICAL_NURSE'].includes((userRole || '').toUpperCase());

  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState(false);
  const [retryToken, setRetryToken] = useState(0);
  const [dateRange, setDateRange] = useState('today');

  useEffect(() => {
    let isCurrentRequest = true;

    const fetchAnalytics = async () => {
      setLoading(true);
      setAnalytics(null);
      setFetchError(false);

      try {
        const response = await fetch(`${BACKEND_URL}/api/analytics/triage?range=${dateRange}`, {
          headers: { Authorization: `Bearer ${localStorage.getItem('token')}` },
        });

        if (!response.ok) {
          throw new Error(`Analytics request failed with status ${response.status}`);
        }

        const data = await response.json();
        if (
          data?.success !== true ||
          !data.dailyMetrics ||
          !Array.isArray(data.waitTimeVsSeverity) ||
          !Array.isArray(data.diagnoses) ||
          !Array.isArray(data.caseReviews) ||
          !Array.isArray(data.staffHours)
        ) {
          throw new Error('Analytics response did not contain the expected live data.');
        }

        if (isCurrentRequest) setAnalytics(data);
      } catch (error) {
        console.error('Error fetching analytics:', error);
        if (isCurrentRequest) {
          setAnalytics(null);
          setFetchError(true);
        }
      } finally {
        if (isCurrentRequest) setLoading(false);
      }
    };

    fetchAnalytics();
    return () => {
      isCurrentRequest = false;
    };
  }, [dateRange, BACKEND_URL, retryToken]);

  const normalizedCaseReviews = useMemo(() => {
    return (analytics?.caseReviews || []).map((caseReview) => {
      const aiSuggestedUrgency = inferUrgency(caseReview.aiSuggestedUrgency);
      const nurseApprovedUrgency = inferUrgency(caseReview.nurseApprovedUrgency);
      const physicianFinalUrgency = inferUrgency(caseReview.physicianFinalUrgency);

      const mistriageFlag = Boolean(
        caseReview.mistriageFlag ?? (
          aiSuggestedUrgency && nurseApprovedUrgency && aiSuggestedUrgency !== nurseApprovedUrgency
        )
      );

      const refinementOccurred = Boolean(
        caseReview.refinementOccurred ?? (
          caseReview.aiSuggestedDiagnosis &&
          caseReview.physicianFinalDiagnosis &&
          String(caseReview.aiSuggestedDiagnosis).trim().toLowerCase() !== String(caseReview.physicianFinalDiagnosis).trim().toLowerCase()
        )
      );

      const reviewOutcome = caseReview.reviewOutcome || 'PENDING_CLINICAL_REVIEW';

      return {
        ...caseReview,
        aiSuggestedUrgency,
        nurseApprovedUrgency,
        physicianFinalUrgency,
        mistriageFlag,
        refinementOccurred,
        reviewOutcome,
      };
    });
  }, [analytics?.caseReviews]);

  const waitTimeVsSeverity = useMemo(() => getBreachPriorityOrder(analytics?.waitTimeVsSeverity || []), [analytics?.waitTimeVsSeverity]);
  const workloadSummary = useMemo(() => summarizeStaffWorkload(analytics?.staffHours || []), [analytics?.staffHours]);
  const diagnosisChartState = useMemo(() => getVisualizationDataState(analytics?.diagnoses || [], 30), [analytics?.diagnoses]);
  const waitTimeChartState = useMemo(
    () => getVisualizationDataState((analytics?.waitTimeVsSeverity || []).map((item) => ({ count: Number(item.count || 0) })), 12),
    [analytics?.waitTimeVsSeverity]
  );

  const totalCaseReviews = normalizedCaseReviews.length;
  const reviewedCaseReviews = normalizedCaseReviews.filter((item) => item.reviewOutcome !== 'PENDING_CLINICAL_REVIEW');
  const mistriageCount = reviewedCaseReviews.filter((item) => item.mistriageFlag).length;
  const correctCount = reviewedCaseReviews.filter((item) => item.reviewOutcome === 'CORRECT').length;
  const diagnosticRefinementCount = reviewedCaseReviews.filter((item) => item.refinementOccurred && !item.mistriageFlag).length;
  const pendingClinicalReviewCount = normalizedCaseReviews.filter((item) => item.reviewOutcome === 'PENDING_CLINICAL_REVIEW').length;
  const mistriageRate = analytics?.dailyMetrics?.mistriageRate;

  const renderPage = (content) => embedded ? content : (
    <SidebarLayout username={userName} handleLogout={() => {}}>
      {content}
    </SidebarLayout>
  );

  if (loading) {
    return renderPage(
      <div role="status" style={{ maxWidth: 1400, padding: 24, border: '1px solid #e2e8f0', borderRadius: 12, color: '#475569' }}>
        Loading live hospital analytics…
      </div>
    );
  }

  if (fetchError || !analytics) {
    return renderPage(
      <div role="alert" style={{ maxWidth: 1400, padding: 24, border: '1px solid #fecaca', borderRadius: 12, background: '#fff7f7', color: '#991b1b' }}>
        <p style={{ margin: '0 0 12px', fontWeight: 700 }}>Live analytics are unavailable.</p>
        <p style={{ margin: '0 0 16px' }}>The hospital analytics service could not provide current data. No sample values are being shown.</p>
        <button
          type="button"
          onClick={() => {
            setAnalytics(null);
            setFetchError(false);
            setLoading(true);
            setRetryToken((current) => current + 1);
          }}
          style={{ padding: '8px 14px', border: 0, borderRadius: 6, background: '#991b1b', color: 'white', cursor: 'pointer', fontWeight: 700 }}
        >
          Retry
        </button>
      </div>
    );
  }

  const caseOutcomeLabel = (caseReview) => {
    if (caseReview.reviewOutcome === 'PENDING_CLINICAL_REVIEW') {
      return '⚠ Pending clinical review';
    }

    if (caseReview.mistriageFlag) {
      return '⚠ Mistriage';
    }

    if (caseReview.refinementOccurred) {
      return '✓ Correct triage, diagnosis refined';
    }

    return '✓ Correct triage';
  };

  const caseOutcomeColor = (caseReview) => {
    if (caseReview.reviewOutcome === 'PENDING_CLINICAL_REVIEW') {
      return { backgroundColor: '#fef3c7', color: '#92400e', border: '1px solid #fcd34d' };
    }

    if (caseReview.mistriageFlag) {
      return { backgroundColor: '#fee2e2', color: '#991b1b', border: '1px solid #fca5a5' };
    }

    if (caseReview.refinementOccurred) {
      return { backgroundColor: '#dbeafe', color: '#1d4ed8', border: '1px solid #93c5fd' };
    }

    return { backgroundColor: '#dcfce7', color: '#166534', border: '1px solid #86efac' };
  };

  const SimpleBarChart = ({ data }) => {
    if (!Array.isArray(data) || data.length === 0) {
      return (
        <div style={{ padding: 16, borderRadius: 10, border: '1px dashed #cbd5e1', color: '#475569', fontSize: 13 }}>
          No recorded wait-time data is available for this period.
        </div>
      );
    }

    const maxValue = Math.max(...data.map((item) => Math.max(item.allocated, item.actual, 10)), 10);

    return (
      <div style={{ display: 'grid', gap: 12 }}>
        {data.map((item, idx) => {
          const severityColor = {
            Low: '#60a5fa',
            Medium: '#f59e0b',
            High: '#f97316',
            Critical: '#dc2626',
          }[item.severity] || '#0369a1';

          const breachPercent = item.actual > item.allocated ? ((item.actual - item.allocated) / item.allocated) * 100 : 0;
          const shouldEmphasize = item.severity === 'Critical' || breachPercent > 25;

          return (
            <div key={idx} style={{ padding: 8, borderRadius: 8, border: shouldEmphasize ? '1px solid #fecaca' : '1px solid #f3f4f6', backgroundColor: shouldEmphasize ? '#fff7ed' : '#ffffff' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                <span style={{ fontSize: 13, fontWeight: 700, color: '#0f172a' }}>{item.severity}</span>
                <span style={{ fontSize: 12, color: '#475569', fontWeight: 600 }}>
                  {item.actual}m / {item.allocated}m ({breachPercent > 0 ? `${breachPercent.toFixed(0)}% overshoot` : 'on target'})
                </span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                <div style={{ height: shouldEmphasize ? 26 : 20, backgroundColor: '#e5e7eb', borderRadius: 6, overflow: 'hidden' }}>
                  <div style={{ width: `${(item.allocated / maxValue) * 100}%`, height: '100%', backgroundColor: '#0369a1', transition: 'width 0.9s cubic-bezier(.2,.9,.2,1)', animation: 'pulseBar 2s ease-in-out infinite alternate' }} />
                </div>
                <div style={{ height: shouldEmphasize ? 26 : 20, backgroundColor: '#e5e7eb', borderRadius: 6, overflow: 'hidden' }}>
                  <div
                    style={{
                      width: `${(item.actual / maxValue) * 100}%`,
                      height: '100%',
                      backgroundColor: item.actual > item.allocated ? severityColor : '#10b981',
                      transition: 'width 0.9s cubic-bezier(.2,.9,.2,1)',
                      animation: 'pulseBar 2s ease-in-out infinite alternate',
                      boxShadow: item.severity === 'Critical' ? 'inset 0 0 0 2px rgba(255,255,255,0.3)' : 'none',
                    }}
                  />
                </div>
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  const SimplePieChart = ({ data }) => {
    const emptyState = !Array.isArray(data) || data.length === 0 || data.every((item) => Number(item.count || 0) <= 0);

    if (emptyState) {
      return (
        <div style={{ padding: 16, borderRadius: 10, border: '1px dashed #cbd5e1', color: '#475569', fontSize: 13 }}>
          No recorded diagnosis data is available for this period.
        </div>
      );
    }

    const total = data.reduce((sum, item) => sum + item.count, 0) || 1;
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
        color: colors[idx % colors.length],
        label: `${item.percentage}%`,
        labelX,
        labelY,
        path,
      };
    });

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 220px), 1fr))', gap: 24, alignItems: 'center', minWidth: 0 }}>
        <svg viewBox="0 0 220 220" style={{ width: '100%', maxWidth: 220, height: 'auto' }}>
          {slices.map((slice, idx) => (
            <g key={idx}>
              <path d={slice.path} fill={slice.color} stroke="white" strokeWidth={2} style={{ transformOrigin: '100px 100px', animation: 'sliceReveal 0.9s ease-out both', animationDelay: `${idx * 80}ms` }} />
              <text x={slice.labelX} y={slice.labelY} textAnchor="middle" dominantBaseline="middle" style={{ fontSize: 12, fontWeight: 700, fill: '#0f172a' }}>
                {slice.label}
              </text>
            </g>
          ))}
        </svg>
        <div style={{ display: 'grid', gap: 8 }}>
          {data.slice(0, 5).map((item, idx) => (
            <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
              <div style={{ width: 12, height: 12, borderRadius: 2, backgroundColor: colors[idx % colors.length] }} />
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
    return ['M', cx, cy, 'L', start.x, start.y, 'A', radius, radius, 0, largeArc, 0, end.x, end.y, 'Z'].join(' ');
  };

  const polarToCartesian = (centerX, centerY, radius, angleInDegrees) => {
    const angleInRadians = (angleInDegrees - 90) * Math.PI / 180.0;
    return {
      x: centerX + radius * Math.cos(angleInRadians),
      y: centerY + radius * Math.sin(angleInRadians),
    };
  };

  const MetricCard = ({ icon: Icon, label, value, unit, sampleSize, minimumN, dataQualityNote }) => {
    const metric = formatValueWithN(value, unit, sampleSize, minimumN || 30);
    const state = metric.state;
    const cardStyles = {
      padding: 20,
      backgroundColor: state === 'INSUFFICIENT' ? '#f8fafc' : 'white',
      borderRadius: 12,
      border: state === 'INSUFFICIENT' ? '1px solid #e2e8f0' : state === 'LOW_CONFIDENCE' ? '1px solid #fcd34d' : '1px solid #bfdbfe',
      boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
      display: 'grid',
      gap: 12,
    };

    const valueTextColor = state === 'INSUFFICIENT' ? '#64748b' : state === 'LOW_CONFIDENCE' ? '#b45309' : '#0f172a';

    return (
      <div style={cardStyles}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <p style={{ margin: 0, fontSize: 12, fontWeight: 700, color: '#6b7280', textTransform: 'uppercase', letterSpacing: 0.5 }}>{label}</p>
            <div style={{ marginTop: 10 }}>
              <span style={{ display: 'block', fontSize: 27, fontWeight: 700, color: valueTextColor }}>
                {state === 'INSUFFICIENT' ? metric.mainText : <AnimatedNumber value={Number(value)} decimals={1} suffix={unit === '%' ? '%' : unit === 'min' ? ' min' : unit === 'hrs' ? ' hrs' : unit === 'cases/hr' ? ' cases/hr' : ''} />}
              </span>
              <span style={{ display: 'block', marginTop: 6, fontSize: 12, color: '#6b7280', fontWeight: 600 }}>{metric.subtitle}</span>
            </div>
          </div>
          <Icon size={24} color={state === 'INSUFFICIENT' ? '#64748b' : '#0369a1'} />
        </div>

        {dataQualityNote && (
          <div style={{ fontSize: 11, color: '#b45309', fontWeight: 700 }}>{dataQualityNote}</div>
        )}

      </div>
    );
  };

  const analyticsContent = (
    <div style={{ maxWidth: 1400, width: '100%', minWidth: 0, boxSizing: 'border-box' }}>
      <div style={{ marginBottom: 24, display: 'grid', gap: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12, minWidth: 0 }}>
          <div style={{ minWidth: 0, maxWidth: '100%' }}>
            <div style={{ fontSize: 11, letterSpacing: '0.12em', textTransform: 'uppercase', fontWeight: 800, color: '#0f766e', marginBottom: 6 }}>
              {mode === 'admin' ? 'Analytics reporting' : 'Clinical analytics'}
            </div>
            <h1 style={{ margin: '0 0 8px 0', fontSize: 28, fontWeight: 700, color: '#0f172a', maxWidth: '100%', overflowWrap: 'anywhere' }}>
              <FiBarChart2 size={32} style={{ marginRight: 12, verticalAlign: 'middle', color: '#0369a1' }} />
              {mode === 'admin' ? 'Admin Analytics Dashboard' : 'Triage Analytics Dashboard'}
            </h1>
          </div>
          {mode === 'admin' && (
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', padding: '8px 12px', borderRadius: 999, background: '#ecfeff', border: '1px solid #a5f3fc', color: '#0f172a', fontSize: 12, fontWeight: 700 }}>
              Live reporting scope • hospital wide
            </div>
          )}
        </div>
        <p style={{ margin: 0, fontSize: 14, color: '#6b7280' }}>
          {mode === 'admin'
          ? 'Live triage, patient flow, and staff workload metrics for this hospital.'
            : 'Clinical review quality metrics with explicit confidence thresholds and sample sizes'}
        </p>
      </div>

      <div style={{ marginBottom: 24, display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 8, width: '100%', minWidth: 0 }}>
        {['today', 'week', 'month'].map((range) => (
          <button
            key={range}
            onClick={() => {
              if (range === dateRange) return;
              setAnalytics(null);
              setFetchError(false);
              setLoading(true);
              setDateRange(range);
            }}
            style={{
              padding: '8px 16px',
              backgroundColor: dateRange === range ? '#0f172a' : '#f3f4f6',
              color: dateRange === range ? 'white' : '#374151',
              border: 'none',
              borderRadius: 6,
              cursor: 'pointer',
              fontSize: 13,
              fontWeight: 600,
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

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 240px), 1fr))', gap: 16, marginBottom: 32, minWidth: 0 }}>
        <MetricCard
          icon={FiUsers}
          label="Total Patients"
          value={analytics.dailyMetrics.totalPatients}
          sampleSize={analytics.dailyMetrics.totalPatients}
          minimumN={1}
        />
        <MetricCard
          icon={FiCheckCircle}
          label="Triage Accuracy"
          value={analytics.dailyMetrics.triageAccuracy}
          unit="%"
          sampleSize={analytics.dailyMetrics.triageAccuracySampleSize}
          minimumN={30}
          dataQualityNote={getMetricConfidenceState(analytics.dailyMetrics.triageAccuracy, analytics.dailyMetrics.triageAccuracySampleSize, 30) === 'LOW_CONFIDENCE' ? 'Low-confidence metric: sample below publication threshold.' : null}
        />
        <MetricCard
          icon={FiAlertTriangle}
          label="Mistriage Rate"
          value={mistriageRate}
          unit="%"
          sampleSize={analytics.dailyMetrics.mistriageSampleSize}
          minimumN={30}
          dataQualityNote={getMetricConfidenceState(mistriageRate, analytics.dailyMetrics.mistriageSampleSize, 30) === 'LOW_CONFIDENCE' ? 'Mistriage rate is based on reviewed urgency comparisons; sample is below the publication threshold.' : null}
        />
        <MetricCard
          icon={FiClock}
          label="Avg Wait Time"
          value={analytics.dailyMetrics.avgWaitTime}
          unit="min"
          sampleSize={analytics.dailyMetrics.avgWaitTimeSampleSize}
          minimumN={1}
        />
        <MetricCard
          icon={FiUsers}
          label="Assigned Cases / Hour"
          value={analytics.dailyMetrics.assignedCasesPerHour}
          unit="cases/hr"
          sampleSize={analytics.dailyMetrics.assignedCasesPerHourSampleSize}
          minimumN={5}
          dataQualityNote={workloadSummary.nearZeroVariance ? 'Variance check triggered: metric is not discriminating between staff.' : null}
        />
        <MetricCard
          icon={FiClock}
          label="Total Staff Hours"
          value={analytics.dailyMetrics.totalStaffHours}
          unit="hrs"
          sampleSize={analytics.dailyMetrics.totalStaffHoursSampleSize}
          minimumN={1}
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 500px), 1fr))', gap: 24, marginBottom: 32, minWidth: 0 }}>
        <div style={{ padding: 24, minWidth: 0, boxSizing: 'border-box', backgroundColor: 'white', borderRadius: 12, border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
          <h2 style={{ margin: '0 0 20px 0', fontSize: 18, fontWeight: 700, color: '#0f172a', display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 8, minWidth: 0 }}>
            <FiBarChart2 size={20} color="#0369a1" />
            Wait Time vs Allocated Time by Severity
          </h2>
          {waitTimeChartState.showPlaceholder
            ? <div style={{ marginBottom: 12, fontSize: 12, color: '#475569', fontWeight: 700 }}>{waitTimeChartState.message}</div>
            : <SimpleBarChart data={waitTimeVsSeverity} />}
          <div style={{ marginTop: 16, padding: '12px 0', borderTop: '1px solid #e5e7eb', fontSize: 12, color: '#6b7280' }}>
            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}><div style={{ width: 12, height: 12, borderRadius: 2, backgroundColor: '#0369a1' }} />Allocations</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}><div style={{ width: 12, height: 12, borderRadius: 2, backgroundColor: '#10b981' }} />On target</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}><div style={{ width: 12, height: 12, borderRadius: 2, backgroundColor: '#dc2626' }} />Breach</div>
            </div>
          </div>
        </div>

        <div style={{ padding: 24, minWidth: 0, boxSizing: 'border-box', backgroundColor: 'white', borderRadius: 12, border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
          <h2 style={{ margin: '0 0 20px 0', fontSize: 18, fontWeight: 700, color: '#0f172a', display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 8, minWidth: 0 }}>
            <FiPieChart size={20} color="#0369a1" />
            Most Common Diagnoses
          </h2>
          {diagnosisChartState.showPlaceholder
            ? <div style={{ marginBottom: 12, fontSize: 12, color: '#475569', fontWeight: 700 }}>{diagnosisChartState.message}</div>
            : <SimplePieChart data={analytics.diagnoses} />}
        </div>
      </div>

      <div style={{ padding: 24, minWidth: 0, boxSizing: 'border-box', backgroundColor: 'white', borderRadius: 12, border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', marginBottom: 32 }}>
        <h2 style={{ margin: '0 0 20px 0', fontSize: 18, fontWeight: 700, color: '#0f172a', display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 8, minWidth: 0 }}>
          <FiUsers size={20} color="#0369a1" />
          Workforce Carrying Load
        </h2>
        <div className="analytics-workforce-table" style={{ overflowX: 'auto', maxWidth: '100%' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                <th style={{ textAlign: 'left', padding: '12px 0', fontWeight: 700, color: '#0f172a' }}>Staff Member</th>
                <th style={{ textAlign: 'center', padding: '12px 0', fontWeight: 700, color: '#0f172a' }}>Shift Hours</th>
                <th style={{ textAlign: 'center', padding: '12px 0', fontWeight: 700, color: '#0f172a' }}>Break</th>
                <th style={{ textAlign: 'center', padding: '12px 0', fontWeight: 700, color: '#0f172a' }}>Assigned Cases</th>
                <th style={{ textAlign: 'center', padding: '12px 0', fontWeight: 700, color: '#0f172a' }}>Cases / Hr</th>
              </tr>
            </thead>
            <tbody>
              {workloadSummary.rows.length ? workloadSummary.rows.map((staff, idx) => (
                <tr key={idx} style={{ borderBottom: '1px solid #e5e7eb' }}>
                  <td style={{ padding: '12px 0', color: '#0f172a', fontWeight: 500 }}>{staff.name}</td>
                  <td style={{ textAlign: 'center', padding: '12px 0', color: '#6b7280' }}>{Number.isFinite(staff.shift) ? `${staff.shift.toFixed(1)}h` : '—'}</td>
                  <td style={{ textAlign: 'center', padding: '12px 0', color: '#6b7280' }}>{Number.isFinite(staff.break) ? `${staff.break.toFixed(1)}h` : '—'}</td>
                  <td style={{ textAlign: 'center', padding: '12px 0', color: '#0f172a', fontWeight: 600 }}>{Number.isFinite(staff.assignedCases) ? staff.assignedCases : '—'}</td>
                  <td style={{ textAlign: 'center', padding: '12px 0' }}>
                    <span style={{
                      display: 'inline-block',
                      padding: '4px 8px',
                      borderRadius: 4,
                      backgroundColor: workloadSummary.nearZeroVariance ? '#fef3c7' : '#dcfce7',
                      color: workloadSummary.nearZeroVariance ? '#92400e' : '#166534',
                      fontWeight: 700,
                    }}>
                      {Number.isFinite(staff.casesPerHour) ? staff.casesPerHour.toFixed(1) : 'Insufficient data'}
                    </span>
                  </td>
                </tr>
              )) : (
                <tr>
                  <td colSpan={5} style={{ padding: 16, textAlign: 'center', color: '#64748b' }}>No staff clock records are available for this period.</td>
                </tr>
              )}
            </tbody>
          </table>
          {workloadSummary.nearZeroVariance && (
            <div style={{ marginTop: 12, padding: '10px 12px', borderRadius: 6, backgroundColor: '#fff7ed', color: '#b45309', fontWeight: 700, fontSize: 12 }}>
              Data quality check: variance across staff is near zero; review formula before promoting this metric.
            </div>
          )}
        </div>
        <div className="analytics-workforce-cards">
          {workloadSummary.rows.length ? workloadSummary.rows.map((staff, idx) => (
            <div key={idx} style={{ padding: 14, border: '1px solid #e5e7eb', borderRadius: 8, background: '#fff' }}>
              <p style={{ margin: '0 0 12px', fontWeight: 700, color: '#0f172a' }}>{staff.name}</p>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 12 }}>
                <div>
                  <span style={{ display: 'block', fontSize: 11, color: '#64748b' }}>Shift hours</span>
                  <strong>{Number.isFinite(staff.shift) ? `${staff.shift.toFixed(1)}h` : '—'}</strong>
                </div>
                <div>
                  <span style={{ display: 'block', fontSize: 11, color: '#64748b' }}>Break</span>
                  <strong>{Number.isFinite(staff.break) ? `${staff.break.toFixed(1)}h` : '—'}</strong>
                </div>
                <div>
                  <span style={{ display: 'block', fontSize: 11, color: '#64748b' }}>Assigned cases</span>
                  <strong>{Number.isFinite(staff.assignedCases) ? staff.assignedCases : '—'}</strong>
                </div>
                <div>
                  <span style={{ display: 'block', fontSize: 11, color: '#64748b' }}>Cases / hr</span>
                  <strong>{Number.isFinite(staff.casesPerHour) ? staff.casesPerHour.toFixed(1) : 'Insufficient data'}</strong>
                </div>
              </div>
            </div>
          )) : (
            <div style={{ padding: 14, color: '#64748b' }}>No staff clock records are available for this period.</div>
          )}
          {workloadSummary.nearZeroVariance && (
            <div style={{ padding: '10px 12px', borderRadius: 6, backgroundColor: '#fff7ed', color: '#b45309', fontWeight: 700, fontSize: 12 }}>
              Data quality check: variance across staff is near zero; review formula before promoting this metric.
            </div>
          )}
        </div>
      </div>

      <div style={{ padding: 24, minWidth: 0, boxSizing: 'border-box', backgroundColor: 'white', borderRadius: 12, border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
        <h2 style={{ margin: '0 0 20px 0', fontSize: 18, fontWeight: 700, color: '#0f172a', display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 8, minWidth: 0 }}>
          <FiAlertTriangle size={20} color="#ef4444" />
          Review Quality Log
        </h2>

        {!canViewCaseDetails && (
          <div style={{ padding: '14px 16px', borderRadius: 8, border: '1px solid #e2e8f0', backgroundColor: '#f8fafc', color: '#475569', fontSize: 14, fontWeight: 600 }}>
            Case-level clinical review detail is restricted to clinical leads and QA users.
          </div>
        )}

        {canViewCaseDetails && (
          <>
            {normalizedCaseReviews.length === 0 ? (
              <div style={{ padding: 14, borderRadius: 8, background: '#f8fafc', color: '#475569', fontSize: 13 }}>
                No recorded triage-review data is available for this period.
              </div>
            ) : (
              <div style={{ display: 'grid', gap: 16 }}>
                {normalizedCaseReviews.map((caseReview, idx) => (
                <div key={idx} style={{ padding: 16, borderRadius: 8, border: '1px solid #e5e7eb', backgroundColor: caseReview.mistriageFlag ? '#fef2f2' : caseReview.refinementOccurred ? '#f8fafc' : '#f0fdf4' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 220px), 1fr))', gap: 16, alignItems: 'start', minWidth: 0 }}>
                    <div>
                      <p style={{ margin: '0 0 4px 0', fontSize: 12, fontWeight: 700, color: '#6b7280', textTransform: 'uppercase' }}>Case</p>
                      <p style={{ margin: 0, fontSize: 15, fontWeight: 700, color: '#0f172a' }}>{caseReview.encounterId}</p>
                    </div>

                    <div>
                      <p style={{ margin: '0 0 8px 0', fontSize: 12, fontWeight: 700, color: '#6b7280' }}>Urgency & diagnosis chain</p>
                      <div style={{ display: 'grid', gap: 6, fontSize: 13 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                          <span style={{ fontWeight: 700, minWidth: 104, color: '#f59e0b' }}>AI: </span>
                          <span>{caseReview.aiSuggestedUrgency || 'Not recorded'} / {caseReview.aiSuggestedDiagnosis || 'Not recorded'}</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                          <span style={{ fontWeight: 700, minWidth: 104, color: '#0369a1' }}>Nurse: </span>
                          <span>{caseReview.nurseApprovedUrgency || 'Not recorded'} / {caseReview.nurseApprovedDiagnosis || 'Not recorded'}</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                          <span style={{ fontWeight: 700, minWidth: 104, color: '#10b981' }}>Physician: </span>
                          <span>{caseReview.physicianFinalUrgency || 'Not recorded'} / {caseReview.physicianFinalDiagnosis || 'Not recorded'}</span>
                        </div>
                      </div>
                    </div>

                    <div style={{ textAlign: 'left', minWidth: 0 }}>
                      <span style={{ display: 'inline-block', padding: '6px 10px', borderRadius: 6, fontSize: 12, fontWeight: 700, ...caseOutcomeColor(caseReview) }}>
                        {caseOutcomeLabel(caseReview)}
                      </span>
                      {caseReview.reviewerNotes && (
                        <p style={{ margin: '8px 0 0 0', fontSize: 12, color: '#475569', textAlign: 'left' }}>{caseReview.reviewerNotes}</p>
                      )}
                    </div>
                  </div>
                </div>
                ))}
              </div>
            )}

            {totalCaseReviews > 0 && (
              <div style={{ marginTop: 20, padding: 16, backgroundColor: '#f3f4f6', borderRadius: 8, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 200px), 1fr))', gap: 12, minWidth: 0 }}>
                <div>
                  <p style={{ margin: '0 0 4px 0', fontSize: 12, fontWeight: 700, color: '#6b7280', textTransform: 'uppercase' }}>Cases with triage data</p>
                  <p style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#0f172a' }}>{totalCaseReviews}</p>
                </div>
                <div>
                  <p style={{ margin: '0 0 4px 0', fontSize: 12, fontWeight: 700, color: '#6b7280', textTransform: 'uppercase' }}>Correct Triage</p>
                  <p style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#10b981' }}>{correctCount} / {reviewedCaseReviews.length}</p>
                </div>
                <div>
                  <p style={{ margin: '0 0 4px 0', fontSize: 12, fontWeight: 700, color: '#6b7280', textTransform: 'uppercase' }}>Mistriage Cases</p>
                  <p style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#ef4444' }}>{mistriageCount} / {reviewedCaseReviews.length}</p>
                </div>
                <div>
                  <p style={{ margin: '0 0 4px 0', fontSize: 12, fontWeight: 700, color: '#6b7280', textTransform: 'uppercase' }}>Diagnosis Refinement</p>
                  <p style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#2563eb' }}>{diagnosticRefinementCount} / {reviewedCaseReviews.length}</p>
                </div>
                <div>
                  <p style={{ margin: '0 0 4px 0', fontSize: 12, fontWeight: 700, color: '#6b7280', textTransform: 'uppercase' }}>Pending Review</p>
                  <p style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#b45309' }}>{pendingClinicalReviewCount} / {totalCaseReviews}</p>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );

  if (mode === 'admin') {
    return analyticsContent;
  }

  return renderPage(analyticsContent);
};

export default TriageAnalytics;
