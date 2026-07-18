const toTimestamp = (value) => {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
};

const formatDayLabel = (date) => {
  if (!date) return 'Unknown date';
  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  });
};

const formatTimeLabel = (date) => {
  if (!date) return 'Time not recorded';
  return date.toLocaleTimeString(undefined, {
    hour: 'numeric',
    minute: '2-digit'
  });
};

export const buildInpatientTimeline = (encounter = {}) => {
  const entries = [];

  const pushEntry = (kind, title, body, timestamp, meta = {}) => {
    const parsedTimestamp = toTimestamp(timestamp);
    if (!parsedTimestamp) return;

    entries.push({
      id: `${kind}-${parsedTimestamp.getTime()}-${entries.length}`,
      kind,
      title,
      body,
      timestamp: parsedTimestamp,
      meta
    });
  };

  (encounter.progression_notes || []).forEach((note, index) => {
    pushEntry(
      'PROGRESSION',
      note.note_type || 'Progress note',
      note.content || 'Progress note recorded',
      note.recorded_at || note.createdAt || encounter.createdAt,
      { note, index }
    );
  });

  (encounter.status_history || []).forEach((historyItem, index) => {
    pushEntry(
      'STATUS',
      historyItem.status || 'Status update',
      historyItem.notes || 'Status updated',
      historyItem.changed_at || historyItem.changedAt || encounter.createdAt,
      { historyItem, index }
    );
  });

  (encounter.treatments || []).forEach((treatment, index) => {
    pushEntry(
      'TREATMENT',
      treatment.name || treatment.intervention_type || 'Treatment recorded',
      treatment.notes || `${treatment.intervention_type || 'Treatment'} recorded`,
      treatment.administered_at || treatment.completed_at || treatment.prescribed_at || encounter.createdAt,
      { treatment, index }
    );
  });

  (encounter.recommendations || []).forEach((recommendation, index) => {
    pushEntry(
      'RECOMMENDATION',
      recommendation.recommendation_type || 'Recommendation',
      recommendation.description || 'Recommendation recorded',
      recommendation.completed_at || recommendation.recommended_at || encounter.createdAt,
      { recommendation, index }
    );
  });

  return entries.sort((a, b) => b.timestamp - a.timestamp);
};

export const groupTimelineByDay = (timeline = []) => {
  const grouped = new Map();

  timeline.forEach((entry) => {
    const key = entry.timestamp.toISOString().split('T')[0];
    if (!grouped.has(key)) {
      grouped.set(key, []);
    }
    grouped.get(key).push(entry);
  });

  return Array.from(grouped.entries())
    .sort(([a], [b]) => (a < b ? 1 : -1))
    .map(([key, entries]) => ({
      key,
      label: formatDayLabel(new Date(key)),
      entries: entries
        .slice()
        .sort((a, b) => b.timestamp - a.timestamp)
        .map((entry) => ({
          ...entry,
          displayTime: formatTimeLabel(entry.timestamp)
        }))
    }));
};
