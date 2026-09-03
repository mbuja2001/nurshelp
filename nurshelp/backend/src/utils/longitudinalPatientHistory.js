const DEFAULT_RECENT_HOURS = 72;
const DEFAULT_TREND_DAYS = 90;
const DEFAULT_SUMMARY_ITEMS = 25;

const getBudgetProfile = (maxCharacters) => {
  if (maxCharacters >= 12000) return { recentHours: 72, maxRelevant: 5, maxSummaryItems: 20 };
  if (maxCharacters >= 6000) return { recentHours: 48, maxRelevant: 3, maxSummaryItems: 15 };
  if (maxCharacters >= 3000) return { recentHours: 24, maxRelevant: 2, maxSummaryItems: 10 };
  return { recentHours: 12, maxRelevant: 1, maxSummaryItems: 5 };
};

const toDate = (value) => {
  const date = value instanceof Date ? value : new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
};

const numeric = (value) => {
  const result = Number(value);
  return Number.isFinite(result) ? result : null;
};

const normalizeMobility = (value) => {
  const key = String(value ?? '').trim().toLowerCase();
  if (['with help', 'help', 'assisted'].includes(key)) return 'With help';
  if (['immobile', 'not walking', 'stretcher', 'stretcher / immobile'].includes(key)) return 'Immobile';
  if (['walks', 'walking'].includes(key)) return 'Walks';
  return 'Unknown';
};

const getTews = (encounter) => numeric(
  encounter.tews?.total ?? encounter.SATS_reasoning?.total ?? encounter.SATS_reasoning?.tews_total ?? encounter.tews_total ?? encounter.tewsTotal
);

const getPriority = (encounter, tewsTotal) => {
  const priority = encounter.tews?.priority ?? encounter.SATS_reasoning?.priority ?? encounter.satsLevel ?? encounter.priority;
  if (priority) return String(priority).toLowerCase();
  if (tewsTotal === null) return 'unknown';
  if (tewsTotal >= 7) return 'red';
  if (tewsTotal >= 5) return 'orange';
  if (tewsTotal >= 3) return 'yellow';
  return 'green';
};

const getVitals = (encounter) => encounter.vitals || {};
const getId = (encounter) => String(encounter?.id ?? encounter?._id ?? encounter?.encounterId ?? '');

const getProblemTags = (encounter) => [...new Set([
  ...(Array.isArray(encounter.problem_tags) ? encounter.problem_tags : []),
  ...(Array.isArray(encounter.problemTags) ? encounter.problemTags : []),
  ...(Array.isArray(encounter.diagnoses) ? encounter.diagnoses : []),
  ...(Array.isArray(encounter.symptoms) ? encounter.symptoms : []),
  ...(encounter.symptoms && !Array.isArray(encounter.symptoms) ? [encounter.symptoms] : [])
].map((value) => String(value || '').trim().toLowerCase()).filter(Boolean))];

const getCriticalFacts = (encounter) => {
  const explicit = Array.isArray(encounter.criticalFacts)
    ? encounter.criticalFacts
    : Array.isArray(encounter.critical_facts) ? encounter.critical_facts : [];
  const candidates = [...explicit];
  const criticalHistory = encounter.criticalHistory || encounter.critical_history || encounter.patient?.criticalHistory || encounter.patient?.critical_history || {};
  const criticalHistoryLabels = {
    allergies: 'allergy',
    adverseReactions: 'adverse_reaction',
    medications: 'high_risk_medication',
    conditions: 'major_condition',
    implantsOrProcedures: 'critical_implant_or_procedure'
  };
  Object.entries(criticalHistoryLabels).forEach(([field, type]) => {
    const values = Array.isArray(criticalHistory[field]) ? criticalHistory[field] : [];
    values.forEach((description) => candidates.push({ type, description, severity: 'high-risk' }));
  });
  if (encounter.allergy || encounter.allergies || encounter.allergyEvent) {
    candidates.push({ type: 'allergy', description: encounter.allergy || encounter.allergies || 'Documented allergy', severity: 'high-risk' });
  }
  if (encounter.adverseReaction || encounter.adverse_reaction) {
    candidates.push({ type: 'adverse_reaction', description: encounter.adverseReaction || encounter.adverse_reaction, severity: 'life-threatening' });
  }
  return candidates.map((fact) => ({
    ...fact,
    source_encounter_id: fact.source_encounter_id || fact.sourceEncounterId || getId(encounter),
    source_date: fact.source_date || fact.sourceDate || encounter.date,
    never_expire: fact.never_expire ?? fact.neverExpire ?? true
  }));
};

const isCritical = (encounter) => Boolean(
  encounter.criticalFlag ?? encounter.critical ?? encounter.allergyEvent ?? encounter.adverseReaction ?? getCriticalFacts(encounter).length
);

const flattenTerms = (encounters, field, maxItems = DEFAULT_SUMMARY_ITEMS) => [...new Set(
  encounters.flatMap((encounter) => {
    const value = encounter[field];
    return Array.isArray(value) ? value : value ? [value] : [];
  }).map((value) => String(value).trim()).filter(Boolean)
)].slice(0, maxItems);

const average = (values) => values.length
  ? Number((values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(2))
  : null;

const buildVitalTrend = (encounters) => {
  const chronological = [...encounters].sort((left, right) => new Date(left.date) - new Date(right.date));
  const fields = [
    ['systolic_bp', (encounter) => numeric(getVitals(encounter).systolic_bp ?? getVitals(encounter).bp_systolic)],
    ['heart_rate', (encounter) => numeric(getVitals(encounter).heart_rate ?? getVitals(encounter).hr)],
    ['respiratory_rate', (encounter) => numeric(getVitals(encounter).respiratory_rate ?? getVitals(encounter).resp)]
  ];

  return Object.fromEntries(fields.map(([name, getter]) => {
    const values = chronological.map(getter).filter((value) => value !== null);
    return [name, { first: values[0] ?? null, latest: values.at(-1) ?? null, change: values.length >= 2 ? values.at(-1) - values[0] : null }];
  }));
};

const buildTrendSummary = (encounters, maxSummaryItems = DEFAULT_SUMMARY_ITEMS) => {
  const scored = encounters.map((encounter) => ({ encounter, total: getTews(encounter) })).filter(({ total }) => total !== null);
  const chronological = [...encounters].sort((left, right) => new Date(left.date) - new Date(right.date));
  const latestScored = [...scored].sort((left, right) => new Date(right.encounter.date) - new Date(left.encounter.date))[0];
  const mobilitySequence = chronological.map((encounter) => normalizeMobility(encounter.mobility ?? encounter.vitals?.mobility ?? encounter.SATS_reasoning?.mobility));

  return {
    encounter_count: encounters.length,
    latest_tews: latestScored?.total ?? null,
    highest_tews: scored.length ? Math.max(...scored.map(({ total }) => total)) : null,
    average_tews: average(scored.map(({ total }) => total)),
    red_cases: scored.filter(({ encounter, total }) => getPriority(encounter, total) === 'red').length,
    orange_cases: scored.filter(({ encounter, total }) => getPriority(encounter, total) === 'orange').length,
    vital_trends: buildVitalTrend(encounters),
    mobility_sequence: mobilitySequence.slice(-maxSummaryItems),
    mobility_changes: mobilitySequence.filter((value, index) => index === 0 || value !== mobilitySequence[index - 1]).slice(-maxSummaryItems),
    trauma_present_count: encounters.filter((encounter) => Boolean(encounter.traumaPresent ?? encounter.vitals?.traumaPresent ?? encounter.SATS_reasoning?.traumaPresent)).length,
    recurrent_diagnoses: flattenTerms(encounters, 'diagnoses', maxSummaryItems),
    repeated_interventions: flattenTerms(encounters, 'interventions', maxSummaryItems),
    dispositions: flattenTerms(encounters, 'disposition', maxSummaryItems)
  };
};

const rankRelevantHistory = (encounters, currentEncounter) => {
  const currentTerms = getProblemTags(currentEncounter || {});

  return encounters.map((encounter) => {
    const terms = [...getProblemTags(encounter), ...(encounter.interventions || [])].map((term) => String(term).toLowerCase());
    const matches = terms.filter((term) => currentTerms.some((current) => term.includes(current) || current.includes(term))).length;
    const tewsTotal = getTews(encounter) ?? 0;
    const trauma = Boolean(encounter.traumaPresent ?? encounter.vitals?.traumaPresent ?? encounter.SATS_reasoning?.traumaPresent);
    return { encounter, relevance: matches * 10 + tewsTotal + (trauma ? 5 : 0) };
  }).sort((left, right) => right.relevance - left.relevance || new Date(right.encounter.date) - new Date(left.encounter.date));
};

export const buildHistoryWindows = ({ encounters = [], currentEncounter = null, asOf = new Date(), recentHours = DEFAULT_RECENT_HOURS, trendDays = DEFAULT_TREND_DAYS, maxRelevant = 5, maxSummaryItems = DEFAULT_SUMMARY_ITEMS } = {}) => {
  const referenceDate = toDate(asOf);
  if (!referenceDate) throw new Error('asOf must be a valid date');

  const recentCutoff = referenceDate.getTime() - recentHours * 60 * 60 * 1000;
  const trendCutoff = referenceDate.getTime() - trendDays * 24 * 60 * 60 * 1000;
  const invalidDateCount = encounters.filter((encounter) => !toDate(encounter.date)).length;
  const ordered = encounters.map((encounter) => ({ ...encounter, date: toDate(encounter.date)?.toISOString() })).filter((encounter) => encounter.date).sort((left, right) => new Date(right.date) - new Date(left.date));
  const currentId = currentEncounter ? getId(currentEncounter) : '';
  if (!currentEncounter) throw new Error('currentEncounter is required for history window construction');
  const eligible = ordered.filter((encounter) => !currentId || getId(encounter) !== currentId);
  const recent = eligible.filter((encounter) => new Date(encounter.date).getTime() >= recentCutoff);
  const trend = eligible.filter((encounter) => {
    const time = new Date(encounter.date).getTime();
    return time >= trendCutoff && time < recentCutoff;
  });
  const critical = eligible.filter(isCritical);
  const criticalOutsideWindows = critical.filter((encounter) => new Date(encounter.date).getTime() < trendCutoff);
  const relevant = rankRelevantHistory(trend, currentEncounter).slice(0, maxRelevant).map(({ encounter, relevance }) => ({ ...encounter, selection_relevance: relevance }));

  return {
    as_of: referenceDate.toISOString(),
    windows: { recent_hours: recentHours, trend_days: trendDays },
    metadata: {
      invalid_date_count: invalidDateCount,
      current_encounter_id: currentId || null,
      source_encounter_count: encounters.length,
      recent_encounter_count: recent.length,
      trend_encounter_count: trend.length,
      critical_encounter_count: criticalOutsideWindows.length,
      relevant_encounter_count: relevant.length
    },
    recent_window: { from: new Date(recentCutoff).toISOString(), to: referenceDate.toISOString(), encounter_count: recent.length, encounters: recent },
    trend_window: { from: new Date(trendCutoff).toISOString(), to: new Date(recentCutoff).toISOString(), summary: buildTrendSummary(trend, maxSummaryItems) },
    critical_history: criticalOutsideWindows,
    critical_facts: criticalOutsideWindows.flatMap(getCriticalFacts),
    relevant_history: relevant
  };
};

export const buildInferenceHistoryPayload = (historyWindows, currentEncounter, maxCharacters = 12000) => {
  const payload = {
    current_encounter: currentEncounter,
    recent_history: historyWindows.recent_window.encounters.filter((encounter) => getId(encounter) !== getId(currentEncounter)),
    historical_summary: historyWindows.trend_window.summary,
    critical_history: historyWindows.critical_history || [],
    relevant_history: historyWindows.relevant_history,
    metadata: { ...historyWindows.metadata, as_of: historyWindows.as_of, recent_window: historyWindows.windows.recent_hours, trend_window: historyWindows.windows.trend_days, max_budget: maxCharacters, truncated: false }
  };

  let serialized = JSON.stringify(payload);
  while (serialized.length > maxCharacters && payload.relevant_history.length > 0) {
    payload.relevant_history.pop();
    serialized = JSON.stringify(payload);
  }
  while (serialized.length > maxCharacters && payload.recent_history.length > 0) {
    payload.recent_history.pop();
    serialized = JSON.stringify(payload);
  }
  payload.metadata.estimated_characters = serialized.length;
  payload.metadata.estimated_tokens = Math.ceil(serialized.length / 4);
  payload.metadata.relevant_encounter_count = payload.relevant_history.length;
  payload.metadata.recent_encounter_count = payload.recent_history.length;
  payload.metadata.critical_encounter_count = payload.critical_history.length;
  payload.metadata.omitted_due_to_budget = Math.max(0, (historyWindows.metadata.relevant_encounter_count || 0) - payload.relevant_history.length);
  payload.metadata.truncated = serialized.length > maxCharacters;
  if (payload.metadata.truncated) payload.metadata.budget_note = 'Critical history and current encounter were retained; remaining sections exceed the configured character budget.';
  return payload;
};

export const sizeWindowsToBudget = (maxCharacters = 12000) => ({
  maxCharacters,
  ...getBudgetProfile(maxCharacters)
});

export const buildLongitudinalHistoryContext = ({ encounters = [], currentEncounter = null, asOf = new Date(), maxCharacters = 12000 } = {}) => {
  const budgetProfile = sizeWindowsToBudget(maxCharacters);
  const windows = buildHistoryWindows({
    encounters,
    currentEncounter,
    asOf,
    recentHours: budgetProfile.recentHours,
    maxRelevant: budgetProfile.maxRelevant,
    maxSummaryItems: budgetProfile.maxSummaryItems
  });
  const payload = buildInferenceHistoryPayload(windows, currentEncounter, maxCharacters);
  return { windows, payload, budgetProfile, serialized: JSON.stringify(payload) };
};

export { normalizeMobility };
