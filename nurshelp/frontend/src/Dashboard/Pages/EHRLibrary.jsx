import React, { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  FiArrowLeft,
  FiCalendar,
  FiChevronLeft,
  FiChevronRight,
  FiClock,
  FiFileText,
  FiSearch,
  FiUser,
} from "react-icons/fi";
import SidebarLayout from "../SidebarLayout";
import { useUserFromToken } from "../../hooks/useUserFromToken";
import { getUserFromToken } from "../../utils/jwt";
import {
  buildInpatientTimeline,
  groupTimelineByDay,
} from "../../PatientSummary/inpatientReportTimeline";
import "./EHRLibrary.css";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

const formatDate = (value) => {
  const date = new Date(value);
  return value && !Number.isNaN(date.getTime())
    ? date.toLocaleDateString(undefined, {
        day: "numeric",
        month: "short",
        year: "numeric",
      })
    : "Date not recorded";
};

const formatDateTime = (value) => {
  const date = new Date(value);
  return value && !Number.isNaN(date.getTime())
    ? `${date.toLocaleDateString(undefined, { day: "2-digit", month: "short", year: "numeric" })} ${date.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false })}`
    : "Timestamp not recorded";
};

const getPatientName = (encounter) => {
  const patient = encounter?.patient || {};
  return (
    [patient.name, patient.surname].filter(Boolean).join(" ") ||
    encounter?.patientName ||
    "Unnamed patient"
  );
};

const getStatusTone = (status = "") => {
  const normalized = status.toUpperCase();
  if (normalized.includes("DISCHARG"))
    return { color: "#047857", background: "#d1fae5" };
  if (normalized.includes("HOSPITAL"))
    return { color: "#1d4ed8", background: "#dbeafe" };
  return { color: "#475569", background: "#f1f5f9" };
};
const getDirectoryDate = (key) => {
  const date = new Date(`${key}T00:00:00`);
  return Number.isNaN(date.getTime()) ? null : date;
};

const buildEncounterDirectory = (encounters = []) => {
  const years = new Map();

  encounters.forEach((encounter) => {
    const timestamp = new Date(encounter.date || encounter.createdAt || encounter.updatedAt);
    if (Number.isNaN(timestamp.getTime())) return;

    const dayKey = timestamp.toISOString().split("T")[0];
    const date = getDirectoryDate(dayKey);
    if (!date) return;
    const yearKey = String(date.getFullYear());
    const monthKey = `${yearKey}-${String(date.getMonth() + 1).padStart(2, "0")}`;
    const monthLabel = date.toLocaleDateString(undefined, { month: "long" });
    const encounterId = String(encounter._id || encounter.encounterId);
    const file = {
      id: encounterId,
      encounterId: encounter._id || encounter.encounterId,
      patientMrn: encounter.patient?.id_number || encounter.patientId || encounter._id,
      title: `Encounter ${encounter.encounterId || encounterId.slice(-8)}`,
      timestamp,
      displayTime: timestamp.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }),
      patientName: getPatientName(encounter),
      status: encounter.status || "ACTIVE",
    };

    if (!years.has(yearKey)) years.set(yearKey, new Map());
    const months = years.get(yearKey);
    if (!months.has(monthKey)) months.set(monthKey, { key: monthKey, label: monthLabel, days: new Map() });
    const month = months.get(monthKey);
    if (!month.days.has(dayKey)) month.days.set(dayKey, { key: dayKey, label: formatDate(`${dayKey}T00:00:00`), entries: [] });
    month.days.get(dayKey).entries.push(file);
  });

  return [...years.entries()]
    .sort(([left], [right]) => right.localeCompare(left))
    .map(([key, months]) => ({
      key,
      months: [...months.values()]
        .sort((left, right) => right.key.localeCompare(left.key))
        .map((month) => ({
          ...month,
          days: [...month.days.values()]
            .sort((left, right) => right.key.localeCompare(left.key))
            .map((day) => ({
              ...day,
              entries: day.entries.sort((left, right) => right.timestamp - left.timestamp),
            })),
        })),
    }));
};

const HospitalEncounterDirectory = ({ encounters, onOpen }) => {
  const directory = buildEncounterDirectory(encounters);

  if (!directory.length) {
    return (
      <div className="ehr-empty-state" style={{ marginTop: 0, border: 0, boxShadow: "none" }}>
        <FiFileText size={24} />
        <strong>No encounters loaded</strong>
        <span>
          This hospital has no encounters matching the current view. Search by patient name or MRN to open a chronological record directory.
        </span>
      </div>
    );
  }

  return (
    <div className="ehr-tree" aria-label="Hospital encounter file explorer">
      {directory.map((year) => (
        <details className="ehr-tree-year" key={year.key} open>
          <summary className="ehr-tree-row ehr-tree-year-row">
            <span className="ehr-tree-caret" aria-hidden="true">›</span>
            <FiCalendar />
            <strong>{year.key}</strong>
            <span className="ehr-tree-count">{year.months.reduce((total, month) => total + month.days.reduce((days, day) => days + day.entries.length, 0), 0)} encounters</span>
          </summary>
          <div className="ehr-tree-months">
            {year.months.map((month) => (
              <details className="ehr-tree-month" key={month.key}>
                <summary className="ehr-tree-row ehr-tree-month-row">
                  <span className="ehr-tree-caret" aria-hidden="true">›</span>
                  <strong>{month.label}</strong>
                  <span className="ehr-tree-count">{month.days.reduce((total, day) => total + day.entries.length, 0)} encounters</span>
                </summary>
                <div className="ehr-tree-days">
                  {month.days.map((day) => (
                    <details className="ehr-tree-day" key={day.key}>
                      <summary className="ehr-tree-row ehr-tree-day-row">
                        <span className="ehr-tree-caret" aria-hidden="true">›</span>
                        <FiCalendar />
                        <strong>{day.label}</strong>
                        <span className="ehr-tree-count">{day.entries.length} encounter{day.entries.length === 1 ? "" : "s"}</span>
                      </summary>
                      <div className="ehr-tree-files">
                        {day.entries.map((file) => (
                          <button className="ehr-tree-file" key={file.id} type="button" onClick={() => onOpen(file.patientMrn)} title={`Open ${file.title}`}>
                            <FiFileText />
                            <span className="ehr-tree-file-copy">
                              <strong>{file.patientName}</strong>
                              <span>{file.title} · {file.status}</span>
                            </span>
                            <time>{file.displayTime}</time>
                            <FiChevronRight className="ehr-tree-file-arrow" />
                          </button>
                        ))}
                      </div>
                    </details>
                  ))}
                </div>
              </details>
            ))}
          </div>
        </details>
      ))}
    </div>
  );
};

const EHRLibrary = () => {
  const navigate = useNavigate();
  const { search } = useLocation();
  const { userName } = useUserFromToken();
  const [token, setToken] = useState(() => localStorage.getItem("token"));
  const userProfile = getUserFromToken(token);
  const userId = userProfile?.id || JSON.parse(localStorage.getItem("user") || "{}")?._id || "anonymous";
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [hospitalEncounters, setHospitalEncounters] = useState([]);
  const [hospitalDirectoryLoading, setHospitalDirectoryLoading] = useState(false);
  const [hospitalDirectoryError, setHospitalDirectoryError] = useState("");
  const [record, setRecord] = useState(null);
  const [days, setDays] = useState([]);
  const [selectedDay, setSelectedDay] = useState(null);
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [month, setMonth] = useState(new Date());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [searchHistory, setSearchHistory] = useState([]);
  const [showSearchHistoryModal, setShowSearchHistoryModal] = useState(false);
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);
  const suggestionRequestRef = React.useRef(null);
  const requestedPatient = new URLSearchParams(search).get("patient");
  const requestedEncounter = new URLSearchParams(search).get("encounter");

  useEffect(() => {
    const refreshToken = () => {
      const nextToken = localStorage.getItem("token");
      console.log("[EHR AUTH] Token refreshed", {
        present: Boolean(nextToken),
        userId: getUserFromToken(nextToken)?.id || null,
        role: getUserFromToken(nextToken)?.role || null,
        hospitalId: getUserFromToken(nextToken)?.hospitalId || null,
      });
      setToken(nextToken);
    };

    window.addEventListener("storage", refreshToken);
    window.addEventListener("nursehelpdesk:auth-changed", refreshToken);
    refreshToken();
    return () => {
      window.removeEventListener("storage", refreshToken);
      window.removeEventListener("nursehelpdesk:auth-changed", refreshToken);
    };
  }, []);

  const loadSearchHistory = async () => {
    console.log("[EHR AUTH] Loading search history", {
      tokenPresent: Boolean(token),
      userId,
      role: userProfile?.role || null,
      hospitalId: userProfile?.hospitalId || null,
    });
    if (!userId || !token) {
      setSearchHistory([]);
      return;
    }

    try {
      const response = await fetch(`${BACKEND_URL}/api/patients/search-history`, {
        headers: { Authorization: `Bearer ${token}` },
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || "Unable to load search history");
      setSearchHistory(Array.isArray(data.history) ? data.history : []);
    } catch (err) {
      console.error("[EHR SEARCH HISTORY] Load failed", err);
      setSearchHistory([]);
    }
  };

  const clearSearchHistory = async () => {
    if (!userId || !token) {
      setSearchHistory([]);
      return;
    }

    try {
      const response = await fetch(`${BACKEND_URL}/api/patients/search-history`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || "Unable to clear search history");
      setSearchHistory(Array.isArray(data.history) ? data.history : []);
      setShowSearchHistoryModal(false);
    } catch (err) {
      console.error("[EHR SEARCH HISTORY] Clear failed", err);
    }
  };

  useEffect(() => {
    loadSearchHistory();
  }, [userId, token]);

  useEffect(() => {
    if (!token) {
      setHospitalEncounters([]);
      return undefined;
    }

    const controller = new AbortController();
    const loadHospitalDirectory = async () => {
      setHospitalDirectoryLoading(true);
      setHospitalDirectoryError("");
      const directoryUrl = `${BACKEND_URL}/api/patients/search?limit=10000&skip=0`;
      console.log("[EHR DIRECTORY] Loading hospital-wide encounter directory", {
        directoryUrl,
        hospitalId: userProfile?.hospitalId || null,
      });
      try {
        const response = await fetch(directoryUrl, {
          headers: { Authorization: `Bearer ${token}` },
          signal: controller.signal,
        });
        const data = await response.json().catch(() => ({}));
        console.log("[EHR DIRECTORY] Hospital directory response", {
          status: response.status,
          hospitalId: data.hospitalId || null,
          count: data.count,
          total: data.total,
        });
        if (!response.ok) throw new Error(data.error || "Unable to load hospital encounters");
        setHospitalEncounters(data.encounters || []);
      } catch (err) {
        if (err.name !== "AbortError") {
          console.error("[EHR DIRECTORY] Hospital directory failed", err);
          setHospitalDirectoryError(err.message || "Unable to load hospital encounters");
          setHospitalEncounters([]);
        }
      } finally {
        if (!controller.signal.aborted) setHospitalDirectoryLoading(false);
      }
    };

    loadHospitalDirectory();
    return () => controller.abort();
  }, [token]);

  useEffect(() => {
    const searchTerm = query.trim();
    if (searchTerm.length < 2 || record) {
      suggestionRequestRef.current?.abort();
      setSuggestions([]);
      setSuggestionsLoading(false);
      return undefined;
    }

    const controller = new AbortController();
    suggestionRequestRef.current?.abort();
    suggestionRequestRef.current = controller;
    const timer = setTimeout(async () => {
      setSuggestionsLoading(true);
      try {
        const response = await fetch(
          `${BACKEND_URL}/api/patients/suggestions?search=${encodeURIComponent(searchTerm)}`,
          { headers: { Authorization: `Bearer ${token}` }, signal: controller.signal },
        );
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.error || "Patient suggestions failed");
        if (!controller.signal.aborted) setSuggestions(data.suggestions || []);
      } catch (err) {
        if (err.name !== "AbortError") console.error("[EHR SEARCH] Suggestions failed", err);
      } finally {
        if (!controller.signal.aborted) setSuggestionsLoading(false);
      }
    }, 250);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [query, record, token]);

  const rememberSearch = async (searchTerm) => {
    const normalized = searchTerm.trim();
    if (!normalized || !token || !userId) return;

    try {
      const response = await fetch(`${BACKEND_URL}/api/patients/search-history`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ term: normalized }),
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || "Unable to save search history");
      if (Array.isArray(data.history)) setSearchHistory(data.history);
    } catch (err) {
      console.error("[EHR SEARCH HISTORY] Save failed", err);
    }
  };

  const doSearch = async (searchTerm, saveHistory = false) => {
    const normalized = searchTerm.trim();
    if (!normalized) {
      setResults([]);
      setError("");
      setLoading(false);
      return;
    }

    setLoading(true);
    setError("");
    setRecord(null);

    if (saveHistory) await rememberSearch(normalized);

    const searchUrl = `${BACKEND_URL}/api/patients/search?search=${encodeURIComponent(normalized)}`;
    console.log("[EHR SEARCH] Starting patient search", { searchTerm: normalized, searchUrl });
    try {
      const response = await fetch(searchUrl, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await response.json().catch(() => ({}));
      console.log("[EHR SEARCH] Response received", {
        status: response.status,
        ok: response.ok,
        requestedHospitalId: userProfile?.hospitalId || null,
        responseHospitalId: data.hospitalId || null,
        count: data.count,
        total: data.total,
        encounters: (data.encounters || []).map((encounter) => ({
          name: encounter.patient?.name,
          surname: encounter.patient?.surname,
          mrn: encounter.patient?.id_number,
          status: encounter.status,
        })),
        error: data.error,
      });
      if (!response.ok) throw new Error(data.error || "Patient search failed");
      setResults(data.encounters || []);
      if (!(data.encounters || []).length)
        setError("No patient records matched that search.");
    } catch (err) {
      console.error("[EHR SEARCH] Request failed", {
        searchTerm: normalized,
        message: err.message,
      });
      setResults([]);
      setError(err.message || "Unable to search patient records.");
    } finally {
      setLoading(false);
    }
  };

  const searchPatients = async (event) => {
    event.preventDefault();
    await doSearch(query, true);
  };

  useEffect(() => {
    const trimmed = query.trim();
    if (!trimmed) {
      setResults([]);
      setError("");
      return undefined;
    }

    const timer = setTimeout(() => {
      doSearch(trimmed, false);
    }, 220);

    return () => clearTimeout(timer);
  }, [query, token]);

  const openRecord = async (mrn, selectedEncounterId = null) => {
    if (!mrn) return;
    setLoading(true);
    setError("");
    try {
      const response = await fetch(
        `${BACKEND_URL}/api/patients/ehr/${encodeURIComponent(mrn)}`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Cache-Control": "no-cache",
          },
        },
      );
      const data = await response.json().catch(() => ({}));
      if (!response.ok)
        throw new Error(data.error || "Unable to load the patient record");
      const timeline = (data.encounters || []).flatMap((encounter) =>
        buildInpatientTimeline(encounter).map((entry) => ({
          ...entry,
          id: `${encounter._id || encounter.encounterId || "encounter"}-${entry.id}`,
          encounterId: encounter._id || encounter.encounterId,
          encounterStatus: encounter.status,
        })),
      );
      const grouped = groupTimelineByDay(timeline).map((day) => ({
        ...day,
        entries: [...day.entries].sort(
          (left, right) => new Date(right.timestamp) - new Date(left.timestamp),
        ),
      }));
      const selectedTimelineEntry = selectedEncounterId
        ? timeline.find((entry) => String(entry.encounterId) === String(selectedEncounterId))
        : null;
      setRecord(data);
      setDays(grouped);
      setSelectedDay(selectedTimelineEntry?.timestamp
        ? new Date(selectedTimelineEntry.timestamp).toISOString().split("T")[0]
        : grouped[0]?.key || null);
      setSelectedEvent(selectedTimelineEntry?.id || grouped[0]?.entries?.[0]?.id || null);
      setMonth(
        grouped[0] ? new Date(`${grouped[0].key}T00:00:00`) : new Date(),
      );
    } catch (err) {
      setError(err.message || "Unable to load the patient record.");
    } finally {
      setLoading(false);
    }
  };

  const openEncounter = async (encounterId) => {
    if (requestedPatient) await openRecord(requestedPatient, encounterId);
  };

  useEffect(() => {
    if (requestedPatient) openRecord(requestedPatient, requestedEncounter);
  }, [requestedPatient, requestedEncounter]);

  useEffect(() => {
    const active = days.find((day) => day.key === selectedDay);
    if (active && !active.entries.some((entry) => entry.id === selectedEvent))
      setSelectedEvent(active.entries[0]?.id || null);
  }, [days, selectedDay, selectedEvent]);

  const patient = record?.encounters?.[0];
  const hasActiveSearch = query.trim().length > 0;
  const activeDay = days.find((day) => day.key === selectedDay) || days[0];
  const activeEntry =
    activeDay?.entries?.find((entry) => entry.id === selectedEvent) ||
    activeDay?.entries?.[0];
  const encounterDirectory = useMemo(() => {
    const daysByKey = new Map();

    (record?.encounters || []).forEach((encounter) => {
      const timestamp = new Date(encounter.date || encounter.createdAt || encounter.updatedAt);
      if (Number.isNaN(timestamp.getTime())) return;
      const key = timestamp.toISOString().split("T")[0];
      const entry = {
        id: String(encounter._id || encounter.encounterId),
        encounterId: encounter._id || encounter.encounterId,
        kind: encounter.manual_triage_report && Object.keys(encounter.manual_triage_report).length
          ? "MANUAL_TRIAGE"
          : "AI_TRIAGE",
        title: `Encounter ${encounter.encounterId || String(encounter._id || "").slice(-8)}`,
        timestamp,
        displayTime: formatDateTime(timestamp).split(" ").slice(1).join(" "),
        body: getPatientName(encounter),
        encounterStatus: encounter.status,
      };
      if (!daysByKey.has(key)) daysByKey.set(key, []);
      daysByKey.get(key).push(entry);
    });

    return [...daysByKey.entries()]
      .sort(([left], [right]) => right.localeCompare(left))
      .map(([key, entries]) => ({
        key,
        label: formatDate(`${key}T00:00:00`),
        entries: entries.sort((left, right) => right.timestamp - left.timestamp),
      }));
  }, [record]);
  const directory = useMemo(() => {
    const years = new Map();

    [...encounterDirectory]
      .sort((left, right) => getDirectoryDate(right.key) - getDirectoryDate(left.key))
      .forEach((day) => {
        const date = getDirectoryDate(day.key);
        if (!date) return;

        const yearKey = String(date.getFullYear());
        const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}`;
        const monthLabel = date.toLocaleDateString(undefined, { month: "long" });
        const year = years.get(yearKey) || { key: yearKey, months: new Map() };
        const month = year.months.get(monthKey) || { key: monthKey, label: monthLabel, days: [] };
        month.days.push(day);
        year.months.set(monthKey, month);
        years.set(yearKey, year);
      });

    return [...years.values()].map((year) => ({
      ...year,
      months: [...year.months.values()].map((month) => ({
        ...month,
        days: month.days.sort((left, right) => getDirectoryDate(right.key) - getDirectoryDate(left.key)),
      })),
    }));
  }, [encounterDirectory]);
  const calendarCells = useMemo(() => {
    const first = new Date(month.getFullYear(), month.getMonth(), 1);
    const offset = (first.getDay() + 6) % 7;
    const total = new Date(
      month.getFullYear(),
      month.getMonth() + 1,
      0,
    ).getDate();
    return Array.from({ length: offset + total }, (_, index) => {
      const day = index - offset + 1;
      if (day < 1) return null;
      const date = new Date(month.getFullYear(), month.getMonth(), day);
      const key = date.toISOString().split("T")[0];
      return { key, day, hasEvents: days.some((item) => item.key === key) };
    });
  }, [month, days]);

  const resetRecord = () => {
    setRecord(null);
    setDays([]);
    setSelectedDay(null);
    setSelectedEvent(null);
  };

  const openOriginalReport = (entry) => {
    const encounterId = entry?.encounterId;
    if (!encounterId) return;

    switch (entry.kind) {
      case "AI_TRIAGE":
        navigate(`/report/triage/${encounterId}`);
        return;
      case "MANUAL_TRIAGE":
        navigate(`/report/manual-triage/${encounterId}`);
        return;
      case "MEDICATION": {
        const prescriptionId = entry.meta?.medication?.prescriptionId;
        if (prescriptionId) navigate(`/prescription/${prescriptionId}`);
        else navigate(`/report/inpatient/${encounterId}`);
        return;
      }
      case "PRESCRIPTION": {
        const prescriptionId = entry.meta?.prescription?._id;
        if (prescriptionId) navigate(`/prescription/${prescriptionId}`);
        return;
      }
      case "DIAGNOSIS":
        navigate(`/report/doctor/${encounterId}`);
        return;
      case "LAB_RESULT": {
        const request = entry.meta?.labRequest;
        navigate(`/report/lab/${encounterId}`, {
          state: {
            request,
            specialty: request?.specialty,
            hospitalContext: { patientName: getPatientName(patient) },
          },
        });
        return;
      }
      case "SURGERY":
        navigate(`/surgery/${encounterId}`);
        return;
      case "DISCHARGE":
        navigate(`/discharge-report/${encounterId}`);
        return;
      default:
        navigate(`/report/inpatient/${encounterId}`);
    }
  };

  return (
    <SidebarLayout username={userName} handleLogout={() => {}}>
      <div className="ehr-library">
        <style>{`
        .ehr-library{min-height:100vh;background:#f5f8fc;color:#10233f;padding:30px 34px 48px}.ehr-shell{max-width:1420px;margin:0 auto}.ehr-kicker{color:#2563a6;font-size:11px;font-weight:800;letter-spacing:.16em;text-transform:uppercase}.ehr-title{margin:8px 0 6px;font-size:clamp(26px,3vw,42px);line-height:1.05}.ehr-subtitle{margin:0;max-width:720px;color:#5c6d83;font-size:15px;line-height:1.6}.ehr-search{display:flex;gap:10px;margin-top:26px;padding:10px;background:#fff;border:1px solid #d9e2ed;box-shadow:0 12px 32px #203b5b12}.ehr-search input{min-width:0;flex:1;border:0;outline:0;padding:12px 8px;color:#10233f;font-size:15px}.ehr-button{border:0;padding:12px 18px;background:#0b3b70;color:#fff;font-weight:800;cursor:pointer}.ehr-panel{background:#fff;border:1px solid #d9e2ed;box-shadow:0 12px 32px #203b5b0f}.ehr-results{margin-top:24px}.ehr-result{display:flex;align-items:center;justify-content:space-between;gap:18px;padding:18px 20px;border-top:1px solid #e7edf4}.ehr-result:first-child{border-top:0}.ehr-result h3{margin:0 0 5px;font-size:16px}.ehr-meta{color:#63748a;font-size:12px;line-height:1.7}.ehr-link{border:1px solid #9db4cd;background:#eef5fc;color:#0b3b70;padding:9px 13px;font-weight:800;cursor:pointer;white-space:nowrap}.ehr-record-head{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;margin-top:28px;padding:22px 24px;background:#0b294b;color:#fff}.ehr-record-head h2{margin:5px 0 9px;font-size:28px}.ehr-record-head .ehr-kicker{color:#9cc7ef}.ehr-record-meta{display:flex;flex-wrap:wrap;gap:8px 22px;color:#c4d6e9;font-size:12px}.ehr-back{display:flex;align-items:center;gap:7px;border:0;background:transparent;color:#fff;cursor:pointer;font-weight:800}.ehr-workspace{display:grid;grid-template-columns:250px minmax(0,1fr) 285px;gap:18px;margin-top:18px;align-items:start}.ehr-column{padding:18px}.ehr-label{margin-bottom:14px;color:#718198;font-size:10px;font-weight:900;letter-spacing:.14em;text-transform:uppercase}.ehr-day{display:block;width:100%;padding:12px;border:0;border-left:3px solid transparent;background:transparent;color:#3e5067;text-align:left;cursor:pointer}.ehr-day:hover,.ehr-day.active{border-left-color:#0b3b70;background:#edf4fb;color:#0b3b70}.ehr-day strong{display:block;font-size:13px}.ehr-day span{display:block;margin-top:3px;color:#7b8ba0;font-size:11px}.ehr-rail{position:relative;padding:2px 4px 2px 25px}.ehr-rail:before{position:absolute;top:12px;bottom:12px;left:8px;width:1px;background:#cbd8e6;content:''}.ehr-event{position:relative;margin-bottom:14px;padding:15px 16px;border:1px solid #e0e8f1;background:#fff;cursor:pointer}.ehr-event:before{position:absolute;top:20px;left:-22px;width:9px;height:9px;border:3px solid #f5f8fc;border-radius:50%;background:#2563a6;box-shadow:0 0 0 1px #9db4cd;content:''}.ehr-event.active{border-color:#78a6d2;box-shadow:0 7px 18px #1848751a}.ehr-event-top{display:flex;justify-content:space-between;gap:12px;color:#73849a;font-size:11px}.ehr-event h3{margin:7px 0 5px;color:#10233f;font-size:15px}.ehr-event p{margin:0;color:#52657c;font-size:13px;line-height:1.55}.ehr-kind{color:#2563a6;font-weight:900;letter-spacing:.08em;text-transform:uppercase}.ehr-calendar-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:15px}.ehr-calendar-head strong{font-size:13px}.ehr-icon{display:inline-flex;border:1px solid #d9e2ed;padding:5px;background:#fff;color:#36516f;cursor:pointer}.ehr-weekdays,.ehr-calendar{display:grid;grid-template-columns:repeat(7,1fr);gap:4px}.ehr-weekdays{margin-bottom:5px;color:#8a99ab;font-size:9px;text-align:center;text-transform:uppercase}.ehr-calendar button,.ehr-calendar span{min-height:31px;border:0;background:transparent;color:#66788d;font-size:11px;text-align:center}.ehr-calendar button{cursor:pointer}.ehr-calendar button.has-events{background:#e3effa;color:#0b3b70;font-weight:900}.ehr-calendar button.selected{background:#0b3b70;color:#fff}.ehr-detail{margin-top:18px;padding:18px}.ehr-detail h3{margin:5px 0 12px;font-size:18px}.ehr-detail p{margin:0;color:#52657c;font-size:13px;line-height:1.6}.ehr-status{display:inline-block;margin-top:14px;padding:5px 8px;font-size:10px;font-weight:900;letter-spacing:.08em}.ehr-alert{margin-top:18px;padding:12px 14px;border:1px solid #f3c2c2;background:#fff4f4;color:#a33a3a;font-size:13px}@media(max-width:1050px){.ehr-workspace{grid-template-columns:210px minmax(0,1fr)}.ehr-side{grid-column:1/-1}}@media(max-width:720px){.ehr-library{padding:22px 15px 35px}.ehr-search{flex-direction:column}.ehr-record-head{flex-direction:column}.ehr-workspace{grid-template-columns:1fr}.ehr-result{align-items:flex-start;flex-direction:column}.ehr-link{width:100%}}
        .ehr-tiles{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-top:26px}.ehr-tile{min-width:0;padding:17px 18px;background:#fff;border:1px solid #d9e2ed;box-shadow:0 12px 24px #203b5b0f}.ehr-tile-label{color:#718198;font-size:10px;font-weight:900;letter-spacing:.12em;text-transform:uppercase}.ehr-tile-value{margin-top:8px;color:#10233f;font-size:25px;font-weight:800;line-height:1.1;overflow-wrap:anywhere}.ehr-tile-note{margin-top:6px;color:#63748a;font-size:12px;line-height:1.45}.ehr-empty-state{display:grid;justify-items:center;gap:8px;margin-top:24px;padding:38px 24px;text-align:center;color:#718198}.ehr-empty-state strong{color:#173b63;font-size:16px}.ehr-empty-state span{max-width:560px;font-size:13px;line-height:1.6}
        .ehr-directory{display:grid;gap:8px}.ehr-directory details{min-width:0}.ehr-directory summary{display:flex;align-items:center;gap:8px;list-style:none;cursor:pointer;color:#173b63}.ehr-directory summary::-webkit-details-marker{display:none}.ehr-directory summary::before{content:"›";color:#7190ad;font-size:18px;line-height:1;transition:transform .15s ease}.ehr-directory details[open]>summary::before{transform:rotate(90deg)}.ehr-directory summary span{margin-left:auto;color:#718198;font-size:11px;font-weight:700}.ehr-directory-year{padding:10px 0;border-bottom:1px solid #e7edf4}.ehr-directory-year>summary{font-size:15px}.ehr-directory-months{display:grid;gap:6px;padding:9px 0 0 18px}.ehr-directory-month>summary{padding:7px 8px;font-size:13px;background:#f5f8fc}.ehr-directory-days{display:grid;gap:4px;padding:5px 0 2px 12px}.ehr-day{display:block;width:100%;padding:10px 12px;border:0;border-left:3px solid transparent;background:transparent;color:#173b63;text-align:left;cursor:pointer}.ehr-day:hover{background:#f3f7fb}.ehr-day.active{border-left-color:#2563a6;background:#e9f2fb}.ehr-day strong,.ehr-day span{display:block}.ehr-day strong{font-size:12px}.ehr-day span{margin-top:3px;color:#718198;font-size:11px}
      `}</style>
        <style>{`
          .ehr-recent-directory { display: grid; gap: 6px; }
          .ehr-recent-folder { display: flex; align-items: center; gap: 10px; width: 100%; padding: 10px 12px; border: 1px solid #d9e2ed; background: #f8fbff; color: #10233f; text-align: left; cursor: pointer; }
          .ehr-recent-folder:hover { border-color: #8fb0d1; background: #eef6ff; }
          .ehr-recent-folder-icon { display: inline-flex; align-items: center; color: #2563a6; }
          .ehr-recent-folder-name { min-width: 0; flex: 1; overflow-wrap: anywhere; font-weight: 700; }
          .ehr-recent-folder-action { color: #63748a; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .08em; }
          .ehr-directory-date { border-left: 3px solid transparent; }
          .ehr-directory-date.active { border-left-color: #2563a6; background: #e9f2fb; }
          .ehr-directory-date > summary { padding: 9px 10px; font-size: 12px; }
          .ehr-directory-date > summary:hover { background: #f3f7fb; }
          .ehr-directory-files { display: grid; gap: 2px; padding: 2px 6px 7px 24px; }
          .ehr-file { display: flex; align-items: center; gap: 7px; width: 100%; min-width: 0; padding: 7px 8px; border: 0; background: transparent; color: #173b63; text-align: left; cursor: pointer; }
          .ehr-file:hover, .ehr-file.active { background: #dbeafe; }
          .ehr-file svg { flex: 0 0 auto; color: #2563a6; }
          .ehr-file-name { min-width: 0; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 11px; }
          .ehr-file time { color: #718198; font-size: 10px; white-space: nowrap; }
        `}</style>
        <div className="ehr-shell">
          {!record ? (
            <>
              <div className="ehr-kicker">Clinical records library</div>
              <h1 className="ehr-title">Electronic Health Record</h1>
              <p className="ehr-subtitle">
                Search the hospital record, then follow each encounter as a
                precise clinical chronology. Every event stays connected to its
                source encounter and recorded time.
              </p>
              <div className="ehr-tiles">
                <div className="ehr-tile">
                  <div className="ehr-tile-label">Patient index</div>
                  <div className="ehr-tile-value">Search</div>
                  <div className="ehr-tile-note">Find a patient by name or MRN</div>
                </div>
                <div className="ehr-tile">
                  <div className="ehr-tile-label">Record model</div>
                  <div className="ehr-tile-value">Timeline</div>
                  <div className="ehr-tile-note">Follow clinical events in order</div>
                </div>
                <div className="ehr-tile">
                  <div className="ehr-tile-label">Time precision</div>
                  <div className="ehr-tile-value">HH:MM:SS</div>
                  <div className="ehr-tile-note">Events retain their recorded timestamp</div>
                </div>
                <div className="ehr-tile">
                  <div className="ehr-tile-label">Library scope</div>
                  <div className="ehr-tile-value">Hospital</div>
                  <div className="ehr-tile-note">Scoped to your clinical workspace</div>
                </div>
              </div>
              <form className="ehr-search" onSubmit={searchPatients}>
                <FiSearch size={20} color="#66819e" />
                <input
                  value={query}
                  onChange={(event) => {
                    const nextValue = event.target.value;
                    setQuery(nextValue);
                    setError("");
                    if (!nextValue.trim()) {
                      setResults([]);
                    }
                  }}
                  placeholder="Search by patient name or MRN"
                  aria-label="Search patients"
                />
                <button className="ehr-button" type="submit" disabled={loading}>
                  {loading ? "Searching..." : "Search records"}
                </button>
                <button
                  type="button"
                  className="ehr-history-button"
                  onClick={() => setShowSearchHistoryModal(true)}
                >
                  History
                </button>
              </form>
              {!record && (
                <>
                  {!hasActiveSearch && (
                    <div style={{ marginTop: 18 }}>
                      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, marginBottom: 8 }}>
                        <div className="ehr-label" style={{ marginBottom: 0 }}>Recent searches</div>
                        {searchHistory.length > 0 && (
                          <button
                            type="button"
                            onClick={clearSearchHistory}
                            style={{
                              border: "1px solid #e2e8f0",
                              background: "#fff",
                              color: "#b91c1c",
                              borderRadius: "8px",
                              padding: "6px 8px",
                              fontWeight: 700,
                              cursor: "pointer",
                              fontSize: "11px"
                            }}
                          >
                            Clear
                          </button>
                        )}
                      </div>
                      {searchHistory.length > 0 ? (
                        <div className="ehr-recent-directory">
                          {searchHistory.map((item) => (
                            <button
                              key={item}
                              type="button"
                              className="ehr-recent-folder"
                              onClick={() => {
                                setQuery(item);
                                setTimeout(() => document.querySelector(".ehr-search")?.requestSubmit(), 0);
                              }}
                            >
                              <span className="ehr-recent-folder-icon"><FiClock size={14} /></span>
                              <span className="ehr-recent-folder-name">{item}</span>
                              <span className="ehr-recent-folder-action">Open</span>
                            </button>
                          ))}
                        </div>
                      ) : (
                        <div className="ehr-recent-empty">No recent searches yet.</div>
                      )}
                    </div>
                  )}

                  {hasActiveSearch ? (
                    <div className="ehr-panel ehr-results">
                      <div className="ehr-column">
                        <div className="ehr-label">Matching patient encounters</div>
                      </div>
                      {loading ? (
                        <div className="ehr-meta" style={{ padding: "0 18px 18px" }}>Searching records...</div>
                      ) : results.length > 0 ? (
                        results.map((encounter) => {
                          const tone = getStatusTone(encounter.status);
                          return (
                            <div className="ehr-result" key={encounter._id}>
                              <div>
                                <h3>{getPatientName(encounter)}</h3>
                                <div className="ehr-meta">
                                  MRN: {encounter.patient?.id_number || "Not recorded"} · Encounter: {formatDate(encounter.date || encounter.createdAt)} · <span style={{ color: tone.color }}>{encounter.status || "ACTIVE"}</span>
                                </div>
                              </div>
                              <button
                                className="ehr-link"
                                type="button"
                                onClick={() => openRecord(encounter.patient?.id_number || encounter._id)}
                              >
                                Open longitudinal record
                              </button>
                            </div>
                          );
                        })
                      ) : (
                        <div className="ehr-meta" style={{ padding: "0 18px 18px" }}>
                          No patient records matched that search.
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="ehr-panel ehr-hospital-explorer">
                      <div className="ehr-explorer-header">
                        <div>
                          <h2 className="ehr-explorer-title">Encounter File Explorer</h2>
                          <div className="ehr-explorer-subtitle">All encounters · Year / Month / Date / Time</div>
                        </div>
                        {hospitalEncounters.length > 0 && (
                          <span className="ehr-explorer-count">{hospitalEncounters.length} loaded</span>
                        )}
                      </div>
                      {hospitalDirectoryLoading ? (
                        <div className="ehr-explorer-loading" role="status" aria-live="polite">
                          <span className="ehr-loading-mark" aria-hidden="true" />
                          <span>
                            <strong>Loading hospital encounters</strong>
                            <small>Preparing the year, month, date, and time directory.</small>
                          </span>
                        </div>
                      ) : hospitalDirectoryError ? (
                        <div className="ehr-alert">{hospitalDirectoryError}</div>
                      ) : (
                        <HospitalEncounterDirectory encounters={hospitalEncounters} onOpen={openRecord} />
                      )}
                    </div>
                  )}
                </>
              )}
              {query.trim().length >= 2 && !record && (
                <div className="ehr-panel" style={{ marginTop: 8, padding: "8px 0" }}>
                  <div className="ehr-label" style={{ padding: "0 18px", marginBottom: 6 }}>
                    Search suggestions
                  </div>
                  {suggestionsLoading ? (
                    <div className="ehr-meta" style={{ padding: "4px 18px" }}>Finding matching records...</div>
                  ) : suggestions.length === 0 ? (
                    <div className="ehr-meta" style={{ padding: "4px 18px" }}>No matching patients yet.</div>
                  ) : null}
                  {suggestions.map((suggestion) => (
                    <button
                      key={`${suggestion.value}-${suggestion.label}`}
                      type="button"
                      onClick={() => {
                        setQuery(suggestion.value);
                        setSuggestions([]);
                      }}
                      style={{ display: "flex", width: "100%", justifyContent: "space-between", gap: 12, padding: "10px 18px", border: 0, borderTop: "1px solid #e7edf4", background: "#fff", color: "#10233f", textAlign: "left", cursor: "pointer" }}
                    >
                      <span><strong>{suggestion.label}</strong><br /><small className="ehr-meta">{suggestion.mrn || "Encounter record"}</small></span>
                      <span className="ehr-meta">{suggestion.status}</span>
                    </button>
                  ))}
                </div>
              )}
              {showSearchHistoryModal && (
                <div style={{
                  position: "fixed",
                  inset: 0,
                  background: "rgba(15, 23, 42, 0.72)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  zIndex: 1200,
                  padding: "20px"
                }}>
                  <div style={{
                    width: "100%",
                    maxWidth: "640px",
                    background: "#ffffff",
                    borderRadius: "16px",
                    boxShadow: "0 20px 50px rgba(15, 23, 42, 0.24)",
                    padding: "20px",
                    color: "#0f172a"
                  }}>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px", gap: "12px" }}>
                      <div>
                        <div style={{ fontSize: "17px", fontWeight: 800 }}>Search History</div>
                        <div style={{ fontSize: "13px", color: "#64748b", marginTop: "4px" }}>Review your recent patient searches.</div>
                      </div>
                      {searchHistory.length > 0 && (
                        <button
                          type="button"
                          onClick={clearSearchHistory}
                          style={{
                            border: "1px solid #e2e8f0",
                            background: "#fff",
                            color: "#b91c1c",
                            borderRadius: "8px",
                            padding: "8px 10px",
                            fontWeight: 700,
                            cursor: "pointer",
                            fontSize: "12px"
                          }}
                        >
                          Clear history
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() => setShowSearchHistoryModal(false)}
                        style={{
                          border: "none",
                          background: "transparent",
                          color: "#64748b",
                          cursor: "pointer",
                          fontSize: "16px"
                        }}
                        aria-label="Close search history"
                      >
                        ×
                      </button>
                    </div>

                    <div style={{ marginTop: "14px", display: "grid", gap: "10px", maxHeight: "320px", overflowY: "auto" }}>
                      {searchHistory.length === 0 ? (
                        <div style={{ padding: "16px", borderRadius: "10px", background: "#f8fafc", color: "#64748b", fontSize: "13px" }}>
                          No recent search history yet.
                        </div>
                      ) : (
                        searchHistory.map((item) => (
                          <button
                            key={item}
                            type="button"
                            onClick={() => {
                              setQuery(item);
                              setShowSearchHistoryModal(false);
                              setTimeout(() => document.querySelector(".ehr-search")?.requestSubmit(), 0);
                            }}
                            style={{
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "space-between",
                              gap: "12px",
                              width: "100%",
                              padding: "14px 16px",
                              border: "1px solid #e2e8f0",
                              borderRadius: "10px",
                              background: "#f9fafb",
                              cursor: "pointer",
                              textAlign: "left",
                              color: "#0f172a",
                              fontWeight: 600,
                              fontSize: "14px"
                            }}
                          >
                            <span style={{ display: "inline-flex", alignItems: "center", gap: "8px" }}>
                              <FiClock size={14} color="#64748b" />
                              {item}
                            </span>
                            <span style={{ color: "#64748b", fontSize: "12px", fontWeight: 700 }}>Open</span>
                          </button>
                        ))
                      )}
                    </div>
                  </div>
                </div>
              )}
              {error && <div className="ehr-alert">{error}</div>}
            </>
          ) : (
            <>
              <div className="ehr-record-head">
                <div>
                  <div className="ehr-kicker">Longitudinal patient record</div>
                  <h2>{getPatientName(patient)}</h2>
                  <div className="ehr-record-meta">
                    <span>MRN: {record.mrn}</span>
                    <span>
                      {record.totalEncounters || record.encounters?.length || 0}{" "}
                      encounter(s)
                    </span>
                    <span>Hospital-scoped clinical history</span>
                  </div>
                </div>
                <button
                  className="ehr-back"
                  type="button"
                  onClick={resetRecord}
                >
                  <FiArrowLeft /> Back to search
                </button>
              </div>
              <div className="ehr-workspace">
                <aside className="ehr-panel ehr-column">
                  <div className="ehr-label">Encounter directory</div>
                  {directory.length ? (
                    <div className="ehr-directory">
                      {directory.map((year) => (
                        <details className="ehr-directory-year" key={year.key} open>
                          <summary>
                            <FiCalendar />
                            <strong>{year.key}</strong>
                            <span>{year.months.reduce((total, monthItem) => total + monthItem.days.length, 0)} days</span>
                          </summary>
                          <div className="ehr-directory-months">
                            {year.months.map((monthItem) => (
                              <details className="ehr-directory-month" key={monthItem.key} open>
                                <summary>
                                  <strong>{monthItem.label}</strong>
                                  <span>{monthItem.days.length} day{monthItem.days.length === 1 ? "" : "s"}</span>
                                </summary>
                                <div className="ehr-directory-days">
                                  {monthItem.days.map((day) => (
                                    <details className={`ehr-directory-date ${selectedDay === day.key ? "active" : ""}`} key={day.key} open={selectedDay === day.key}>
                                      <summary onClick={() => setSelectedDay(day.key)}>
                                        <FiCalendar />
                                        <strong>{day.label}</strong>
                                        <span>{day.entries.length} encounter{day.entries.length === 1 ? "" : "s"}</span>
                                      </summary>
                                      <div className="ehr-directory-files">
                                        {day.entries.map((entry) => (
                                          <button
                                            key={entry.id}
                                            type="button"
                                            className={`ehr-file ${selectedEvent === entry.id ? "active" : ""}`}
                                            onClick={() => {
                                              setSelectedDay(day.key);
                                              setSelectedEvent(entry.id);
                                              openOriginalReport(entry);
                                            }}
                                          >
                                            <FiFileText />
                                            <span className="ehr-file-name">{entry.title}</span>
                                            <time>{entry.displayTime}</time>
                                          </button>
                                        ))}
                                      </div>
                                    </details>
                                  ))}
                                </div>
                              </details>
                            ))}
                          </div>
                        </details>
                      ))}
                    </div>
                  ) : (
                    <div className="ehr-meta">
                      No timestamped clinical events recorded yet.
                    </div>
                  )}
                </aside>
                <main className="ehr-panel ehr-column">
                  <div className="ehr-label">
                    Clinical chronology{" "}
                    {activeDay ? `· ${activeDay.label}` : ""}
                  </div>
                  {activeDay ? (
                    <div className="ehr-rail">
                      {activeDay.entries.map((entry) => (
                        <article
                          className={`ehr-event ${activeEntry?.id === entry.id ? "active" : ""}`}
                          key={entry.id}
                          onClick={() => {
                            setSelectedEvent(entry.id);
                            openOriginalReport(entry);
                          }}
                          onKeyDown={(event) => {
                            if (event.key === "Enter" || event.key === " ") {
                              event.preventDefault();
                              setSelectedEvent(entry.id);
                              openOriginalReport(entry);
                            }
                          }}
                          role="button"
                          tabIndex={0}
                          aria-label={`Open ${entry.title}`}
                        >
                          <div className="ehr-event-top">
                            <span className="ehr-kind">
                              {entry.kind.replace("_", " ")}
                            </span>
                            <span>
                              <FiClock style={{ verticalAlign: "middle" }} />{" "}
                              {entry.displayTime}
                            </span>
                          </div>
                          <h3>{entry.title}</h3>
                          <p>{entry.body}</p>
                        </article>
                      ))}
                    </div>
                  ) : (
                    <div className="ehr-meta">
                      This record has no timestamped events to display.
                    </div>
                  )}
                </main>
                <aside className="ehr-panel ehr-column ehr-side">
                  <div className="ehr-calendar-head">
                    <button
                      className="ehr-icon"
                      type="button"
                      onClick={() =>
                        setMonth(
                          (value) =>
                            new Date(
                              value.getFullYear(),
                              value.getMonth() - 1,
                              1,
                            ),
                        )
                      }
                      aria-label="Previous month"
                    >
                      <FiChevronLeft />
                    </button>
                    <strong>
                      {month.toLocaleDateString(undefined, {
                        month: "long",
                        year: "numeric",
                      })}
                    </strong>
                    <button
                      className="ehr-icon"
                      type="button"
                      onClick={() =>
                        setMonth(
                          (value) =>
                            new Date(
                              value.getFullYear(),
                              value.getMonth() + 1,
                              1,
                            ),
                        )
                      }
                      aria-label="Next month"
                    >
                      <FiChevronRight />
                    </button>
                  </div>
                  <div className="ehr-weekdays">
                    {["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"].map((day) => (
                      <span key={day}>{day}</span>
                    ))}
                  </div>
                  <div className="ehr-calendar">
                    {calendarCells.map((cell, index) =>
                      cell ? (
                        <button
                          key={cell.key}
                          className={`${cell.hasEvents ? "has-events" : ""} ${selectedDay === cell.key ? "selected" : ""}`}
                          type="button"
                          onClick={() =>
                            cell.hasEvents && setSelectedDay(cell.key)
                          }
                        >
                          {cell.day}
                        </button>
                      ) : (
                        <span key={`empty-${index}`} />
                      ),
                    )}
                  </div>
                  <div className="ehr-panel ehr-detail">
                    <div className="ehr-label">Selected event</div>
                    <FiFileText size={24} color="#2563a6" />
                    <h3>{activeEntry?.title || "No event selected"}</h3>
                    <p>
                      {activeEntry
                        ? activeEntry.body
                        : "Choose a day and event to inspect its clinical detail."}
                    </p>
                    {activeEntry && (
                      <>
                        <div className="ehr-meta" style={{ marginTop: 13 }}>
                          <FiCalendar style={{ verticalAlign: "middle" }} />{" "}
                          {formatDateTime(activeEntry.timestamp)}
                        </div>
                        <span
                          className="ehr-status"
                          style={getStatusTone(activeEntry.encounterStatus)}
                        >
                          {activeEntry.encounterStatus || "RECORDED EVENT"}
                        </span>
                      </>
                    )}
                  </div>
                  <div className="ehr-panel ehr-detail">
                    <div className="ehr-label">Patient identity</div>
                    <div className="ehr-meta">
                      <FiUser style={{ verticalAlign: "middle" }} />{" "}
                      {getPatientName(patient)}
                      <br />
                      MRN: {record.mrn}
                      <br />
                      DOB: {patient?.patient?.dob || "Not recorded"}
                    </div>
                  </div>
                </aside>
              </div>
            </>
          )}
        </div>
      </div>
    </SidebarLayout>
  );
};

export default EHRLibrary;
