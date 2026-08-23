import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
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
import {
  buildInpatientTimeline,
  groupTimelineByDay,
} from "../../PatientSummary/inpatientReportTimeline";

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

const EHRLibrary = () => {
  const navigate = useNavigate();
  const { userName } = useUserFromToken();
  const token = localStorage.getItem("token");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [record, setRecord] = useState(null);
  const [days, setDays] = useState([]);
  const [selectedDay, setSelectedDay] = useState(null);
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [month, setMonth] = useState(new Date());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const searchPatients = async (event) => {
    event.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError("");
    setRecord(null);
    const searchTerm = query.trim();
    const searchUrl = `${BACKEND_URL}/api/patients/search?search=${encodeURIComponent(searchTerm)}`;
    console.log("[EHR SEARCH] Starting patient search", { searchTerm, searchUrl });
    try {
      const response = await fetch(searchUrl, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await response.json().catch(() => ({}));
      console.log("[EHR SEARCH] Response received", {
        status: response.status,
        ok: response.ok,
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
        searchTerm,
        message: err.message,
      });
      setResults([]);
      setError(err.message || "Unable to search patient records.");
    } finally {
      setLoading(false);
    }
  };

  const openRecord = async (mrn) => {
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
          encounterId: encounter._id,
          encounterStatus: encounter.status,
        })),
      );
      const grouped = groupTimelineByDay(timeline);
      setRecord(data);
      setDays(grouped);
      setSelectedDay(grouped[0]?.key || null);
      setSelectedEvent(grouped[0]?.entries?.[0]?.id || null);
      setMonth(
        grouped[0] ? new Date(`${grouped[0].key}T00:00:00`) : new Date(),
      );
    } catch (err) {
      setError(err.message || "Unable to load the patient record.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const active = days.find((day) => day.key === selectedDay);
    if (active && !active.entries.some((entry) => entry.id === selectedEvent))
      setSelectedEvent(active.entries[0]?.id || null);
  }, [days, selectedDay, selectedEvent]);

  const patient = record?.encounters?.[0];
  const activeDay = days.find((day) => day.key === selectedDay) || days[0];
  const activeEntry =
    activeDay?.entries?.find((entry) => entry.id === selectedEvent) ||
    activeDay?.entries?.[0];
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
      default:
        navigate(`/report/inpatient/${encounterId}`);
    }
  };

  return (
    <SidebarLayout username={userName} handleLogout={() => {}}>
      <div className="ehr-library">
        <style>{`
        .ehr-library{min-height:100vh;background:#f5f8fc;color:#10233f;padding:30px 34px 48px}.ehr-shell{max-width:1420px;margin:0 auto}.ehr-kicker{color:#2563a6;font-size:11px;font-weight:800;letter-spacing:.16em;text-transform:uppercase}.ehr-title{margin:8px 0 6px;font-size:clamp(26px,3vw,42px);line-height:1.05}.ehr-subtitle{margin:0;max-width:720px;color:#5c6d83;font-size:15px;line-height:1.6}.ehr-search{display:flex;gap:10px;margin-top:26px;padding:10px;background:#fff;border:1px solid #d9e2ed;box-shadow:0 12px 32px #203b5b12}.ehr-search input{min-width:0;flex:1;border:0;outline:0;padding:12px 8px;color:#10233f;font-size:15px}.ehr-button{border:0;padding:12px 18px;background:#0b3b70;color:#fff;font-weight:800;cursor:pointer}.ehr-panel{background:#fff;border:1px solid #d9e2ed;box-shadow:0 12px 32px #203b5b0f}.ehr-results{margin-top:24px}.ehr-result{display:flex;align-items:center;justify-content:space-between;gap:18px;padding:18px 20px;border-top:1px solid #e7edf4}.ehr-result:first-child{border-top:0}.ehr-result h3{margin:0 0 5px;font-size:16px}.ehr-meta{color:#63748a;font-size:12px;line-height:1.7}.ehr-link{border:1px solid #9db4cd;background:#eef5fc;color:#0b3b70;padding:9px 13px;font-weight:800;cursor:pointer;white-space:nowrap}.ehr-record-head{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;margin-top:28px;padding:22px 24px;background:#0b294b;color:#fff}.ehr-record-head h2{margin:5px 0 9px;font-size:28px}.ehr-record-head .ehr-kicker{color:#9cc7ef}.ehr-record-meta{display:flex;flex-wrap:wrap;gap:8px 22px;color:#c4d6e9;font-size:12px}.ehr-back{display:flex;align-items:center;gap:7px;border:0;background:transparent;color:#fff;cursor:pointer;font-weight:800}.ehr-workspace{display:grid;grid-template-columns:250px minmax(0,1fr) 285px;gap:18px;margin-top:18px;align-items:start}.ehr-column{padding:18px}.ehr-label{margin-bottom:14px;color:#718198;font-size:10px;font-weight:900;letter-spacing:.14em;text-transform:uppercase}.ehr-day{display:block;width:100%;padding:12px;border:0;border-left:3px solid transparent;background:transparent;color:#3e5067;text-align:left;cursor:pointer}.ehr-day:hover,.ehr-day.active{border-left-color:#0b3b70;background:#edf4fb;color:#0b3b70}.ehr-day strong{display:block;font-size:13px}.ehr-day span{display:block;margin-top:3px;color:#7b8ba0;font-size:11px}.ehr-rail{position:relative;padding:2px 4px 2px 25px}.ehr-rail:before{position:absolute;top:12px;bottom:12px;left:8px;width:1px;background:#cbd8e6;content:''}.ehr-event{position:relative;margin-bottom:14px;padding:15px 16px;border:1px solid #e0e8f1;background:#fff;cursor:pointer}.ehr-event:before{position:absolute;top:20px;left:-22px;width:9px;height:9px;border:3px solid #f5f8fc;border-radius:50%;background:#2563a6;box-shadow:0 0 0 1px #9db4cd;content:''}.ehr-event.active{border-color:#78a6d2;box-shadow:0 7px 18px #1848751a}.ehr-event-top{display:flex;justify-content:space-between;gap:12px;color:#73849a;font-size:11px}.ehr-event h3{margin:7px 0 5px;color:#10233f;font-size:15px}.ehr-event p{margin:0;color:#52657c;font-size:13px;line-height:1.55}.ehr-kind{color:#2563a6;font-weight:900;letter-spacing:.08em;text-transform:uppercase}.ehr-calendar-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:15px}.ehr-calendar-head strong{font-size:13px}.ehr-icon{display:inline-flex;border:1px solid #d9e2ed;padding:5px;background:#fff;color:#36516f;cursor:pointer}.ehr-weekdays,.ehr-calendar{display:grid;grid-template-columns:repeat(7,1fr);gap:4px}.ehr-weekdays{margin-bottom:5px;color:#8a99ab;font-size:9px;text-align:center;text-transform:uppercase}.ehr-calendar button,.ehr-calendar span{min-height:31px;border:0;background:transparent;color:#66788d;font-size:11px;text-align:center}.ehr-calendar button{cursor:pointer}.ehr-calendar button.has-events{background:#e3effa;color:#0b3b70;font-weight:900}.ehr-calendar button.selected{background:#0b3b70;color:#fff}.ehr-detail{margin-top:18px;padding:18px}.ehr-detail h3{margin:5px 0 12px;font-size:18px}.ehr-detail p{margin:0;color:#52657c;font-size:13px;line-height:1.6}.ehr-status{display:inline-block;margin-top:14px;padding:5px 8px;font-size:10px;font-weight:900;letter-spacing:.08em}.ehr-alert{margin-top:18px;padding:12px 14px;border:1px solid #f3c2c2;background:#fff4f4;color:#a33a3a;font-size:13px}@media(max-width:1050px){.ehr-workspace{grid-template-columns:210px minmax(0,1fr)}.ehr-side{grid-column:1/-1}}@media(max-width:720px){.ehr-library{padding:22px 15px 35px}.ehr-search{flex-direction:column}.ehr-record-head{flex-direction:column}.ehr-workspace{grid-template-columns:1fr}.ehr-result{align-items:flex-start;flex-direction:column}.ehr-link{width:100%}}
        .ehr-tiles{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-top:26px}.ehr-tile{min-width:0;padding:17px 18px;background:#fff;border:1px solid #d9e2ed;box-shadow:0 12px 24px #203b5b0f}.ehr-tile-label{color:#718198;font-size:10px;font-weight:900;letter-spacing:.12em;text-transform:uppercase}.ehr-tile-value{margin-top:8px;color:#10233f;font-size:25px;font-weight:800;line-height:1.1;overflow-wrap:anywhere}.ehr-tile-note{margin-top:6px;color:#63748a;font-size:12px;line-height:1.45}
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
                <div className="ehr-tile" style={{ borderTop: "3px solid #2563a6" }}>
                  <div className="ehr-tile-label">Patient index</div>
                  <div className="ehr-tile-value">Search</div>
                  <div className="ehr-tile-note">Find a patient by name or MRN</div>
                </div>
                <div className="ehr-tile" style={{ borderTop: "3px solid #0f766e" }}>
                  <div className="ehr-tile-label">Record model</div>
                  <div className="ehr-tile-value">Timeline</div>
                  <div className="ehr-tile-note">Follow clinical events in order</div>
                </div>
                <div className="ehr-tile" style={{ borderTop: "3px solid #c2410c" }}>
                  <div className="ehr-tile-label">Time precision</div>
                  <div className="ehr-tile-value">HH:MM:SS</div>
                  <div className="ehr-tile-note">Events retain their recorded timestamp</div>
                </div>
                <div className="ehr-tile" style={{ borderTop: "3px solid #7c3aed" }}>
                  <div className="ehr-tile-label">Library scope</div>
                  <div className="ehr-tile-value">Hospital</div>
                  <div className="ehr-tile-note">Scoped to your clinical workspace</div>
                </div>
              </div>
              <form className="ehr-search" onSubmit={searchPatients}>
                <FiSearch size={20} color="#66819e" />
                <input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Search by patient name or MRN"
                  aria-label="Search patients"
                />
                <button className="ehr-button" type="submit" disabled={loading}>
                  {loading ? "Searching..." : "Search records"}
                </button>
              </form>
              {error && <div className="ehr-alert">{error}</div>}
              {results.length > 0 && (
                <div className="ehr-panel ehr-results">
                  <div className="ehr-column">
                    <div className="ehr-label">Matching patient encounters</div>
                  </div>
                  {results.map((encounter) => {
                    const tone = getStatusTone(encounter.status);
                    return (
                      <div className="ehr-result" key={encounter._id}>
                        <div>
                          <h3>{getPatientName(encounter)}</h3>
                          <div className="ehr-meta">
                            MRN:{" "}
                            {encounter.patient?.id_number || "Not recorded"} ·
                            Encounter: {formatDate(encounter.createdAt)} ·{" "}
                            <span style={{ color: tone.color }}>
                              {encounter.status || "ACTIVE"}
                            </span>
                          </div>
                        </div>
                        <button
                          className="ehr-link"
                          type="button"
                          onClick={() =>
                            openRecord(encounter.patient?.id_number || encounter._id)
                          }
                        >
                          Open longitudinal record
                        </button>
                      </div>
                    );
                  })}
                </div>
              )}
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
                  <div className="ehr-label">Record days</div>
                  {days.length ? (
                    days.map((day) => (
                      <button
                        key={day.key}
                        className={`ehr-day ${selectedDay === day.key ? "active" : ""}`}
                        onClick={() => setSelectedDay(day.key)}
                        type="button"
                      >
                        <strong>{day.label}</strong>
                        <span>
                          {day.entries.length} recorded event
                          {day.entries.length === 1 ? "" : "s"}
                        </span>
                      </button>
                    ))
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
