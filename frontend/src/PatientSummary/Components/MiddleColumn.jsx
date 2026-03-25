// src/PatientSummary/Components/MiddleColumn.jsx
import React from "react";

export default function MiddleColumn({ data = [], setData }) {
  // update text for an item
  const handleUpdate = (id, newText) => {
    if (typeof setData !== "function") return;
    setData(prev => prev.map(item => (item.id === id ? { ...item, text: newText } : item)));
  };

  // add new item (question or answer)
  const addItem = (type) => {
    if (typeof setData !== "function") return;
    const nextId = data.length ? Math.max(...data.map(i => i.id)) + 1 : 1;
    const placeholder = type === "question" ? "New question..." : "New answer...";
    setData(prev => [...prev, { id: nextId, type, text: placeholder }]);
  };

  // delete item
  const deleteItem = (id) => {
    if (!window.confirm("Delete this line?")) return;
    if (typeof setData !== "function") return;
    setData(prev => prev.filter(i => i.id !== id));
  };

  return (
    <div className="middle-column-layout">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h4>Transcript</h4>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="edit-action-btn" onClick={() => addItem("question")} title="Add question">＋Q</button>
          <button className="edit-action-btn" onClick={() => addItem("answer")} title="Add answer">＋A</button>
        </div>
      </div>

      <div className="transcript-container">
        {(data || []).map((item) => (
          <div key={item.id} className={`chat-block ${item.type === "question" ? "q-block" : "a-block"}`}>
            <div className="block-header-row">
              <span className="block-tag">{item.type.toUpperCase()}</span>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <span className="edit-hint">Click to edit</span>
                <button
                  className="delete-btn"
                  onClick={() => deleteItem(item.id)}
                  aria-label={`Delete ${item.type}`}
                  title="Delete"
                >
                  ×
                </button>
              </div>
            </div>

            <div className="editable-wrapper">
              <textarea
                className="editable-answer"
                value={item.text}
                onChange={(e) => handleUpdate(item.id, e.target.value)}
                aria-label={`${item.type} ${item.id}`}
                rows={item.type === "question" ? 2 : 3}
                placeholder={item.type === "question" ? "Type the question..." : "Type the answer..."}
              />
              <span className="edit-pencil">✎</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}