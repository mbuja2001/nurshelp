# NurseHelpDesk

NurseHelpDesk is a hospital operations and clinical workflow platform designed to support hospital command-center workflows, patient intake, triage, staffing, bed allocation, pharmacy, radiology, and live analytics for operational and clinical leadership.

## What this project does

This project brings together:

- clinical triage and review workflows
- hospital staffing and shift operations
- patient and encounter management
- bed allocation and department coordination
- pharmacy and pathology/radiology configuration workflows
- live hospital event tracking and operational analytics
- admin analytics for leadership review and configuration

The application is meant to provide real hospital operational visibility with configuration controls for administration while keeping the live reporting experience grounded in actual hospital activity data.

## Live behavior and verification goals

The system is designed to:

- capture real hospital events from encounters, shifts, staff, and operational activity
- derive trustable analytics from live data
- animate charts and counters from zero to the actual measured value
- allow hospital leadership to export a live snapshot of hospital efficiency and review outcomes

## Notes

This code is intended to provide a complete operational and analytics platform for hospital administration and clinical teams. It should be treated as a live workflow product, not as a mock demo, and all reporting should rely on actual captured hospital events whenever available.
All MD files tell a a product dvelopment progression story and anything older than a month should be treated as deprecated and outdated

## Local development

The application code is under `nurshelp/`. Run the frontend and backend from their own directories; the commands below use paths relative to the repository root, so they work regardless of where you clone the repository.

### Requirements

- Node.js 22 LTS (and npm)
- A running MongoDB instance, or a MongoDB connection string

### Configure the backend

Create `nurshelp/backend/.env` and set `MONGO_URI` to your MongoDB connection string:

```dotenv
MONGO_URI=mongodb://127.0.0.1:27017/nursehelpdesk
JWT_SECRET=replace-with-a-long-random-secret
PORT=5001
```

Use a local MongoDB service or a MongoDB Atlas/other MongoDB connection string that is reachable from your machine. `MONGO_URI` is required. `PORT` is optional and defaults to `5001`. Set `JWT_SECRET` to a long, unique secret. Keep `.env` files and real credentials private; do not commit them. The backend exits if `MONGO_URI` is missing and cannot connect until the database is available.

### Install dependencies

From the repository root, open a terminal for each app and install its dependencies:

```sh
cd nurshelp/backend
npm install
```

```sh
cd nurshelp/frontend
npm install
```

### Run the applications

Start the backend in its terminal:

```sh
cd nurshelp/backend
npm run dev
```

Start the frontend in a second terminal:

```sh
cd nurshelp/frontend
npm run dev
```

The backend listens at `http://localhost:5001` by default, and the Vite frontend is available at `http://localhost:5173`. The frontend uses `http://localhost:5001` as its default backend URL. To use a different backend URL, create `nurshelp/frontend/.env.local` and set:

```dotenv
VITE_BACKEND_URL=http://localhost:5001
```

### Run the ESI/LLM triage engine

`nurshelp/backend/ML/ESI_Engine.py` is a separate Flask service. Run it in its own terminal as well as the Node.js backend and frontend. The Node backend calls the engine at `http://localhost:5000/llm-triage`; the browser submits hospital triage to the Node route `POST /api/triage/hospital-request`, not directly to the Python service.

In PowerShell:

```powershell
cd .\nurshelp\backend\ML
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt
python .\ESI_Engine.py
```

If PowerShell activation is unavailable, create the environment and install its dependencies using the same commands, then run its interpreter directly:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe .\ESI_Engine.py
```

`nurshelp/backend/ML/requirements.txt` pins the direct Python runtime dependencies used by the engine and its ML/RAG/document-processing paths. Create a fresh virtual environment for a new checkout; do not rely on the local `ML\venv` being present or committed. The engine can start without the GGUF runtime/model, but MedGemma inference will not be available in that case.

The service listens on port `5000` by default. To choose another port, set `BACKEND_PORT` before starting the process; the Node backend's `PYTHON_URL_LLM` currently points to port `5000`, so that setting must also be updated if the engine uses a different port.

```powershell
$env:BACKEND_PORT = "5000"
python .\ESI_Engine.py
```

#### GGUF model location and loading

The engine resolves its default model path relative to `ESI_Engine.py`, not relative to the shell's current directory:

```text
nurshelp/backend/ML/models/medgemma-4b-local/models--unsloth--medgemma-4b-it-bnb-4bit/medgemma-4b-it-Q4_K_M.gguf
```

That GGUF file is present in this working copy and is about 2.49 GB. At startup, the engine passes the file to `llama_cpp.Llama` and configures `n_gpu_layers=0`, so this code loads the model for CPU inference. The GGUF is the model weights file; `ESI_Engine.py` is the application/service code that provides triage routes and loads the model. The adjacent `models` directory holds the default model asset, while `document_processor` contains an engine-imported helper used by its document/RAG path. `ESI_Engine_Complete` is not referenced by the active `ESI_Engine.py` startup/model lookup.

`GGUF_MODEL_PATH` is checked only after the default model candidate. Consequently, setting it does not override the default when that file already exists; it is a fallback path when the default is absent. Hugging Face/Transformers offline flags are enabled by the engine, so do not expect startup to download missing assets automatically. If the GGUF cannot be found or loaded, inspect the startup output and the capability endpoint before relying on LLM inference.

Check that the server is responding and that the GGUF loaded:

```powershell
Invoke-RestMethod http://localhost:5000/health
Invoke-RestMethod http://localhost:5000/api/capabilities | ConvertTo-Json -Depth 5
```

In the capabilities response, check `models.gguf_loaded`. `/health` confirms service availability but does not report the GGUF status.

#### Triage endpoint and clinical input

The Python endpoint is `POST http://localhost:5000/llm-triage` (no `/api` prefix). It expects JSON with a patient identifier, patient details, vitals, and optional clinical/history fields. The engine rejects an empty/very small request or a missing patient identifier, and returns `409` if the same patient ID is submitted again within 15 seconds. Use a fresh, fictional ID for each direct smoke test; do not put real patient data in a test request.

This PowerShell example sends synthetic data directly to the engine. A direct engine call does not query MongoDB for prior encounters; include `longitudinalHistory` yourself when testing that interface:

```powershell
$qaId = "QA-$([guid]::NewGuid().ToString())"
$request = @{
  patient = @{
    id_number = $qaId
    age = 37
    gender = "female"
    symptoms = "Synthetic example: fever and cough since yesterday"
    history = "Synthetic test patient; no other history supplied"
    criticalHistoryStatus = "reported"
    criticalHistory = @{ allergies = @("Synthetic example: penicillin") }
  }
  patientId = $qaId
  vitals = @{
    temp = 38.2
    hr = 102
    resp = 20
    bp_systolic = 118
    bp_diastolic = 76
    o2 = 97
    mobility = "Walks"
  }
  chiefComplaint = "Synthetic example: fever and cough since yesterday"
  triageNotes = "Synthetic QA request"
  longitudinalHistory = @{
    recent_history = @(
      @{
        id = "qa-previous-encounter"
        date = (Get-Date).AddDays(-1).ToString("o")
        diagnoses = @("Synthetic example: cough")
        symptoms = @("Synthetic example: cough")
        tews = @{ total = 2; priority = "green" }
      }
    )
    relevant_history = @()
    critical_history = @()
    historical_summary = @{}
  }
}
$body = $request | ConvertTo-Json -Depth 10
Invoke-RestMethod -Uri "http://localhost:5000/llm-triage" `
  -Method Post -ContentType "application/json" -Body $body
```

The Node hospital workflow supplies more than this direct example: it validates the hospital request, gets the patient's prior encounters for that hospital using the patient ID, looks up currently clocked-in staff, then transforms the patient/vitals/context into the Python endpoint's schema. Clinical fields include demographics, symptoms or presenting complaint, patient history, transcript, duration, pain level, triage notes, and normalized vitals. The Python engine performs its deterministic vital/TEWS and red-flag checks alongside the model path; the model prompt is instructed to use only supplied facts. The returned assessment is decision support and must be reviewed by clinical staff.

#### Longitudinal patient-history flow

For `POST /api/triage/hospital-request`, the Node controller queries encounters matching both the resolved hospital ID and `patient.id_number`, newest first. It passes those encounters and the current presentation to `longitudinalPatientHistory.js`, which creates the history payload sent as `longitudinalHistory`:

- The current presentation is kept separately from prior encounters. Invalid-date encounters are excluded.
- With the hospital route's 12,000-character budget, the helper considers prior encounters in a 72-hour recent window and a trend window spanning 90 days, excluding the recent window.
- The trend window produces a summary (including TEWS counts/statistics, vital trends, mobility changes, and recurrent diagnoses/interventions/dispositions). Up to five trend-window encounters are separately ranked for relevance using matching problem terms, TEWS, and trauma, with newer encounters breaking ties.
- Encounters marked critical and older than the 90-day trend window are retained in a separate critical-history section. Recent encounters are kept in the recent section.
- If the serialized payload exceeds its character budget, relevant encounters are removed first, then recent encounters. The current presentation and critical-history section are retained; the payload can still exceed the budget if those retained sections alone are too large. Metadata records the estimate and whether truncation remains.

The Python engine deduplicates encounter records across the critical, relevant, and recent sections, then builds a compact table of at most eight rows, prioritizing flagged critical encounters and then the newest non-critical encounters. That table, critical-history status/facts, demographics, complaint, medical history, vitals, and detected red flags form the MedGemma prompt; historical encounter citations are accepted only when their IDs were supplied in the table. The trend summary is included in the Node-generated payload, but the current Python prompt builder does not add its `historical_summary` object to the prompt. Also, `triageNotes` contributes to the engine's combined text for deterministic checks and concept/specialty processing, but is not a separate section in the current MedGemma prompt.

### Build the frontend

The frontend has a production build step; run it from its directory:

```sh
cd nurshelp/frontend
npm run build
```

Vite writes the production files to `nurshelp/frontend/dist/`. The backend is a Node.js service and has no separate compile/build script; `npm run dev` starts it directly. Its entry point is `nurshelp/backend/src/server.js`.

### Changes and verification

Frontend and backend dependencies are installed independently from their respective `package.json` files. The frontend production build was verified with `npm run build`; the backend entry point was verified with `node --check src/server.js`. Build output is generated under `nurshelp/frontend/dist/`, not edited by hand.
