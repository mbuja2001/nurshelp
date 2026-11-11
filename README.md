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

### Build the frontend

The frontend has a production build step; run it from its directory:

```sh
cd nurshelp/frontend
npm run build
```

Vite writes the production files to `nurshelp/frontend/dist/`. The backend is a Node.js service and has no separate compile/build script; `npm run dev` starts it directly. Its entry point is `nurshelp/backend/src/server.js`.

### Changes and verification

Frontend and backend dependencies are installed independently from their respective `package.json` files. The frontend production build was verified with `npm run build`; the backend entry point was verified with `node --check src/server.js`. Build output is generated under `nurshelp/frontend/dist/`, not edited by hand.
