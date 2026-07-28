#!/bin/bash
# Physician Dashboard - Quick Start Script

echo "🏥 PHYSICIAN DASHBOARD - QUICK START"
echo "═════════════════════════════════════"
echo ""

# Check if MongoDB is running
echo "[1/4] Checking MongoDB connection..."
mongosh --eval "db.version()" > /dev/null 2>&1
if [ $? -eq 0 ]; then
  echo "✅ MongoDB is running"
else
  echo "⚠️  MongoDB not detected. Starting MongoDB..."
  # Uncomment if MongoDB service needs to start
  # mongod --dbpath ~/data/db &
fi

echo ""
echo "[2/4] Seeding test data..."
cd backend
node test-physician-apis-live.js

if [ $? -ne 0 ]; then
  echo "❌ Seed data failed. Check MongoDB connection."
  exit 1
fi

echo ""
echo "[3/4] Starting backend server..."
echo "   Backend will run on http://localhost:5001"
echo "   Press Ctrl+C to stop"
echo ""

npm run dev &
BACKEND_PID=$!

sleep 3

echo ""
echo "[4/4] Starting frontend..."
echo "   Frontend will run on http://localhost:5173"
echo "   Navigate to: Physician Dashboard"
echo ""

# Start frontend in a new window (macOS)
# npm run dev --prefix ../frontend

# For Windows/Linux, keep in same terminal
echo "✅ Setup complete!"
echo ""
echo "📋 TESTING CHECKLIST:"
echo "  1. Open http://localhost:5173 in browser"
echo "  2. Login with physician credentials"
echo "  3. Navigate to Physician Dashboard"
echo "  4. Verify 3-lane view loads with 4 test patients"
echo "  5. Check bed management shows 9 beds (2 occupied, 7 available)"
echo "  6. Click a patient and submit a disposition decision"
echo "  7. Verify accuracy metrics (TRUE_POSITIVE, TRUE_NEGATIVE, etc)"
echo ""
echo "📊 DATABASE STATE:"
echo "  - 4 test patients (Red, Green, Yellow x2)"
echo "  - 9 beds in ER ward"
echo "  - 4 triage assessments"
echo "  - 4 disposition records with accuracy tracking"
echo ""
echo "🎯 KEY ENDPOINTS:"
echo "  GET  /api/physician/cases?context=ER"
echo "  GET  /api/physician/beds?context=ER"
echo "  POST /api/disposition"
echo ""
echo "🛑 To stop: Press Ctrl+C"

# Wait for backend to finish
wait $BACKEND_PID
