@echo off
REM Physician Dashboard - Quick Start Script for Windows

echo.
echo ========================================
echo 🏥 PHYSICIAN DASHBOARD - QUICK START
echo ========================================
echo.

echo [1/4] Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js not found. Please install Node.js first.
    exit /b 1
)
echo ✅ Node.js ready

echo.
echo [2/4] Seeding test data...
echo   Running: node backend\test-physician-apis-live.js
cd backend
call node test-physician-apis-live.js
if errorlevel 1 (
    echo ❌ Seed data failed. Check MongoDB connection.
    cd ..
    exit /b 1
)
cd ..

echo.
echo [3/4] Starting backend server...
echo   Backend will run on http://localhost:5001
echo.
start cmd /k "cd backend && npm run dev"

echo.
echo [4/4] Waiting for backend to start (5 seconds)...
timeout /t 5 /nobreak

echo.
echo [5/5] Starting frontend server...
echo   Frontend will run on http://localhost:5173
echo.
start cmd /k "npm run dev"

timeout /t 3

echo.
echo ✅ SETUP COMPLETE!
echo.
echo 📋 TESTING CHECKLIST:
echo.
echo   1. Open http://localhost:5173 in browser
echo   2. Login with physician credentials
echo   3. Navigate to Physician Dashboard
echo   4. Verify 3-lane view loads with 4 test patients
echo   5. Check bed management shows 9 beds (2 occupied, 7 available)
echo   6. Click a patient and submit a disposition decision
echo   7. Verify accuracy metrics (TRUE_POSITIVE, TRUE_NEGATIVE, etc)
echo.
echo 📊 DATABASE STATE:
echo.
echo   - 4 test patients (Red, Green, Yellow x2)
echo   - 9 beds in ER ward
echo   - 4 triage assessments
echo   - 4 disposition records with accuracy tracking
echo.
echo 🎯 KEY ENDPOINTS:
echo.
echo   GET  /api/physician/cases?context=ER
echo   GET  /api/physician/beds?context=ER
echo   POST /api/disposition
echo.
echo 🛑 TO STOP: Close the command windows or press Ctrl+C
echo.
pause
