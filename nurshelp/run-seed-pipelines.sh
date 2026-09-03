#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="${SCRIPT_DIR}/backend"

usage() {
  cat <<'EOF'
Usage: ./run-seed-pipelines.sh [--skip-hospital] [--skip-staff]

Runs the backend hospital/staff seed pipelines.

Options:
  --skip-hospital   Skip the multi-hospital seed pipeline
  --skip-staff      Skip the staff seed pipeline
  -h, --help        Show this help message
EOF
}

SKIP_HOSPITAL=false
SKIP_STAFF=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-hospital)
      SKIP_HOSPITAL=true
      shift
      ;;
    --skip-staff)
      SKIP_STAFF=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "${SKIP_HOSPITAL}" == true && "${SKIP_STAFF}" == true ]]; then
  echo "Nothing to run: both seed pipelines are skipped." >&2
  exit 1
fi

if [[ ! -d "${BACKEND_DIR}" ]]; then
  echo "Backend directory not found: ${BACKEND_DIR}" >&2
  exit 1
fi

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js is required to run the seed pipelines." >&2
  exit 1
fi

cd "${BACKEND_DIR}"

if [[ "${SKIP_HOSPITAL}" != true ]]; then
  echo "[1/2] Running multi-hospital seed pipeline..."
  node src/seeds/seedMultiHospitalSetup.js
  echo "✅ Multi-hospital seed completed."
fi

if [[ "${SKIP_STAFF}" != true ]]; then
  echo "[2/2] Running staff seed pipeline..."
  node src/seeds/seedHospitalStaff.js
  echo "✅ Staff seed completed."
fi

echo "🎯 Seed pipeline run complete."
