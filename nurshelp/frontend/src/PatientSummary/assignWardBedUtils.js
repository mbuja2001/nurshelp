export const getPatientDisplayName = (encounter = {}) => {
  return (
    encounter?.patientName ||
    encounter?.patient?.name ||
    encounter?.firstName ||
    encounter?.patient?.firstName ||
    "Patient"
  );
};

export const getEncounterIdentifier = (encounter = {}) => {
  return encounter?._id || encounter?.id || "";
};

export const getBedLabel = (bed = {}) => {
  return bed?.bedId || bed?.bed_number || bed?.roomNumber || "Bed";
};
