const scoreBand = (value, bands) => {
  if (!Number.isFinite(value)) return 0;
  return bands.find((band) => band.test(value))?.points ?? 0;
};

export const getInitialTewsProfile = (patient = {}) => {
  const age = Number(patient.ageYears ?? patient.age ?? (patient.ageMonths != null ? patient.ageMonths / 12 : NaN));
  const height = Number(patient.height_cm);
  if (Number.isFinite(age) && age >= 13) return 'adult';
  if (!Number.isFinite(age) && (!Number.isFinite(height) || height > 150)) return 'adult';
  if (Number.isFinite(height) && height <= 95) return 'child-younger';
  return 'child-older';
};

export const calculateTews = (profile, vitals = {}) => {
  const temperature = Number(vitals.temperature);
  const heartRate = Number(vitals.heartRate);
  const respiratoryRate = Number(vitals.respiratoryRate);
  const systolicBP = Number(vitals.systolicBP);
  const avpuScore = ({ alert: 0, a: 0, voice: 1, v: 1, pain: 2, p: 2, unresponsive: 3, u: 3 })[String(vitals.avpu ?? '').trim().toLowerCase()] ?? 0;
  const mobilityScore = ({ walks: 0, walking: 0, 'with help': 1, help: 1, immobile: 2, 'not walking': 2, 'stretcher / immobile': 2 })[String(vitals.mobility ?? '').trim().toLowerCase()] ?? 0;
  const components = { temperature: 0, heartRate: 0, respiratoryRate: 0, systolicBP: 0, avpu: avpuScore, mobility: mobilityScore, trauma: vitals.traumaPresent ? 1 : 0 };

  if (profile === 'adult') {
    components.temperature = scoreBand(temperature, [{ test: (v) => v < 35, points: 3 }, { test: (v) => v >= 38.5, points: 2 }, { test: (v) => v >= 35 && v <= 38.4, points: 0 }]);
    components.heartRate = scoreBand(heartRate, [{ test: (v) => v < 41, points: 3 }, { test: (v) => v >= 41 && v <= 50, points: 2 }, { test: (v) => v >= 51 && v <= 100, points: 1 }, { test: (v) => v >= 101 && v <= 110, points: 0 }, { test: (v) => v >= 111 && v <= 129, points: 1 }, { test: (v) => v >= 130, points: 3 }]);
    components.respiratoryRate = scoreBand(respiratoryRate, [{ test: (v) => v < 9, points: 3 }, { test: (v) => v >= 9 && v <= 20, points: 0 }, { test: (v) => v >= 21 && v <= 29, points: 1 }, { test: (v) => v >= 30, points: 3 }]);
    components.systolicBP = scoreBand(systolicBP, [{ test: (v) => v < 71, points: 3 }, { test: (v) => v >= 71 && v <= 80, points: 1 }, { test: (v) => v >= 81 && v <= 199, points: 0 }, { test: (v) => v >= 200, points: 2 }]);
  } else if (profile === 'child-younger') {
    components.heartRate = scoreBand(heartRate, [{ test: (v) => v < 80 || v > 200, points: 3 }, { test: (v) => v < 90 || v > 180, points: 2 }, { test: (v) => v < 100 || v > 160, points: 1 }]);
    components.respiratoryRate = scoreBand(respiratoryRate, [{ test: (v) => v < 30 || v > 60, points: 3 }, { test: (v) => v > 50, points: 2 }, { test: (v) => v > 40, points: 1 }]);
  } else {
    components.heartRate = scoreBand(heartRate, [{ test: (v) => v < 60 || v > 160, points: 3 }, { test: (v) => v < 70 || v > 140, points: 2 }, { test: (v) => v < 80 || v > 120, points: 1 }]);
    components.respiratoryRate = scoreBand(respiratoryRate, [{ test: (v) => v < 20 || v > 50, points: 3 }, { test: (v) => v > 40, points: 2 }, { test: (v) => v > 30, points: 1 }]);
    components.systolicBP = scoreBand(systolicBP, [{ test: (v) => v < 70 || v >= 180, points: 3 }, { test: (v) => v < 80 || v >= 160, points: 2 }, { test: (v) => v < 90 || v >= 130, points: 1 }]);
  }

  const total = Object.entries(components).filter(([key]) => key !== 'trauma').reduce((sum, [, value]) => sum + value, 0);
  const complete = [temperature, heartRate, respiratoryRate, systolicBP].every(Number.isFinite);
  const priority = total >= 7 ? 'red' : total >= 5 ? 'orange' : total >= 3 ? 'yellow' : 'green';
  return { total, priority, components, traumaModifier: components.trauma, complete };
};
