import { describe, expect, it } from 'vitest';
import { normalizeClinicalAnalysis } from './utils/triageHelpers';

describe('normalizeClinicalAnalysis', () => {
  it('normalizes classic clinicalAnalysis shape into AI-style keys', () => {
    const report = {
      clinicalAnalysis: {
        clinicalSummary: 'Patient has acute abdominal pain',
        pathophysiology: 'Likely bowel obstruction',
        clinicalDiscriminators: ['distended abdomen'],
        differentialDiagnoses: ['appendicitis', 'pancreatitis'],
        immediateActions: ['IV fluids', 'pain control'],
        diagnosticWorkup: ['CT abdomen', 'CBC'],
        dispositionRationale: 'Urgent surgical review required',
        managementActions: ['NPO', 'NG tube'],
        riskFactors: ['smoker'],
        pathognomonicMarkers: ['guarding'],
        lifeThreatsExcluded: ['peritonitis excluded'],
        recommendedSpecialists: ['Surgery']
      },
      selected_priority: 'orange',
      sats_color: 'orange',
      tews_total: 3
    };

    const normalized = normalizeClinicalAnalysis(report);

    expect(normalized.clinical_summary).toBe('Patient has acute abdominal pain');
    expect(normalized.clinical_reasoning_summary).toBe('');
    expect(normalized.recommendations).toEqual([]);
    expect(normalized.pathophysiology).toBe('Likely bowel obstruction');
    expect(normalized.clinical_discriminators).toEqual(['distended abdomen']);
    expect(normalized.differential_diagnoses).toEqual(['appendicitis', 'pancreatitis']);
    expect(normalized.immediate_actions).toEqual(['IV fluids', 'pain control']);
    expect(normalized.diagnostic_workup).toEqual(['CT abdomen', 'CBC']);
    expect(normalized.disposition_rationale).toBe('Urgent surgical review required');
    expect(normalized.management_actions).toEqual(['NPO', 'NG tube']);
    expect(normalized.risk_factors).toEqual(['smoker']);
    expect(normalized.pathognomonic_markers).toEqual(['guarding']);
    expect(normalized.life_threats_excluded).toEqual(['peritonitis excluded']);
    expect(normalized.recommended_specialists).toEqual(['Surgery']);
    expect(normalized.sats_level).toBe('orange');
    expect(normalized.tews_total).toBe(3);
  });

  it('preserves manual recommendations and clinical reasoning summary from nested analysis data', () => {
    const report = {
      clinicalAnalysis: {
        clinical_summary: 'Acute respiratory distress',
        clinical_reasoning_summary: 'Likely pneumonia based on examination and history',
        recommendations: ['Start antibiotics', 'Administer oxygen'],
        differential_diagnoses: ['COPD exacerbation'],
        immediate_actions: ['Oxygen therapy']
      }
    };

    const normalized = normalizeClinicalAnalysis(report);

    expect(normalized.clinical_summary).toBe('Acute respiratory distress');
    expect(normalized.clinical_reasoning_summary).toBe('Likely pneumonia based on examination and history');
    expect(normalized.recommendations).toEqual(['Start antibiotics', 'Administer oxygen']);
    expect(normalized.differential_diagnoses).toEqual(['COPD exacerbation']);
    expect(normalized.immediate_actions).toEqual(['Oxygen therapy']);
  });

  it('normalizes snake_case payloads and list strings correctly', () => {
    const report = {
      aiTriageReport: {
        clinical_summary: 'Severe dehydration',
        pathophysiology: 'Vomiting and diarrhea',
        clinical_discriminators: ['dry mucosa', 'tachycardia'],
        differential_diagnoses: 'Gastroenteritis; food poisoning',
        immediate_actions: 'IV fluids\nmonitor urine output',
        diagnostic_workup: ['electrolytes'],
        disposition_rationale: 'Monitor and rehydrate',
        management_actions: 'oral rehydration; antiemetics',
        risk_factors: ['recent travel'],
        pathognomonic_markers: ['sunken eyes'],
        life_threats_excluded: ['shock'],
        recommended_specialists: 'Internal Medicine',
        urgency_level: 'High',
        priority_code: 'ORANGE'
      }
    };

    const normalized = normalizeClinicalAnalysis(report);

    expect(normalized.clinical_summary).toBe('Severe dehydration');
    expect(normalized.clinical_discriminators).toEqual(['dry mucosa', 'tachycardia']);
    expect(normalized.differential_diagnoses).toEqual(['Gastroenteritis', 'food poisoning']);
    expect(normalized.immediate_actions).toEqual(['IV fluids', 'monitor urine output']);
    expect(normalized.management_actions).toEqual(['oral rehydration', 'antiemetics']);
    expect(normalized.recommended_specialists).toEqual(['Internal Medicine']);
    expect(normalized.priority_code).toBe('ORANGE');
    expect(normalized.urgencyScore).toBeNull();
  });
});
