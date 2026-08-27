**Scaling Plan: Patient Search**

For a national health repository, the current MongoDB regex search is suitable only for the prototype. It should not become the national search architecture unchanged.

The target architecture should separate:

```text
Clinical source systems
        ↓
Validated integration pipeline
        ↓
National patient identity service
        ↓
Search index
        ↓
Authorized clinical application
        ↓
FHIR resources and audit trail
```

### Phase 1: Harden the prototype

Keep MongoDB and `Encounter` as the clinical source of truth, but improve the search layer.

1. Escape user input before building regex queries.
2. Use exact matching for MRNs and national identifiers.
3. Use prefix matching for autocomplete:

```js
/^lea/i
```

4. Add indexes for:

```js
{ hospitalId: 1, "patient.id_number": 1 }
{ hospitalId: 1, "patient.name": 1 }
{ hospitalId: 1, "patient.surname": 1 }
```

5. Always include `hospitalId` or an equivalent authorization scope.
6. Add pagination and maximum result limits.
7. Never return full clinical records from search suggestions.
8. Log access events without logging sensitive clinical payloads.

Important: regex input must be escaped. Otherwise characters such as `.*` can alter the query and cause expensive searches.

### Phase 2: Create a patient identity layer

A national repository needs a stable identity model separate from individual hospital encounters.

```text
National Patient
  ├── national identifier
  ├── verified demographic identity
  ├── identity confidence
  └── linked hospital identities

Hospital Encounter
  ├── hospital ID
  ├── local MRN
  ├── admission data
  └── clinical reports
```

The search result should resolve:

```text
search term
  -> national patient identity
  -> authorized linked records
  -> hospital encounter history
```

Do not use a hospital encounter ID as the national patient identity. It identifies one encounter, not the person.

### Phase 3: Introduce a dedicated search index

For national-scale autocomplete and fuzzy matching, use MongoDB Atlas Search, OpenSearch, or Elasticsearch.

Recommended fields:

```text
patientId
nationalIdentifier
hospitalIds
localMrns
givenNames
familyNames
dateOfBirth
normalizedName
phoneticName
recordStatus
```

The search index should contain only searchable identity and routing metadata. Detailed clinical documents should remain in protected clinical stores.

Search capabilities can then include:

- Prefix autocomplete
- Typo tolerance
- Exact MRN lookup
- Name matching
- Date-of-birth confirmation
- Phonetic matching
- Relevance ranking
- Search-result highlighting

For example:

```text
"Jon Miler"
```

could suggest:

```text
"John Miller"
```

but the system must require additional identity verification before exposing clinical data.

### Phase 4: Build patient matching carefully

Search and patient matching are different problems.

Search asks:

```text
Which records look like this query?
```

Patient matching asks:

```text
Are these records definitely the same person?
```

A safe matching sequence is:

1. Exact national identifier.
2. Exact verified identifier plus date of birth.
3. Exact local MRN within a hospital.
4. Name plus date of birth.
5. Fuzzy name plus additional demographic evidence.
6. Manual review for ambiguous matches.

Never merge patients automatically based only on similar names.

### Phase 5: Use FHIR for interoperability

The national interface should expose FHIR resources rather than a custom encounter response.

Core resources:

```text
Patient
Encounter
Observation
Condition
MedicationRequest
MedicationAdministration
Procedure
DiagnosticReport
ServiceRequest
DocumentReference
Provenance
AuditEvent
```

The EHR Library can continue presenting a unified timeline, but internally it should map events to their FHIR resource types.

For example:

```text
Encounter.vitals
  -> Observation

Encounter.diagnosis
  -> Condition

Encounter.lab_requests
  -> ServiceRequest / DiagnosticReport

Encounter.surgery_records
  -> Procedure

Encounter.medication_administrations
  -> MedicationAdministration
```

FHIR `Patient` alone is not the complete clinical record.

### Phase 6: Secure national access

A national repository needs more than a bearer token.

Required controls include:

- Entra ID or equivalent national identity provider
- Role-based and attribute-based access control
- Hospital, department, and care-team scoping
- Break-glass emergency access
- Purpose-of-use recording
- Consent and confidentiality flags
- Full `AuditEvent` logging
- Encryption in transit and at rest
- Key rotation
- Rate limiting
- Suspicious-search detection
- Automatic session expiry
- Data minimization

Every clinical record access should answer:

```text
Who accessed it?
Which patient?
When?
From which organization?
For what purpose?
What was viewed?
Was emergency access used?
```

Privilege to work with the data increases the duty to protect it. It should never be used as a reason to bypass authorization, consent, audit, or least-privilege controls.

### Phase 7: Scale the data architecture

At national scale, use separate workloads:

```text
Operational database
  -> writes and transactional clinical updates

Search index
  -> autocomplete and patient discovery

Analytics store
  -> reporting and population health

FHIR gateway
  -> standardized external interoperability

Audit store
  -> immutable access and change records
```

Use event-driven synchronization:

```text
Encounter updated
  -> validation
  -> audit event
  -> search index update
  -> FHIR projection update
```

This avoids making the search index the clinical source of truth.

### Recommended target design

```text
MongoDB Encounter collection
  = hospital clinical source of truth

National Patient Identity Service
  = verified cross-hospital identity

Atlas Search/OpenSearch
  = fast, fuzzy, indexed search

FHIR Gateway
  = national interoperability boundary

EHR Library
  = authorized timeline and clinical viewing interface

AuditEvent/Provenance
  = immutable accountability layer
```

### Most important next steps

1. Fix and escape regex input.
2. Add compound hospital-scoped indexes.
3. Separate identity search from clinical record retrieval.
4. Add exact MRN and national-ID lookup paths.
5. Introduce an explicit patient identity model.
6. Add an indexed search service.
7. Add audit and purpose-of-use enforcement.
8. Define FHIR mappings for all encounter event types.
9. Load-test search with realistic national-scale data.
10. Conduct privacy, security, clinical-safety, and interoperability reviews before connecting to real national systems.

The prototype’s current approach is a good starting point, but national scaling requires treating search as an authorized identity-resolution service, not merely as a MongoDB regex query.
