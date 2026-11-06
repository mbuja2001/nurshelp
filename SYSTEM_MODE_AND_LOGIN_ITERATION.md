# System Mode, Mock Simulation, and Login Flow

## Objective

Finalize the admin control iteration by separating:

- live HL7/FHIR integration mode
- mock sandbox simulation mode
- seeded-test user login flow for product demos and operator testing

This establishes a clean operating model before the next implementation pass.

## Core requirement

The system must clearly and conditionally indicate whether the current hospital workspace is:

- running with live HL7/FHIR integrated data
- or using synthetic/mock seeded data for simulation and testing

The user experience should not be ambiguous. The header status badge must read explicitly and visibly.

## Required UI state model

The app should resolve a single active system mode:

- `live`
- `mock`

This mode is derived from the actual current data source, not just static text.

### Live mode conditions

Set to `live` when:

- HL7/FHIR adapter is enabled and healthy
- hospital workspace is tied to real configured data
- no mock/synthetic simulation context is active

Header label:

- `Live Active HL7/FHIR Integration`

### Mock mode conditions

Set to `mock` when:

- synthetic staff, patients, beds, shifts, or encounters are being used
- the user is in a demo/test sandbox session
- the environment intentionally simulates a hospital day instead of using live operational records

Header label:

- `Mock Simulation`

## Operational behavior

### Live mode

- reads hospital data from configured live sources
- accepts operational records from connected workflows
- uses current shift and workload policy state
- allows admin to launch the actual hospital day operations

### Mock mode

- seeds a routine hospital environment with synthetic staff, beds, encounters, and shifts
- allows product demos or user acceptance testing without touching real data
- still uses the same UI workflow and role model, but clearly labels the data source as synthetic
- supports a user to simulate a hospital day from the admin dashboard

## Admin controls to keep in scope

During this iteration, the admin controls should stay inside the admin dashboard and support the following:

- create new hospital
- start hospital day
- create shifts
- set workload policy
- review readiness
- simulate or live-integrate the operating mode

The specialist dashboard shortcuts remain outside the admin overview; they are not the primary control surface for this iteration.

## Login and staff credential story

After the mock/live mode split, the next step is to reverse-engineer the login flow as a testing and onboarding path.

### Goal

Allow a user to operate the system using the seeded credentials already created for staff members.

### Pattern

- seed a mock hospital with staff members and credentials
- list the available demo accounts in a test interface or login example panel
- user logs in as one of the seeded staff members
- app routes them by role and department
- user completes the day flow in the dashboard as that staff actor

### Example login use case

- admin / manager account
- triage nurse
- clinical nurse
- physician
- pharmacist
- radiologist
- pathologist

The user should be able to test the operational flow from the perspective of a real hospital actor rather than only as an admin.

## Implementation principle

The system should be built as two honest modes:

- one for real hospital operations with live EHR integration
- one for a client-side mock sandbox for testing user behavior and workflows

This keeps validation honest and prevents confusion between production behavior and synthetic demo state.

## Acceptance criteria

- The header clearly displays whether the workspace is live or mock
- The status is derived from current conditions and not static text
- The mock flow uses synthetic hospital data and is explicitly labeled as such
- The live flow uses real integration health and real source data
- The admin dashboard remains the control surface for hospital operations
- Staff login credentials created in the seeded hospital can be used for realistic user journey testing

## Next implementation pass

The next step is to convert this plan into executable UI and state changes:

1. add a single `systemMode` state
2. set it via live adapter health + mock/synthetic data detection
3. render the big status badge in the header
4. add a mock mode toggle / sandbox entry flow
5. provide seeded staff login examples for the user journey

This closes the admin controls iteration and opens the final testing and login iteration.
