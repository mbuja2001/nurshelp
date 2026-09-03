Priority Queue Perfection Plan

## Current phase

We have finished the security-aware, human-in-the-loop AI triage inference layer.

The next milestone is the refinement of the priority queueing system for the triage nurse workflow. This is the operational layer that makes the AI triage output actionable in real time, with a transparent and defensible queue order that aligns with emergency medicine principles.

We are now focused on making the clinical queue board precise, explainable, and trustworthy for live use in the browser and in live clinical operations.

---

## Where we are

### Completed
- Secure user registration flow
- Hospital seeding integrated into the registering environment
- Staff seeding has started
- Human-in-the-loop AI triage inference is live and working
- EHR context and nurse approval UI are being validated with live endpoints
- Hospital administration model is being prepared for patient and staff management

### In progress
- Seed encounters and patient flow records
- Perfect the admin dashboard as the patient and operational control center
- Define the role-based dashboard model for admin, triage nurse, staff, and clinical teams
- Finalize queue prioritization logic and patient movement through waiting, acknowledged, and treatment stages
- Validate transparency and explainability of queue decisions

---

## Product direction

### Admin-controlled hospital operating model
The customer is the hospital admin. The admin should be able to:
- create and configure the hospital environment
- assign clinicians and staff to shifts
- manage operational responsibilities across departments
- assign dashboards and clinical workspaces to each role
- monitor activity across patient intake, triage, and treatment workflows
- manage staffing, scheduling, and operational visibility

This is the administrative layer that enables a complete clinical operating model, with responsibility ownership distributed across staffing and workflows.

---

## Role-driven clinical experience

### Admin dashboard
- Hospital setup and configuration
- Department and service assignment
- Staff scheduling and shift assignment
- Clinical operation oversight
- Queue performance and throughput monitoring
- Access and permission configuration

### Triage nurse dashboard
- Priority queue board
- Real-time patient acuity visibility
- AI-assisted triage support
- Human approval and override controls
- EHR context review and documentation connection
- Queue movement between awaiting and acknowledged states

### Clinical staff dashboards
- Assigned shift responsibilities
- Clinical analytics and workload board
- Patient tracking and task status
- Manual triage and escalation actions
- EHR records, lab requests, and surgery-related workflow support
- Staff communication and collaboration tools

### Operational modules
- Auto triage and manual triage embedded into the clinical workflow
- Surgery recording support
- Lab request and tracking
- Staff instant messaging/chat system
- Analytics board and operational summaries
- Shift understanding and departmental accountability

---

## High-level application concept

The system is evolving toward a hospital operations platform with a centralized admin control layer and role-based clinical dashboards.

At a high level, this includes:
- secure identity and role management
- hospital and staff seeding
- patient encounter creation and workflow tracking
- AI-assisted triage with human approval
- transparent priority queue logic
- clinical team communication and coordination
- EHR-linked operational records and workflow actions

This is not just a triage tool; it is a clinical operations environment where the queue, staff assignments, and patient movement operate together as a unified system.

---

## Priority queue focus

The current objective is to perfect the priority queue board on the right side of the interface for the triage nurse.

This queue should clearly separate:
- Awaiting
- Acknowledged
- Escalated or urgent
- Reassessed
- Treated / active care

The user should be able to see who is waiting, who has been acknowledged, and why a patient sits above or below another patient in the queue.

---

## Priority queue algorithm direction

We are refining the queue to be:
- clinically defensible
- transparent to nursing staff
- explainable to the triage process
- grounded in emergency medicine priority logic
- consistent with SATS EMS-style triage principles

The queue should not feel opaque or hidden behind an AI black box. The system must show a clear rationale for ranking.

### Core principles
- Immediate danger to life or limb rises highest
- Severity and urgency drive ordering
- Time-sensitive conditions require faster review
- Reassessments must pull patients back into visibility when symptoms change
- The algorithm should support both AI suggestion and explicit human override
- Human approval must remain visible and auditable

### Queue transparency requirements
Each patient in the queue should display:
- acuity score or urgency indicator
- key presenting complaint
- time waiting
- reason for priority order
- AI recommended urgency
- nurse reviewed / approved status
- manual override if applied

This ensures the queue supports trust, accountability, and clinical decision-making.

---

## Queue board UX expectations

The right-side queue board should feel like a clinical operations surface, not a generic dashboard.

### Display should include
- patient name / identifier
- presenting complaint summary
- urgency level
- waiting duration
- triage stage
- nurse acknowledgment state
- quick action controls for review or escalation

### Workflow behavior
- When a patient is added to the queue, they appear in awaiting status
- Once the triage nurse acknowledges them, they move into acknowledged state
- If the patient is clinically urgent, they move higher in the queue automatically
- If clinical status changes, the queue reorder should update immediately
- Human approval of AI triage output should be clearly visible and logged

---

## Clinical safety and human-in-the-loop requirement

We are not handing full autonomous control to the AI system.

The process must remain:
- explainable
- auditable
- visibly supervised by a nurse
- capable of human override
- aligned to operational and clinical safety standards

The AI should support triage decision-making, but the nurse remains responsible for final approval and clinical judgment.

---

## Live validation plan

We are validating the system in real time through the browser while the live endpoints are active.

This means the goal is not just code correctness; it is workflow correctness.

### Validation targets
- live EHR context is loaded correctly
- triage recommendation appears in the correct queue state
- nurse approval UI behaves correctly
- queue order updates according to priority logic
- interface is understandable during active clinical intake
- waiting/acknowledged board actions reflect actual backend state

The current validation stream is focused on seeing the end-to-end system behave correctly as a real operator would use it.

---

## Immediate next work items

1. Complete encounter seeding for realistic patient flow
2. Finalize queue states and their transitions
3. Implement transparent priority logic with visible reasoning
4. Validate the triage nurse board in the browser against live endpoints
5. Confirm EHR context and nurse approval flow is operational
6. Add admin controls for staffing, dashboard assignment, and shift management
7. Continue hardening the safety and permission model

---

## Working principle for today

We are perfecting this today.

The focus is not broad feature expansion; it is precision in the operational triage experience:
- clear queue order
- reliable decisions
- explainable ranking
- trust in the AI-assisted process
- direct usability for the triage nurse

This is the phase where the application moves from working clinical logic into a polished, operationally credible emergency workflow.

---

## Summary

The application is at the point where the core security and AI triage logic is sound, and the next challenge is operational excellence.

The priority queue is now the critical milestone. We are refining the system so the triage nurse sees a transparent, clinically grounded, trustworthy board that reflects urgency in a way that is explainable, safe, and ready for live validation.

This is the next layer of maturity before the full admin dashboard and hospital control environment are finalized.
