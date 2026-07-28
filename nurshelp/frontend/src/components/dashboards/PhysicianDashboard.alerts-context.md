# Physician Dashboard Alerts Context

- `Alerts` should not appear in the page header toolbar that is specific to the physician dashboard header.
- The header toolbar in `PhysicianDashboard.jsx` is the top section containing staff bio and buttons for dashboard tools.
- Those toolbar buttons represent constant update actions and should not duplicate the sidebar notification system.
- The main alerts system belongs in `SidebarLayout.jsx` and acts as the global notification hub for:
  - requests and responses
  - orders arrived
  - staff messages
  - scheduled surgeries
  - any mission-critical notifications
- Keep the `Alerts` control only in the sidebar layout and use page header actions for workflow tools, not the alert tray.
