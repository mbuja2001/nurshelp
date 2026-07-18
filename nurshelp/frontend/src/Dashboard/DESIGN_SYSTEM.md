# Unified Dashboard Design System

## Overview

All dashboards (Physician, Nurse, Settings) now use a **consistent, theme-based design system** that ensures:

- ✅ Unified color palette and typography
- ✅ Consistent button, table, card, and badge styles
- ✅ Responsive design across all device sizes
- ✅ Accessible components with proper contrast and focus states
- ✅ Smooth animations and transitions
- ✅ Semantic role-specific color schemes

## File Structure

```
frontend/src/
├── Dashboard/
│   ├── DashboardTheme.css          ← Main theme file (imported by all)
│   ├── Dashboard.index.css          ← Global dashboard styles
│   ├── NurseDashboard.jsx           ← Updated to use theme
│   └── Pages/
│       ├── Settings.jsx             ← Updated to use theme
│       └── Settings.css             ← Settings-specific styles
└── components/dashboards/
    ├── PhysicianDashboard.jsx       ← Updated to use theme
    └── PhysicianDashboard.css       ← Physician-specific styles
```

## Color Variables (DashboardTheme.css)

### Primary Colors
```css
--color-primary: #0369a1;           /* Blue accent */
--color-primary-dark: #0c4a6e;
--color-primary-light: #e0f2fe;
```

### Role-Specific Colors
```css
--color-physician: #dc3545;         /* Red for Physician */
--color-nurse: #3b82f6;             /* Blue for Nurse */
--color-settings: #8b5cf6;          /* Purple for Settings */
```

### Status & Semantic Colors
```css
--color-danger: #ef4444;            /* Red for errors/alerts */
--color-success: #10b981;           /* Green for success */
--color-warning: #f59e0b;           /* Amber for warnings */
--color-info: #3b82f6;              /* Blue for info */
```

### Neutrals
```css
--color-bg-primary: #ffffff;        /* Main background */
--color-bg-secondary: #f8fafc;      /* Secondary background */
--color-bg-tertiary: #f1f5f9;       /* Tertiary background */
--color-text-primary: #0f172a;      /* Main text (dark navy) */
--color-text-secondary: #475569;    /* Secondary text */
--color-text-muted: #94a3b8;        /* Muted/disabled text */
```

## Component Styles

### Buttons

All buttons inherit from the theme and support multiple variants:

```jsx
// Primary Button
<button className="btn btn-primary">Primary Action</button>

// Secondary Button
<button className="btn btn-secondary">Secondary Action</button>

// Danger Button
<button className="btn btn-danger">Delete</button>

// Success Button
<button className="btn btn-success">Confirm</button>

// Ghost Button
<button className="btn btn-ghost">Cancel</button>

// Size Variants
<button className="btn btn-small">Small</button>
<button className="btn btn-large">Large</button>

// Disabled State
<button className="btn btn-primary" disabled>Disabled</button>
```

### Tables

Tables use consistent styling with hover effects and alert states:

```jsx
<div className="table-wrapper">
  <table className="table">
    <thead>
      <tr>
        <th>Header 1</th>
        <th>Header 2</th>
      </tr>
    </thead>
    <tbody>
      <tr className="table-alert-row">
        <td>Alert content</td>
      </tr>
      <tr className="table-success-row">
        <td>Success content</td>
      </tr>
    </tbody>
  </table>
</div>
```

### Cards

Cards are used for grouped content with consistent padding and shadows:

```jsx
<div className="card">
  <h3>Card Title</h3>
  <div className="card-info-item">
    <span className="label">Label:</span>
    <span className="value">Value</span>
  </div>
</div>
```

### Tabs

Tab navigation with active state indication:

```jsx
<div className="tab-navigation">
  <button className="tab active">Active Tab</button>
  <button className="tab">Tab 2</button>
</div>
<div className="tab-content">
  {/* Tab content */}
</div>
```

### Badges

Status badges for quick visual identification:

```jsx
<span className="badge badge-active">Active</span>
<span className="badge badge-danger">Critical</span>
<span className="badge badge-warning">Pending</span>
<span className="badge badge-success">Completed</span>
```

### Alerts

Alert messages with semantic coloring:

```jsx
<div className="alert alert-success">
  <div className="alert-icon">✓</div>
  <p>Operation successful!</p>
</div>

<div className="alert alert-danger">
  <div className="alert-icon">!</div>
  <p>An error occurred</p>
</div>
```

## Typography

Consistent font sizing and weights:

```css
--font-xs: 12px;        /* Small labels */
--font-sm: 14px;        /* Form text */
--font-base: 16px;      /* Body text */
--font-lg: 18px;        /* Subheadings */
--font-xl: 20px;        /* Headings */
--font-2xl: 24px;       /* Large headings */
--font-3xl: 28px;       /* Page titles */
```

## Spacing Scale

All components use a consistent spacing scale:

```css
--space-xs: 4px;        /* Micro spacing */
--space-sm: 8px;        /* Small spacing */
--space-md: 12px;       /* Medium spacing */
--space-lg: 16px;       /* Large spacing */
--space-xl: 24px;       /* Extra large */
--space-2xl: 32px;      /* 2x extra large */
```

## Border Radius

```css
--radius-sm: 4px;       /* Slightly rounded */
--radius-md: 6px;       /* Medium rounded */
--radius-lg: 8px;       /* Regular rounded */
--radius-xl: 12px;      /* Highly rounded */
```

## Shadows

Consistent shadow depths:

```css
--shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
--shadow-md: 0 4px 6px rgba(0, 0, 0, 0.07);
--shadow-lg: 0 6px 18px rgba(0, 0, 0, 0.08);
--shadow-xl: 0 20px 25px rgba(0, 0, 0, 0.1);
```

## Transitions

Smooth animations with consistent timing:

```css
--transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
--transition-base: 200ms cubic-bezier(0.4, 0, 0.2, 1);
--transition-slow: 300ms cubic-bezier(0.4, 0, 0.2, 1);
```

## Responsive Breakpoints

The theme includes responsive media queries:

```css
/* Tablet and below */
@media (max-width: 1024px) { }

/* Mobile and below */
@media (max-width: 768px) { }

/* Small mobile */
@media (max-width: 640px) { }
```

## Dashboard-Specific Customization

Each dashboard can override role-specific colors by setting `:root` variables:

### PhysicianDashboard.css
```css
:root {
  --color-role-primary: #dc3545;      /* Physician red */
  --color-role-light: #ffe8e8;
}
```

### Settings.css
```css
:root {
  --color-role-primary: #8b5cf6;      /* Settings purple */
  --color-role-light: #f5f3ff;
}
```

### NurseDashboard (implicit)
```css
/* Uses default primary color (#0369a1) */
```

## Implementation Guidelines

### For New Components

1. Import the theme CSS first:
```jsx
import './DashboardTheme.css';
```

2. Use CSS variables instead of hardcoded colors:
```css
/* ✅ Good */
color: var(--color-text-primary);
background: var(--color-bg-secondary);
padding: var(--space-lg);

/* ❌ Avoid */
color: #0f172a;
background: #f8fafc;
padding: 16px;
```

3. Use semantic class names:
```jsx
/* ✅ Good */
<button className="btn btn-primary">Action</button>
<table className="table">...</table>
<div className="card">...</div>

/* ❌ Avoid */
<button style={{ background: '#0369a1' }}>Action</button>
<table style={{ ... }}>...</table>
<div style={{ ... }}>...</div>
```

### For Consistent Tables

```jsx
<div className="table-wrapper">
  <table className="table">
    <thead>
      <tr>
        <th>Column 1</th>
        <th>Column 2</th>
        <th>Column 3</th>
      </tr>
    </thead>
    <tbody>
      <tr className={isAlert ? 'table-alert-row' : ''}>
        <td>Data 1</td>
        <td>Data 2</td>
        <td>
          <button className="btn btn-small btn-primary">Action</button>
        </td>
      </tr>
    </tbody>
  </table>
</div>
```

### For Consistent Cards

```jsx
<div className="card">
  <h3>📊 Card Title</h3>
  <div className="card-stat">
    <span className="label">Label:</span>
    <span className="value">42</span>
  </div>
  <div className="divider"></div>
  <div className="card-info-item">
    <span className="label">Info:</span>
    <span className="value">Value</span>
  </div>
</div>
```

## Quality Checklist

When updating dashboard components, ensure:

- ✅ Uses CSS variables (not hardcoded colors)
- ✅ Respects the spacing scale
- ✅ Uses semantic class names
- ✅ Supports hover/active states
- ✅ Mobile responsive (max-width: 768px)
- ✅ Accessible focus states
- ✅ Consistent button styles
- ✅ Tables use proper structure
- ✅ Cards have proper hierarchy
- ✅ Status badges are semantic

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari 12+
- Mobile browsers (iOS Safari, Chrome Mobile)

## Future Enhancements

- [ ] Dark mode support
- [ ] Custom theme builder
- [ ] RTL language support
- [ ] Animation performance optimization
- [ ] Accessibility audit with WCAG 2.1 AA compliance
