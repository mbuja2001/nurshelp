# Responsive Design Quick Reference - NurseHelpDesk

## Breakpoints Used

```
Mobile:         < 480px
Tablet:         480px - 768px  
Tablet Large:   768px - 1024px
Desktop:        1024px - 1200px
Desktop Large:  > 1200px
```

## Key Responsive CSS Patterns

### Flexbox Responsive Layout
```css
/* Desktop: horizontal layout */
@media (min-width: 768px) {
  .dashboard-grid {
    flex-direction: row;
    gap: 15px;
  }
}

/* Mobile: vertical stack */
@media (max-width: 768px) {
  .dashboard-grid {
    flex-direction: column;
    gap: 10px;
  }
}
```

### CSS Grid Responsive
```css
/* Desktop: 2 columns */
@media (min-width: 768px) {
  .vitals-grid {
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }
}

/* Mobile: 1 column */
@media (max-width: 768px) {
  .vitals-grid {
    grid-template-columns: 1fr;
    gap: 10px;
  }
}
```

### Responsive Typography
```css
/* Fluid font scaling */
h1 { font-size: clamp(1.5rem, 5vw, 2.5rem); }
p  { font-size: clamp(0.875rem, 2vw, 1.125rem); }

/* Mobile-first */
@media (min-width: 768px) {
  body { font-size: 15px; }
}
@media (min-width: 1024px) {
  body { font-size: 16px; }
}
```

### Responsive Padding & Gaps
```css
/* Use clamp for automatic scaling */
padding: clamp(12px, 4%, 40px);
gap: clamp(8px, 2%, 20px);

/* Or media queries */
@media (max-width: 768px) {
  padding: 12px;
}
@media (min-width: 769px) {
  padding: 40px;
}
```

## Touch-Friendly Design

```css
/* Minimum touch target size */
button {
  min-height: 44px;
  min-width: 44px;
  padding: 12px 16px;
}

/* Prevent iOS zoom on input focus */
input {
  font-size: 16px; /* Must be 16px or larger */
}

/* Responsive spacing for touch */
@media (max-width: 768px) {
  button, input {
    padding: 14px 18px;
  }
}
```

## Mobile-First Approach

```css
/* Start with mobile styles (default) */
.container {
  display: flex;
  flex-direction: column;
  width: 100%;
}

/* Add desktop enhancements */
@media (min-width: 768px) {
  .container {
    flex-direction: row;
    width: auto;
  }
}
```

## Common Responsive Components

### Responsive Modal
```css
.modal-content {
  width: 90%;
  max-width: 500px;
  max-height: 90vh;
  overflow-y: auto;
}

@media (max-width: 480px) {
  .modal-content {
    width: 98%;
    max-width: calc(100vw - 16px);
    max-height: 95vh;
  }
}
```

### Responsive Navigation
```css
.nav-item {
  padding: 8px 8px;
  font-size: 14px;
}

@media (max-width: 768px) {
  .nav-item {
    padding: 6px 6px;
    font-size: 12px;
  }
}

@media (max-width: 480px) {
  .nav-item {
    padding: 5px 4px;
    font-size: 11px;
  }
}
```

### Responsive Table (Horizontal Scroll on Mobile)
```css
table {
  width: 100%;
  overflow-x: auto;
}

@media (max-width: 768px) {
  .table-wrapper {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch; /* Smooth scroll iOS */
  }
}
```

## Viewport Meta Tag (HTML Head)

```html
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
```

## CSS Media Query Best Practices

### ✅ DO
```css
/* Mobile-first */
.element { /* mobile styles */ }
@media (min-width: 768px) { /* tablet+ */ }
@media (min-width: 1024px) { /* desktop+ */ }

/* Use logical values */
width: 100%;
max-width: 1200px;
margin: 0 auto;

/* Flexible units */
padding: clamp(12px, 4%, 40px);
font-size: clamp(14px, 2vw, 18px);
```

### ❌ DON'T
```css
/* Desktop-first (outdated) */
.element { /* desktop styles */ }
@media (max-width: 768px) { /* mobile */ }

/* Fixed widths */
width: 1920px;

/* Magic numbers */
margin: 27.3px;
padding: 13.7%;
```

## Testing at Different Sizes

### Browser DevTools
1. Press F12 to open DevTools
2. Click device toolbar (Ctrl+Shift+M on Windows, Cmd+Shift+M on Mac)
3. Select presets: iPhone, iPad, Desktop
4. Or enter custom dimensions

### Common Device Sizes
- **Mobile:** 375x667 (iPhone SE), 414x896 (iPhone 12)
- **Tablet:** 768x1024 (iPad), 600x960 (Android tablet)
- **Desktop:** 1920x1080, 1366x768

## Performance Tips

1. **Use CSS Grid & Flexbox** - more efficient than floats
2. **Minimize media queries** - use CSS properties that scale naturally
3. **Avoid layout thrashing** - don't trigger multiple reflows
4. **Use `clamp()` and `min()`/`max()`** - reduces media query count
5. **Mobile-first cascade** - reduces CSS overhead

## Files to Check

| Component | File | Responsive | Status |
|-----------|------|------------|--------|
| Sidebar | SidebarLayout.jsx | ✅ | Complete |
| Patient Summary | PS.css | ✅ | Complete |
| Vitals Form | Vitals.jsx | ✅ | Complete |
| Dashboard | Dashboard.index.css | ✅ | Complete |
| Global | index.css | ✅ | Complete |

---

## Troubleshooting

**Issue:** Elements stack weirdly on mobile  
**Fix:** Check `flex-direction` and `flex-basis` in media queries

**Issue:** Touch targets too small  
**Fix:** Ensure `min-height: 44px` and `min-width: 44px` on interactive elements

**Issue:** iOS zooms on input focus  
**Fix:** Set `font-size: 16px` or larger on inputs

**Issue:** Horizontal scroll on mobile  
**Fix:** Add `box-sizing: border-box` and check max-width on containers

---

**Last Updated:** June 5, 2026  
**Version:** 1.0  
**Maintained By:** Development Team
