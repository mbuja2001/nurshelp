import React from "react";

function Footer() {
  return (
    <footer className="ps-footer">
      <div className="ps-footer-brand">
        <div>
          <strong>Nurse Help Desk</strong>
          <span>Clinical intake workspace</span>
        </div>
      </div>
      <div className="ps-footer-meta">
        <span>v1.0</span>
        <span>2026 Living Compute Labs</span>
        <span className="ps-footer-status"><i aria-hidden="true" /> Secure session</span>
      </div>
    </footer>
  );
}

export default Footer;