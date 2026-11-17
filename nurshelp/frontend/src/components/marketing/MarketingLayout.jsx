import React, { useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { FiArrowUpRight, FiMenu, FiX } from 'react-icons/fi';
import nhpLogo from '../../Dashboard/NHP_Icon.png';
import './marketing.css';

const links = [
  { label: 'Platform', to: '/what-we-do' },
  { label: 'Our approach', to: '/ethics' },
  { label: 'Clinical validation', to: '/validation' },
  { label: 'Trust & regulation', to: '/security' },
  { label: 'Journal', to: '/blog' }
];

export default function MarketingLayout({ children }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const navigate = useNavigate();

  const closeMenu = () => setMenuOpen(false);

  return (
    <div className="marketing-site">
      <header className="marketing-header">
        <div className="marketing-header-inner">
          <button className="marketing-brand" type="button" onClick={() => { closeMenu(); navigate('/'); }} aria-label="NurseHelpDesk home">
            <img src={nhpLogo} alt="" />
            <span>NurseHelpDesk<small>CARE, CONNECTED.</small></span>
          </button>
          <button
            type="button"
            className="marketing-menu-toggle"
            aria-label={menuOpen ? 'Close navigation menu' : 'Open navigation menu'}
            aria-expanded={menuOpen}
            aria-controls="marketing-navigation"
            onClick={() => setMenuOpen((open) => !open)}
          >
            {menuOpen ? <FiX /> : <FiMenu />}
          </button>
          <nav id="marketing-navigation" className={`marketing-navigation${menuOpen ? ' is-open' : ''}`} aria-label="Main navigation">
            <div className="marketing-nav-links">
              {links.map((link) => (
                <NavLink key={link.to} to={link.to} onClick={closeMenu} className={({ isActive }) => isActive ? 'is-active' : ''}>
                  {link.label}
                </NavLink>
              ))}
            </div>
            <div className="marketing-nav-actions">
              <button type="button" className="marketing-login-link" onClick={() => { closeMenu(); navigate('/dashboard'); }}>Log in</button>
              <button type="button" className="marketing-nav-cta" onClick={() => { closeMenu(); navigate('/dashboard?register=account'); }}>
                Get started <FiArrowUpRight aria-hidden="true" />
              </button>
            </div>
          </nav>
        </div>
      </header>
      {children}
      <footer className="marketing-footer">
        <div className="marketing-footer-main">
          <div className="marketing-footer-brand">
            <button className="marketing-brand" type="button" onClick={() => navigate('/')} aria-label="NurseHelpDesk home">
              <img src={nhpLogo} alt="" />
              <span>NurseHelpDesk<small>CARE, CONNECTED.</small></span>
            </button>
            <p>Building more connected, resilient workflows and responsible AI tools for South Africa's rural public-care teams.</p>
          </div>
          <div className="marketing-footer-links">
            <div><span>Explore</span><NavLink to="/what-we-do">Platform</NavLink><NavLink to="/team">Our team</NavLink><NavLink to="/blog">Journal</NavLink></div>
            <div><span>Principles</span><NavLink to="/ethics">Our approach</NavLink><NavLink to="/validation">Clinical validation</NavLink><NavLink to="/security">Trust & regulation</NavLink></div>
            <div><span>Access</span><NavLink to="/dashboard">Log in</NavLink><NavLink to="/dashboard?register=account">Create account</NavLink></div>
          </div>
        </div>
        <div className="marketing-footer-bottom"><span>Copyright {new Date().getFullYear()} NurseHelpDesk</span><span>Designed around people, not just processes.</span></div>
      </footer>
    </div>
  );
}
