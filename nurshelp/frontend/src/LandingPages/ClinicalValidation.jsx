import React from "react";
import { useNavigate } from "react-router-dom";
import nhpLogo from "../Dashboard/NHP_Icon.png";

const ClinicalValidation = () => {
  const navigate = useNavigate();

  const handleLinkHover = (e, color) => {
    e.target.style.color = color;
  };

  const navLinks = [
    { label: "Team", path: "/team" },
    { label: "What We Do", path: "/what-we-do" },
    { label: "Ethics & Responsibility", path: "/ethics" },
    { label: "Clinical Validation", path: "/validation" },
    { label: "Security & Compliance", path: "/security" }
  ];

  return (
    <div style={{ fontFamily: "sans-serif", backgroundColor: "#fff", minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      {/* Navigation Bar */}
      <nav style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "20px 8%",
        backgroundColor: "#0f172a",
        color: "#fff",
        borderBottom: "1px solid #1e293b"
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer" }} onClick={() => navigate("/")}>
          <img src={nhpLogo} alt="NurseHelpDesk" style={{ height: '6.3rem', width: 'auto' }} />
          <h1 style={{ margin: 0, fontSize: 24, fontWeight: 700 }}>NurseHelpDesk</h1>
        </div>
        <div style={{ display: "flex", gap: 24, alignItems: "center" }}>
          {navLinks.map(link => (
            <a
              key={link.path}
              onClick={() => navigate(link.path)}
              style={{
                color: link.path === "/validation" ? "#10b981" : "#cbd5e1",
                cursor: "pointer",
                textDecoration: "none",
                fontWeight: link.path === "/validation" ? 600 : 500,
                transition: "all 200ms"
              }}
              onMouseEnter={(e) => {
                e.target.style.color = "#10b981";
              }}
              onMouseLeave={(e) => {
                e.target.style.color = link.path === "/validation" ? "#10b981" : "#cbd5e1";
              }}
            >
              {link.label}
            </a>
          ))}
        </div>
        <div style={{ display: "flex", gap: 12 }}>
          <button
            onClick={() => navigate("/dashboard")}
            style={{
              padding: "10px 20px",
              borderRadius: 8,
              border: "1px solid #fff",
              backgroundColor: "transparent",
              color: "#fff",
              cursor: "pointer",
              fontWeight: 600,
              transition: "all 200ms"
            }}
            onMouseEnter={(e) => {
              e.target.style.backgroundColor = "#fff";
              e.target.style.color = "#0f172a";
            }}
            onMouseLeave={(e) => {
              e.target.style.backgroundColor = "transparent";
              e.target.style.color = "#fff";
            }}
          >
            Log In
          </button>
        </div>
      </nav>

      {/* Clinical Validation Content */}
      <section style={{
        backgroundColor: "#fff",
        padding: "80px 8%",
        minHeight: "80vh",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center"
      }}>
        <h1 style={{ fontSize: 48, fontWeight: 800, marginBottom: 20, color: "#0f172a" }}>
          Clinical Validation
        </h1>
        <p style={{ fontSize: 18, color: "#64748b", maxWidth: 800, lineHeight: 1.8, marginBottom: 40 }}>
          Every feature in NurseHelpDesk is validated through rigorous clinical evidence. Our algorithms are trained on peer-reviewed literature, clinical trial data, and real-world ED operations.
        </p>
        
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 40, marginTop: 40 }}>
          {[
            { 
              title: "Evidence-Based Protocols", 
              description: "Triage algorithms follow ESI and CTAS standards. Decision support backed by clinical guidelines and literature."
            },
            { 
              title: "Pilot Studies", 
              description: "Validated through pilot deployments in 5+ hospitals. Results published in peer-reviewed journals."
            },
            { 
              title: "Continuous Monitoring", 
              description: "Real-time performance metrics tracked for accuracy, sensitivity, and specificity in live ED environments."
            },
            { 
              title: "Expert Review", 
              description: "Clinical advisory board reviews every update. Annual independent audits by external medical institutions."
            }
          ].map(feature => (
            <div key={feature.title} style={{
              padding: 24,
              backgroundColor: "#f9fafb",
              borderRadius: 12,
              border: "1px solid #e5e7eb"
            }}>
              <h3 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: "#0f172a", marginBottom: 8 }}>{feature.title}</h3>
              <p style={{ margin: 0, fontSize: 14, color: "#64748b", lineHeight: 1.6 }}>{feature.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Footer Section */}
      <section style={{
        backgroundColor: "#f3f4f6",
        borderTop: "2px solid #10b981",
        width: "100%",
        display: "block",
        boxSizing: "border-box",
        marginTop: "auto"
      }}>
        <footer style={{
          color: "#000",
          padding: "80px 8% 60px",
          width: "100%",
          display: "block",
          boxSizing: "border-box",
          margin: 0,
          height: "auto",
          minHeight: "auto",
          flexShrink: 0,
          position: "relative"
        }}>
        {/* Top Section with Logo and Description */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 3fr", gap: 50, marginBottom: 80, alignItems: "start" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
              <img src={nhpLogo} alt="NurseHelpDesk" style={{ height: '4rem', width: 'auto' }} />
              <div>
                <h3 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#fff" }}>NurseHelpDesk</h3>
                <p style={{ margin: "4px 0 0 0", fontSize: 11, color: "#94a3b8", fontWeight: 500 }}>ERA 2026</p>
              </div>
            </div>
            <p style={{ margin: "20px 0 0 0", fontSize: 14, color: "#111", lineHeight: 1.6, maxWidth: 250 }}>
              Advancing emergency department operations through intelligent patient triage and clinical decision support.
            </p>
          </div>
          
          {/* Main Footer Links Grid */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 40 }}>
            <div>
              <h4 style={{ color: "#000", margin: "0 0 20px 0", fontSize: 12, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.5px" }}>Product</h4>
              <ul style={{ margin: 0, padding: 0, listStyle: "none", display: "flex", flexDirection: "column", gap: 12 }}>
                <li><a href="#" style={{ color: "#111", textDecoration: "none", fontSize: 14, fontWeight: 500, transition: "color 200ms" }} onMouseEnter={(e) => handleLinkHover(e, "#10b981")} onMouseLeave={(e) => handleLinkHover(e, "#111")}>Features</a></li>
                <li><a href="#" style={{ color: "#111", textDecoration: "none", fontSize: 14, fontWeight: 500, transition: "color 200ms" }} onMouseEnter={(e) => handleLinkHover(e, "#10b981")} onMouseLeave={(e) => handleLinkHover(e, "#111")}>Pricing</a></li>
                <li><a href="#" style={{ color: "#111", textDecoration: "none", fontSize: 14, fontWeight: 500, transition: "color 200ms" }} onMouseEnter={(e) => handleLinkHover(e, "#10b981")} onMouseLeave={(e) => handleLinkHover(e, "#111")}>Demo</a></li>
              </ul>
            </div>
            <div>
              <h4 style={{ color: "#000", margin: "0 0 20px 0", fontSize: 12, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.5px" }}>Company</h4>
              <ul style={{ margin: 0, padding: 0, listStyle: "none", display: "flex", flexDirection: "column", gap: 12 }}>
                <li><a href="#" onClick={() => navigate("/team")} style={{ color: "#111", textDecoration: "none", fontSize: 14, fontWeight: 500, transition: "color 200ms" }} onMouseEnter={(e) => handleLinkHover(e, "#10b981")} onMouseLeave={(e) => handleLinkHover(e, "#111")}>Team</a></li>
                <li><a href="#" style={{ color: "#111", textDecoration: "none", fontSize: 14, fontWeight: 500, transition: "color 200ms" }} onMouseEnter={(e) => handleLinkHover(e, "#10b981")} onMouseLeave={(e) => handleLinkHover(e, "#111")}>Blog</a></li>
                <li><a href="#" style={{ color: "#111", textDecoration: "none", fontSize: 14, fontWeight: 500, transition: "color 200ms" }} onMouseEnter={(e) => handleLinkHover(e, "#10b981")} onMouseLeave={(e) => handleLinkHover(e, "#111")}>Careers</a></li>
              </ul>
            </div>
            <div>
              <h4 style={{ color: "#000", margin: "0 0 20px 0", fontSize: 12, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.5px" }}>Resources</h4>
              <ul style={{ margin: 0, padding: 0, listStyle: "none", display: "flex", flexDirection: "column", gap: 12 }}>
                <li><a href="#" onClick={() => navigate("/validation")} style={{ color: "#111", textDecoration: "none", fontSize: 14, fontWeight: 500, transition: "color 200ms" }} onMouseEnter={(e) => handleLinkHover(e, "#10b981")} onMouseLeave={(e) => handleLinkHover(e, "#111")}>Clinical Validation</a></li>
                <li><a href="#" onClick={() => navigate("/security")} style={{ color: "#111", textDecoration: "none", fontSize: 14, fontWeight: 500, transition: "color 200ms" }} onMouseEnter={(e) => handleLinkHover(e, "#10b981")} onMouseLeave={(e) => handleLinkHover(e, "#111")}>Security</a></li>
                <li><a href="#" style={{ color: "#111", textDecoration: "none", fontSize: 14, fontWeight: 500, transition: "color 200ms" }} onMouseEnter={(e) => handleLinkHover(e, "#10b981")} onMouseLeave={(e) => handleLinkHover(e, "#111")}>Support</a></li>
              </ul>
            </div>
            <div>
              <h4 style={{ color: "#000", margin: "0 0 20px 0", fontSize: 12, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.5px" }}>Legal</h4>
              <ul style={{ margin: 0, padding: 0, listStyle: "none", display: "flex", flexDirection: "column", gap: 12 }}>
                <li><a href="#" style={{ color: "#111", textDecoration: "none", fontSize: 14, fontWeight: 500, transition: "color 200ms" }} onMouseEnter={(e) => handleLinkHover(e, "#10b981")} onMouseLeave={(e) => handleLinkHover(e, "#111")}>Privacy</a></li>
                <li><a href="#" onClick={() => navigate("/ethics")} style={{ color: "#111", textDecoration: "none", fontSize: 14, fontWeight: 500, transition: "color 200ms" }} onMouseEnter={(e) => handleLinkHover(e, "#10b981")} onMouseLeave={(e) => handleLinkHover(e, "#111")}>Ethics & Compliance</a></li>
                <li><a href="#" style={{ color: "#111", textDecoration: "none", fontSize: 14, fontWeight: 500, transition: "color 200ms" }} onMouseEnter={(e) => handleLinkHover(e, "#10b981")} onMouseLeave={(e) => handleLinkHover(e, "#111")}>Contact</a></li>
              </ul>
            </div>
          </div>
        </div>

        {/* Divider */}
        <div style={{ borderTop: "1px solid #e5e7eb", marginBottom: 40 }}></div>

        {/* Bottom Section */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <p style={{ margin: 0, fontSize: 13, color: "#000", fontWeight: 500 }}>
            © 2026 NurseHelpDesk. All rights reserved.
          </p>
          <p style={{ margin: 0, fontSize: 13, color: "#000" }}>
            Advancing Emergency Department Operations.
          </p>
        </div>
      </footer>
      </section>
    </div>
  );
};

export default ClinicalValidation;
