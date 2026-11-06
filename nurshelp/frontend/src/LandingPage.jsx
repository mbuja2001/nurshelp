import React from "react";
import { useNavigate } from "react-router-dom";
import { FiArrowRight, FiUsers, FiTrendingUp, FiShield, FiClock } from "react-icons/fi";
import nhpLogo from "./Dashboard/NHP_Icon.png";

const LandingPage = () => {
  const navigate = useNavigate();

  const handleLinkHover = (e, color) => {
    e.target.style.color = color;
  };

  const handleGetStarted = () => {
    navigate("/dashboard?register=account");
  };

  const handleLogin = () => {
    navigate("/dashboard");
  };

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
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <img src={nhpLogo} alt="NurseHelpDesk" style={{ height: '6.3rem', width: 'auto' }} />
          <h1 style={{ margin: 0, fontSize: 24, fontWeight: 700 }}>NurseHelpDesk</h1>
        </div>
        <div style={{ display: "flex", gap: 24, alignItems: "center" }}>
          {[
            { label: "Team", path: "/team" },
            { label: "What We Do", path: "/what-we-do" },
            { label: "Ethics & Responsibility", path: "/ethics" },
            { label: "Clinical Validation", path: "/validation" },
            { label: "Security & Compliance", path: "/security" }
          ].map(link => (
            <a
              key={link.path}
              onClick={() => navigate(link.path)}
              style={{
                color: "#cbd5e1",
                cursor: "pointer",
                textDecoration: "none",
                fontWeight: 500,
                transition: "all 200ms"
              }}
              onMouseEnter={(e) => {
                e.target.style.color = "#10b981";
              }}
              onMouseLeave={(e) => {
                e.target.style.color = "#cbd5e1";
              }}
            >
              {link.label}
            </a>
          ))}
        </div>
        <div style={{ display: "flex", gap: 12 }}>
          <button
            onClick={handleLogin}
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
          <button
            onClick={handleGetStarted}
            style={{
              padding: "10px 20px",
              borderRadius: 8,
              border: "none",
              backgroundColor: "#10b981",
              color: "#fff",
              cursor: "pointer",
              fontWeight: 600,
              transition: "all 200ms"
            }}
            onMouseEnter={(e) => {
              e.target.style.backgroundColor = "#059669";
            }}
            onMouseLeave={(e) => {
              e.target.style.backgroundColor = "#10b981";
            }}
          >
            Create Account
          </button>
        </div>
      </nav>

      {/* Hero Section */}
      <section style={{
        backgroundColor: "#0f172a",
        color: "#fff",
        padding: "100px 8%",
        textAlign: "center",
        minHeight: "80vh",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center"
      }}>
        <h2 style={{ fontSize: 56, fontWeight: 800, margin: "0 0 20px 0", lineHeight: 1.2 }}>
          Intelligent Triage & Patient Care Management
        </h2>
        <p style={{ fontSize: 20, color: "#cbd5e1", maxWidth: 700, margin: "0 0 40px 0", lineHeight: 1.6 }}>
          Streamline patient intake, AI-powered clinical assessments, and real-time staff coordination. 
          Designed for modern emergency departments.
        </p>
        <div style={{ display: "flex", gap: 16, justifyContent: "center", flexWrap: "wrap" }}>
          <button
            onClick={handleGetStarted}
            style={{
              padding: "15px 40px",
              borderRadius: 8,
              border: "none",
              backgroundColor: "#10b981",
              color: "#fff",
              fontSize: 16,
              fontWeight: 700,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: 8,
              transition: "all 200ms"
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = "#059669";
              e.currentTarget.style.transform = "translateY(-2px)";
              e.currentTarget.style.boxShadow = "0 10px 20px rgba(16, 185, 129, 0.3)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = "#10b981";
              e.currentTarget.style.transform = "translateY(0)";
              e.currentTarget.style.boxShadow = "none";
            }}
          >
            Create Account
            <FiArrowRight size={20} />
          </button>
          <button
            onClick={handleLogin}
            style={{
              padding: "15px 40px",
              borderRadius: 8,
              border: "2px solid #64748b",
              backgroundColor: "transparent",
              color: "#fff",
              fontSize: 16,
              fontWeight: 700,
              cursor: "pointer",
              transition: "all 200ms"
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = "#fff";
              e.currentTarget.style.backgroundColor = "rgba(255, 255, 255, 0.08)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = "#64748b";
              e.currentTarget.style.backgroundColor = "transparent";
            }}
          >
            Learn More
          </button>
        </div>
      </section>

      {/* Features Section */}
      <section style={{
        padding: "100px 8%",
        backgroundColor: "#fff",
        minHeight: "80vh",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center"
      }}>
        <h3 style={{ textAlign: "center", fontSize: 42, fontWeight: 800, marginBottom: 60, color: "#0f172a" }}>
          Core Features
        </h3>
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(2, 1fr)",
          gap: 40,
          maxWidth: 1200,
          margin: "0 auto",
          width: "100%"
        }}>
          {[
            {
              icon: FiTrendingUp,
              title: "AI-Powered Triage",
              desc: "Intelligent patient assessment with real-time priority scoring and recommendations based on clinical data."
            },
            {
              icon: FiUsers,
              title: "Staff Coordination",
              desc: "Seamless collaboration tools for nurses, specialists, and administrators with real-time availability tracking."
            },
            {
              icon: FiClock,
              title: "Wait Time Tracking",
              desc: "Monitor and optimize patient wait times with predictive analytics and resource allocation insights."
            },
            {
              icon: FiShield,
              title: "Clinical Security",
              desc: "HIPAA-compliant patient data handling with role-based access control and comprehensive audit logging."
            }
          ].map((feature, idx) => {
            const Icon = feature.icon;
            return (
              <div key={idx} style={{
                padding: 32,
                border: "1px solid #e2e8f0",
                borderRadius: 12,
                backgroundColor: "#f8fafc",
                transition: "all 200ms"
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = "#10b981";
                e.currentTarget.style.boxShadow = "0 10px 30px rgba(16, 185, 129, 0.1)";
                e.currentTarget.style.transform = "translateY(-4px)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = "#e2e8f0";
                e.currentTarget.style.boxShadow = "none";
                e.currentTarget.style.transform = "translateY(0)";
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 16 }}>
                  <div style={{
                    width: 48,
                    height: 48,
                    borderRadius: 8,
                    backgroundColor: "#10b981",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    color: "#fff"
                  }}>
                    <Icon size={24} />
                  </div>
                  <h4 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#0f172a" }}>
                    {feature.title}
                  </h4>
                </div>
                <p style={{ margin: 0, fontSize: 14, color: "#64748b", lineHeight: 1.6 }}>
                  {feature.desc}
                </p>
              </div>
            );
          })}
        </div>
      </section>

      {/* CTA Section */}
      <section style={{
        backgroundColor: "#0f172a",
        color: "#fff",
        padding: "80px 8%",
        textAlign: "center"
      }}>
        <h3 style={{ fontSize: 42, fontWeight: 800, margin: "0 0 20px 0", color: "#fff" }}>
          Ready to Improve Patient Care?
        </h3>
        <p style={{ fontSize: 18, color: "#cbd5e1", margin: "0 0 40px 0", maxWidth: 600, marginLeft: "auto", marginRight: "auto" }}>
          Join healthcare teams across the region using NurseHelpDesk to streamline operations and enhance clinical outcomes.
        </p>
        <button
          onClick={handleGetStarted}
          style={{
            padding: "15px 45px",
            borderRadius: 8,
            border: "none",
            backgroundColor: "#10b981",
            color: "#fff",
            fontSize: 16,
            fontWeight: 700,
            cursor: "pointer",
            display: "inline-flex",
            alignItems: "center",
            gap: 8,
            transition: "all 200ms"
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = "#059669";
            e.currentTarget.style.transform = "translateY(-2px)";
            e.currentTarget.style.boxShadow = "0 10px 20px rgba(16, 185, 129, 0.3)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = "#10b981";
            e.currentTarget.style.transform = "translateY(0)";
            e.currentTarget.style.boxShadow = "none";
          }}
        >
          Get Started Now
          <FiArrowRight size={20} />
        </button>
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
      
        {/* Use explicit footer element and override any global footer CSS */}
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
                <h3 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#000000" }}>NurseHelpDesk</h3>
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

export default LandingPage;
