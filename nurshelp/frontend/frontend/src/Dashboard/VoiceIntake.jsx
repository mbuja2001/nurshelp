import React from 'react';
import { styles } from './styles';

const VoiceIntake = ({ setView }) => {
  return (
    <div style={styles.patientBg}>
      <style>{`@keyframes pulse { 0% { height: 15px; opacity: 0.5; } 50% { height: 60px; opacity: 1; } 100% { height: 15px; opacity: 0.5; } }`}</style>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%' }}>
        <h1 style={{ fontSize: '2.5rem', marginBottom: '10px' }}>Voice Intake</h1>
        <p style={{ color: '#94a3b8', marginBottom: '40px' }}>Listening to your concern...</p>
        <div style={styles.macRecorder}>
          <div style={styles.waveGroup}>
            {[0.1, 0.3, 0.6, 0.9, 0.6, 0.3, 0.1].map((delay, i) => (
              <div key={i} style={{ ...styles.waveBar, animation: `pulse 1s infinite ease-in-out`, animationDelay: `${delay}s` }}></div>
            ))}
          </div>
          <button onClick={() => setView('patient')} style={styles.stopBtn}>
            <div style={{ width: '16px', height: '16px', backgroundColor: 'white', borderRadius: '2px' }}></div>
          </button>
        </div>
        <button onClick={() => setView('patient')} style={{...styles.btnOutline, marginTop: '50px', width: '200px'}}>Cancel</button>
      </div>
    </div>
  );
};

export default VoiceIntake;
