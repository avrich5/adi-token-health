import './ForcesPanel.css';

function ForceBar({ name, value, isMockup }) {
  const pct = Math.abs(value) * 50;
  const isPos = value >= 0;
  const color = isPos ? 'var(--green)' : 'var(--red)';
  return (
    <div className="force-row">
      <div className="force-name">
        {name}{isMockup && <span className="force-mock-dot">●</span>}
      </div>
      <div className="force-track">
        <div className="force-center" />
        {isPos ? (
          <div className="force-fill force-pos" style={{ width: `${pct}%`, left: '50%', background: color }} />
        ) : (
          <div className="force-fill force-neg" style={{ width: `${pct}%`, right: '50%', background: color }} />
        )}
      </div>
      <div className="force-val" style={{ color }}>{value >= 0 ? '+' : ''}{value.toFixed(2)}</div>
    </div>
  );
}

export default function ForcesPanel({ forces }) {
  if (!forces) return null;
  const mockupCount = forces.items.filter(f => f.is_mockup).length;
  return (
    <div className="forces-panel card">
      <div className="fp-label">ACTIVE FORCES</div>
      <div className="fp-list">
        {forces.items.map(f => (
          <ForceBar key={f.id} name={f.name} value={f.value} isMockup={f.is_mockup} />
        ))}
      </div>
      {mockupCount > 0 && (
        <div className="fp-footer">
          <span className="fp-mock-badge">Mockup data</span>
          <span className="fp-mock-note">{mockupCount} of {forces.items.length} forces estimated</span>
        </div>
      )}
    </div>
  );
}
