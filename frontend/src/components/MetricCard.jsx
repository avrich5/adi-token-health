import './MetricCard.css';

export default function MetricCard({ label, value, sub, change, changeColor }) {
  return (
    <div className="metric-card card">
      <div className="mc-label">{label}</div>
      <div className="mc-value" style={changeColor === 'red' ? {color:'var(--red)'} : {}}>{value}</div>
      {sub && <div className="mc-sub">{sub}</div>}
      {change && (
        <div className="mc-change" style={{ color: changeColor === 'red' ? 'var(--red)' : changeColor === 'amber' ? 'var(--amber)' : 'var(--green)' }}>
          {change}
        </div>
      )}
    </div>
  );
}
