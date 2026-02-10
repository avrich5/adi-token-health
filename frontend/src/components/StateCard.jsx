import { CATEGORY_COLORS } from '../utils/format';
import './StateCard.css';

const R = 80, C = 2 * Math.PI * R;

export default function StateCard({ state, resistance }) {
  if (!state) return null;
  const cat = CATEGORY_COLORS[state.category] || CATEGORY_COLORS.neutral;
  const health = state.confidence * resistance;
  const healthNorm = Math.min(1, Math.max(0, health));
  const offset = C * (1 - healthNorm * 0.75);
  const icon = state.category === 'positive' ? '✦' : state.category === 'warning' ? '⚠' : '◈';

  return (
    <div className="state-card card">
      <div className="state-label">ECONOMIC STATE</div>
      <div className="state-ring-wrap">
        <svg viewBox="0 0 200 200" className="state-ring">
          <circle cx="100" cy="100" r={R} fill="none" stroke="var(--border)" strokeWidth="6"
            strokeDasharray={`${C * 0.75} ${C * 0.25}`} strokeDashoffset={0}
            transform="rotate(135 100 100)" strokeLinecap="round"/>
          <circle cx="100" cy="100" r={R} fill="none" stroke={cat.color} strokeWidth="6"
            strokeDasharray={`${C * 0.75} ${C * 0.25}`}
            strokeDashoffset={offset}
            transform="rotate(135 100 100)" strokeLinecap="round"
            style={{ transition: 'stroke-dashoffset 1.5s ease-out', filter: `drop-shadow(0 0 8px ${cat.color})` }}/>
        </svg>
        <div className="state-ring-inner">
          <div className="state-icon">{icon}</div>
          <div className="state-health" style={{ color: cat.color }}>{healthNorm.toFixed(2)}</div>
          <div className="state-health-label">HEALTH INDEX</div>
        </div>
      </div>
      <div className="state-name" style={{ color: cat.color }}>{state.name}</div>
      <div className="state-desc">{state.description}</div>
      <div className="state-badge">STATE #{state.id} · {state.category}</div>
    </div>
  );
}
