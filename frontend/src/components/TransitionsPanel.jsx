import { CATEGORY_COLORS } from '../utils/format';
import './TransitionsPanel.css';

export default function TransitionsPanel({ transitions }) {
  if (!transitions || transitions.length === 0) return null;
  return (
    <div className="trans-panel card">
      <div className="tp-label">PROBABLE TRANSITIONS</div>
      <div className="tp-list">
        {transitions.map((t, i) => {
          const cat = CATEGORY_COLORS[t.category] || CATEGORY_COLORS.neutral;
          return (
            <div className="tp-item" key={i}>
              <div className="tp-head">
                <span className="tp-name" style={{ color: cat.color }}>{t.to_state}</span>
                <span className="tp-pct" style={{ color: cat.color }}>{t.probability}%</span>
              </div>
              <div className="tp-bar-track">
                <div className="tp-bar-fill" style={{ width: `${t.probability}%`, background: cat.color, opacity: 0.6 }} />
              </div>
              <div className="tp-trigger">{t.trigger}</div>
            </div>
          );
        })}
      </div>
      <div className="tp-badge">Scenario model</div>
    </div>
  );
}
