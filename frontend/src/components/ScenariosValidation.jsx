import { useState, useMemo } from 'react';
import { useScenariosData } from '../hooks/useScenariosData';
import './ScenariosValidation.css';

const CAT_COLORS = {
  healthy: '#00E5A0',
  caution: '#FFB547',
  warning: '#FF5C5C',
};

export default function ScenariosValidation() {
  const { data, error, loading } = useScenariosData();
  const [expandedDay, setExpandedDay] = useState(null);

  const days = useMemo(() => data?.scenario_days || [], [data]);
  const summary = data?.summary;

  if (loading) return <div className="sv-loading">Loading scenarios…</div>;
  if (error) return <div className="sv-error">⚠ {error}</div>;
  if (!days.length) return null;

  return (
    <section className="sv-section">
      {/* Header */}
      <div className="sv-header">
        <div className="sv-header-left">
          <h2 className="sv-title">Scenario Validation</h2>
          <span className="sv-sub">From probable to proven · Daily bruteforce analysis</span>
        </div>
        <div className="sv-summary-badges">
          <div className="sv-badge knowledge">
            <span className="sv-badge-num">{summary?.knowledge_units ?? 0}</span>
            <span className="sv-badge-label">validated units</span>
          </div>
          <div className="sv-badge rate">
            <span className="sv-badge-num">{summary?.validation_rate ?? 0}%</span>
            <span className="sv-badge-label">hit rate</span>
          </div>
          <div className="sv-badge total">
            <span className="sv-badge-num">{summary?.scenarios_per_day ?? 0}</span>
            <span className="sv-badge-label">scenarios/day</span>
          </div>
        </div>
      </div>

      {/* Validation chain */}
      <div className="sv-chain">
        {days.map((day, idx) => {
          const v = day.validation;
          const isLast = idx === days.length - 1;
          const isExpanded = expandedDay === idx;
          const statusClass = !v ? 'pending' : v.status === 'validated' ? 'validated' : 'missed';

          return (
            <div className="sv-day-wrapper" key={day.date}>
              {/* Card */}
              <div className={`sv-card ${statusClass}`}>
                {/* Top: current state + date */}
                <div className="sv-card-top">
                  <span className="sv-card-date">{day.date.slice(5)}</span>
                  <span className="sv-card-state"
                    style={{ color: CAT_COLORS[day.current_state.category] || '#8B90A8' }}>
                    {day.current_state.short_name}
                  </span>
                </div>

                {/* Top-3 scenarios */}
                <div className="sv-card-scenarios">
                  {day.top_scenarios.map((s, si) => (
                    <div className="sv-scenario-row" key={si}>
                      <span className="sv-scenario-rank">#{s.rank}</span>
                      <span className="sv-scenario-name"
                        style={{ color: CAT_COLORS[s.category] || '#8B90A8' }}>
                        {s.short_name || s.state_name}
                      </span>
                      <span className="sv-scenario-prob">{s.probability}%</span>
                    </div>
                  ))}
                </div>

                {/* Validation result */}
                <div className={`sv-card-validation ${statusClass}`}>
                  {!v ? (
                    <span className="sv-val-pending">⏳ Awaiting validation</span>
                  ) : v.status === 'validated' ? (
                    <span className="sv-val-ok">✓ Rank #{v.matched_rank} confirmed</span>
                  ) : (
                    <span className="sv-val-miss">✗ Actual: {v.actual_short || v.actual_state}</span>
                  )}
                </div>

                {/* Explore all button */}
                <button
                  className="sv-explore-btn"
                  onClick={() => setExpandedDay(isExpanded ? null : idx)}
                >
                  {isExpanded ? 'Collapse' : `Explore all ${day.scenarios_total}`}
                </button>

                {/* Expanded: risk distribution */}
                {isExpanded && (
                  <div className="sv-expanded">
                    <div className="sv-risk-grid">
                      <div className="sv-risk-item critical">
                        <span className="sv-risk-num">{day.risk_distribution.critical}</span>
                        <span className="sv-risk-label">critical</span>
                      </div>
                      <div className="sv-risk-item attention">
                        <span className="sv-risk-num">{day.risk_distribution.attention}</span>
                        <span className="sv-risk-label">attention</span>
                      </div>
                      <div className="sv-risk-item stable">
                        <span className="sv-risk-num">{day.risk_distribution.stable}</span>
                        <span className="sv-risk-label">stable</span>
                      </div>
                    </div>
                    <div className="sv-expanded-note">
                      {day.scenarios_total} force combinations analyzed
                      <br />
                      3 levels × 5 forces = full permutation space
                    </div>
                  </div>
                )}
              </div>

              {/* Arrow between cards */}
              {!isLast && (
                <div className={`sv-arrow ${statusClass}`}>
                  <svg width="24" height="16" viewBox="0 0 24 16">
                    <path d="M0 8 L18 8 M14 3 L20 8 L14 13"
                      stroke="currentColor" strokeWidth="1.5" fill="none" />
                  </svg>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Bottom: growing knowledge counter */}
      <div className="sv-knowledge-bar">
        <div className="sv-kb-track">
          <div
            className="sv-kb-fill"
            style={{ width: `${summary?.validation_rate ?? 0}%` }}
          />
        </div>
        <div className="sv-kb-text">
          Validated economic memory: <strong>{summary?.knowledge_units ?? 0}</strong> of{' '}
          {summary?.validatable_count ?? 0} scenarios ({summary?.validation_rate ?? 0}%)
          <span className="sv-kb-insight">
            · Each validated day = knowledge unit for governance decisions
          </span>
        </div>
      </div>
    </section>
  );
}
