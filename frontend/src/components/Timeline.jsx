import { useMemo } from 'react';
import {
  ComposedChart, Area, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, ReferenceArea
} from 'recharts';
import { useHistoryData } from '../hooks/useHistoryData';
import './Timeline.css';

const STATE_COLORS = {
  healthy: { fill: 'rgba(0,229,160,0.08)', stroke: '#00E5A0' },
  caution: { fill: 'rgba(255,181,71,0.06)', stroke: '#FFB547' },
  warning: { fill: 'rgba(255,92,92,0.06)', stroke: '#FF5C5C' },
};

export default function Timeline() {
  const { data, error, loading } = useHistoryData();

  // Build chart data + state zones
  const { chartData, stateZones } = useMemo(() => {
    if (!data?.timeline?.length) return { chartData: [], stateZones: [] };
    const tl = data.timeline;

    // Normalize resistance to 0-1 via log scale for overlay
    const resValues = tl.map(d => d.resistance).filter(v => v > 0);
    const logMin = Math.log(Math.min(...resValues) || 1);
    const logMax = Math.log(Math.max(...resValues) || 2);
    const logRange = logMax - logMin || 1;

    const cd = tl.map(d => ({
      date: d.date,
      dateShort: d.date.slice(5), // MM-DD
      price: d.price,
      resistance: d.resistance,
      resistanceNorm: d.resistance > 0
        ? Math.max(0, Math.min(1, (Math.log(d.resistance) - logMin) / logRange))
        : 0,
      stateId: d.state_id,
      stateName: d.state_name,
      category: d.category,
      change7d: d.change_7d_pct,
      volumeUsd: d.volume_usd,
    }));

    // Build contiguous state zones for background coloring
    const zones = [];
    let current = null;
    for (let i = 0; i < cd.length; i++) {
      const cat = cd[i].category;
      const stId = cd[i].stateId;
      if (!current || current.stateId !== stId) {
        if (current) current.x2 = cd[i - 1].date;
        current = {
          stateId: stId, stateName: cd[i].stateName,
          category: cat, x1: cd[i].date, x2: cd[i].date,
        };
        zones.push(current);
      }
    }
    if (current && cd.length) current.x2 = cd[cd.length - 1].date;

    return { chartData: cd, stateZones: zones };
  }, [data]);

  if (loading) return <div className="timeline-loading">Loading history…</div>;
  if (error) return <div className="timeline-error">⚠ {error}</div>;
  if (!chartData.length) return null;

  return (
    <section className="timeline-section">
      <div className="timeline-header">
        <h2 className="timeline-title">Economic State Timeline</h2>
        <span className="timeline-sub">{data.period_days} days · {data.data_source}</span>
      </div>

      {/* State legend ribbon */}
      <div className="state-ribbon">
        {stateZones.map((z, i) => (
          <div key={i} className={`ribbon-segment cat-${z.category}`}
               style={{ flex: 1 }}
               title={`${z.stateName} (${z.x1} → ${z.x2})`}>
            <span className="ribbon-label">{z.stateName}</span>
          </div>
        ))}
      </div>

      <div className="timeline-chart-wrap">
        <ResponsiveContainer width="100%" height={320}>
          <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />

            {/* State zone backgrounds */}
            {stateZones.map((z, i) => (
              <ReferenceArea
                key={i} x1={z.x1} x2={z.x2}
                fill={STATE_COLORS[z.category]?.fill || 'transparent'}
                fillOpacity={1} ifOverflow="extendDomain"
              />
            ))}

            <XAxis
              dataKey="date" tick={{ fill: '#555874', fontSize: 11 }}
              tickFormatter={v => v.slice(5)} interval="preserveStartEnd"
              axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
            />
            <YAxis
              yAxisId="price" orientation="left"
              tick={{ fill: '#8B90A8', fontSize: 11 }}
              domain={['auto', 'auto']}
              tickFormatter={v => `$${v.toFixed(2)}`}
              axisLine={false} tickLine={false}
            />
            <YAxis
              yAxisId="resistance" orientation="right"
              tick={{ fill: '#00DCFF', fontSize: 11 }}
              domain={[0, 1]}
              tickFormatter={v => `${(v * 100).toFixed(0)}%`}
              axisLine={false} tickLine={false}
            />

            <Tooltip content={<CustomTooltip />} />

            {/* Price area */}
            <Area
              yAxisId="price" type="monotone" dataKey="price"
              stroke="#F0F2F8" strokeWidth={2}
              fill="rgba(240,242,248,0.04)"
              dot={false} activeDot={{ r: 4, fill: '#F0F2F8' }}
            />

            {/* Resistance line */}
            <Line
              yAxisId="resistance" type="monotone" dataKey="resistanceNorm"
              stroke="#00DCFF" strokeWidth={2} strokeDasharray="6 3"
              dot={false} activeDot={{ r: 4, fill: '#00DCFF' }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="timeline-legend">
        <span className="legend-item"><span className="legend-line price-line" /> Price (USD)</span>
        <span className="legend-item"><span className="legend-line resistance-line" /> Market Resistance (normalized)</span>
        <span className="legend-item"><span className="legend-dot cat-healthy" /> Healthy</span>
        <span className="legend-item"><span className="legend-dot cat-caution" /> Caution</span>
        <span className="legend-item"><span className="legend-dot cat-warning" /> Warning</span>
      </div>
    </section>
  );
}


function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;

  const catColor = d.category === 'healthy' ? '#00E5A0'
    : d.category === 'warning' ? '#FF5C5C' : '#FFB547';

  return (
    <div className="tl-tooltip">
      <div className="tl-tooltip-date">{d.date}</div>
      <div className="tl-tooltip-state" style={{ color: catColor }}>{d.stateName}</div>
      <div className="tl-tooltip-row">
        <span>Price</span>
        <strong>${d.price.toFixed(4)}</strong>
      </div>
      <div className="tl-tooltip-row">
        <span>Resistance</span>
        <strong>{d.resistance.toLocaleString()}</strong>
      </div>
      <div className="tl-tooltip-row">
        <span>Volume (USD)</span>
        <strong>${Number(d.volumeUsd).toLocaleString()}</strong>
      </div>
      <div className="tl-tooltip-row">
        <span>7d Change</span>
        <strong style={{ color: d.change7d >= 0 ? '#00E5A0' : '#FF5C5C' }}>
          {d.change7d >= 0 ? '+' : ''}{d.change7d.toFixed(1)}%
        </strong>
      </div>
    </div>
  );
}
