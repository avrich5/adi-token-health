export const fmt = {
  usd: (v) => `$${Number(v).toFixed(2)}`,
  usdCompact: (v) => {
    const n = Number(v);
    if (n >= 1e9) return `$${(n / 1e9).toFixed(1)}B`;
    if (n >= 1e6) return `$${(n / 1e6).toFixed(2)}M`;
    if (n >= 1e3) return `$${(n / 1e3).toFixed(1)}K`;
    return `$${n.toFixed(2)}`;
  },
  pct: (v) => `${v >= 0 ? '+' : ''}${Number(v).toFixed(1)}%`,
  pctAbs: (v) => `${Number(v).toFixed(0)}%`,
  num: (v, d = 2) => Number(v).toFixed(d),
  time: (d) => {
    const dt = new Date(d);
    return dt.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
      + ' · ' + dt.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', timeZoneName: 'short' });
  },
};

export const CATEGORY_COLORS = {
  positive: { color: 'var(--green)', bg: 'var(--green-dim)' },
  caution:  { color: 'var(--amber)', bg: 'var(--amber-dim)' },
  warning:  { color: 'var(--red)',   bg: 'var(--red-dim)' },
  neutral:  { color: 'var(--cyan)',  bg: 'var(--cyan-dim)' },
};
