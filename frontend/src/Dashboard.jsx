import { useDashboardData } from './hooks/useDashboardData';
import { fmt } from './utils/format';
import Header from './components/Header';
import StateCard from './components/StateCard';
import MetricCard from './components/MetricCard';
import ForcesPanel from './components/ForcesPanel';
import TransitionsPanel from './components/TransitionsPanel';
import Timeline from './components/Timeline';
import Footer from './components/Footer';
import './Dashboard.css';

export default function Dashboard() {
  const { data, error, loading, refetch } = useDashboardData(60000);

  if (loading) return <div className="dash-loading">Connecting to API…</div>;
  if (error) return (
    <div className="dash-error">
      <span>⚠ {error}</span>
      <button onClick={refetch}>Retry</button>
    </div>
  );
  if (!data) return null;

  const { state, price, resistance, forces, transitions, timestamp } = data;
  const volRatio = price.volume_24h / (price.volume_24h * 0.85);

  return (
    <div className="dashboard">
      <Header timestamp={timestamp} />

      <div className="hero-grid">
        <StateCard state={state} resistance={resistance.value} />

        <div className="metrics-row">
          <MetricCard
            label="PRICE" value={fmt.usd(price.current)} sub="ADI / USD"
            change={`▴ ${fmt.pct(price.change_7d_pct)} 7d`}
            changeColor={price.change_7d_pct >= 0 ? 'green' : 'red'}
          />
          <MetricCard
            label="VOLUME 24H" value={fmt.usdCompact(price.volume_24h)} sub="Trading volume"
            change={`▴ ${volRatio.toFixed(2)}x vs 7d avg`}
            changeColor={volRatio >= 1 ? 'green' : 'amber'}
          />
          <MetricCard
            label="MARKET CAP" value={fmt.usdCompact(price.market_cap)} sub="Fully diluted"
          />
          <MetricCard
            label="MARKET RESISTANCE" value={fmt.num(resistance.value)} sub="Absorption capacity"
            change={resistance.value < 0.1 ? '▾ Critically low' : resistance.value < 0.3 ? '◈ Low' : '▴ Adequate'}
            changeColor={resistance.value < 0.1 ? 'red' : resistance.value < 0.3 ? 'amber' : 'green'}
          />
        </div>

        <ForcesPanel forces={forces} />
        <TransitionsPanel transitions={transitions} />
      </div>

      <Timeline />

      <Footer />
    </div>
  );
}
