import './Footer.css';

export default function Footer() {
  return (
    <footer className="ftr">
      <span className="ftr-icon">💡</span>
      <div>
        <strong>Economic Observability Layer</strong> — This dashboard provides formalized economic state detection, not trading signals. States are computed from real-time market data (CoinGecko, Kraken) combined with on-chain and governance parameters. Forces marked with <span className="ftr-dot">●</span> are estimated and will be replaced with real data sources.
      </div>
    </footer>
  );
}
