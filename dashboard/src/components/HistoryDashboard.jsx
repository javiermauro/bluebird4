import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';

// API endpoints
const API_BASE = 'http://localhost:8000';

function HistoryDashboard() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);

  // State for equity curve
  const [equity, setEquity] = useState({
    dates: [],
    equity: [],
    peak: 0,
    peak_date: null,
    trough: 0,
    trough_date: null,
    current: 0,
    starting: 100000,
    recovery_pct: 0,
    total_return_pct: 0
  });

  // State for realized P/L
  const [pnl, setPnl] = useState({
    total: 0,
    by_day: [],
    by_symbol: {},
    metrics: {
      win_rate: 0,
      total_trades: 0,
      total_orders: 0,
      avg_per_symbol: 0,
      best_day: { date: null, pnl: 0 },
      worst_day: { date: null, pnl: 0 }
    }
  });

  // State for trade history
  const [trades, setTrades] = useState({
    trades: [],
    total: 0
  });

  // Filter state
  const [symbolFilter, setSymbolFilter] = useState('all');
  const [periodFilter, setPeriodFilter] = useState('1M');

  // Fetch all history data
  const fetchHistoryData = async () => {
    try {
      setLoading(true);

      const [equityRes, pnlRes, tradesRes] = await Promise.all([
        fetch(`${API_BASE}/api/history/equity?period=${periodFilter}`),
        fetch(`${API_BASE}/api/history/realized-pnl?days=30`),
        fetch(`${API_BASE}/api/history/trades?days=7&limit=100`)
      ]);

      const equityData = await equityRes.json();
      const pnlData = await pnlRes.json();
      const tradesData = await tradesRes.json();

      if (!equityData.error) setEquity(equityData);
      if (!pnlData.error) setPnl(pnlData);
      if (!tradesData.error) setTrades(tradesData);

      setLastUpdate(new Date());
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistoryData();
    const interval = setInterval(fetchHistoryData, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, [periodFilter]);

  // Format currency
  const formatCurrency = (value, decimals = 2) => {
    if (value === null || value === undefined) return '$0.00';
    const sign = value >= 0 ? '' : '-';
    return `${sign}$${Math.abs(value).toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}`;
  };

  // Format percentage
  const formatPercent = (value) => {
    if (value === null || value === undefined) return '0.0%';
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(1)}%`;
  };

  // Chart configuration for equity curve
  const equityChartData = {
    labels: equity.dates,
    datasets: [
      {
        label: 'Equity',
        data: equity.equity,
        borderColor: '#d4af37',
        backgroundColor: 'rgba(212, 175, 55, 0.1)',
        tension: 0.4,
        pointRadius: 0,
        fill: true,
        borderWidth: 2
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(26, 26, 46, 0.95)',
        titleColor: '#d4af37',
        bodyColor: '#e8e6e3',
        borderColor: 'rgba(212, 175, 55, 0.3)',
        borderWidth: 1,
        callbacks: {
          label: function(context) {
            return `$${context.parsed.y.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
          }
        }
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(212, 175, 55, 0.05)' },
        ticks: {
          color: '#6b7a94',
          maxRotation: 0,
          maxTicksLimit: 8
        }
      },
      y: {
        grid: { color: 'rgba(212, 175, 55, 0.05)' },
        ticks: {
          color: '#6b7a94',
          callback: (value) => '$' + (value / 1000).toFixed(0) + 'k'
        }
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    }
  };

  // Get symbols for filter
  const symbols = ['all', ...Object.keys(pnl.by_symbol)];

  // Filter trades by symbol
  const filteredTrades = symbolFilter === 'all'
    ? trades.trades
    : trades.trades.filter(t => t.symbol === symbolFilter);

  return (
    <div className="history-dashboard">
      {/* Header */}
      <div className="history-header">
        <div className="history-title">
          <h2>Account History</h2>
          <span className="history-subtitle">Realized P/L & Performance Tracking</span>
        </div>
        <div className="history-controls">
          <select
            value={periodFilter}
            onChange={(e) => setPeriodFilter(e.target.value)}
            className="history-select"
          >
            <option value="1W">1 Week</option>
            <option value="1M">1 Month</option>
            <option value="3M">3 Months</option>
            <option value="1A">1 Year</option>
          </select>
          <button onClick={fetchHistoryData} className="btn btn-secondary btn-sm">
            {loading ? 'Loading...' : 'Refresh'}
          </button>
        </div>
      </div>

      {error && (
        <div className="history-error">
          <span>Error loading data: {error}</span>
        </div>
      )}

      {/* Key Metrics Row */}
      <div className="history-metrics-grid">
        <div className="history-metric-card">
          <div className="metric-label">REALIZED P/L</div>
          <div className={`metric-value ${pnl.total >= 0 ? 'positive' : 'negative'}`}>
            {formatCurrency(pnl.total)}
          </div>
          <div className="metric-sublabel">Last 30 Days</div>
        </div>

        <div className="history-metric-card">
          <div className="metric-label">WIN RATE</div>
          <div className="metric-value">
            {pnl.metrics.win_rate}%
          </div>
          <div className="metric-sublabel">
            {pnl.metrics.total_trades} symbols traded
          </div>
        </div>

        <div className="history-metric-card">
          <div className="metric-label">TOTAL RETURN</div>
          <div className={`metric-value ${equity.total_return_pct >= 0 ? 'positive' : 'negative'}`}>
            {formatPercent(equity.total_return_pct)}
          </div>
          <div className="metric-sublabel">From $100,000 start</div>
        </div>

        <div className="history-metric-card">
          <div className="metric-label">RECOVERY</div>
          <div className="metric-value recovery">
            {equity.recovery_pct.toFixed(1)}%
          </div>
          <div className="metric-sublabel">
            ${(equity.starting - equity.current).toLocaleString()} to break-even
          </div>
        </div>
      </div>

      {/* Equity Curve Chart */}
      <div className="history-section">
        <div className="section-header">
          <h3>Equity Curve</h3>
          <div className="equity-stats">
            <span className="stat">
              Peak: {formatCurrency(equity.peak)}
              {equity.peak_date && <small> ({equity.peak_date.slice(0,10)})</small>}
            </span>
            <span className="stat">
              Trough: {formatCurrency(equity.trough)}
              {equity.trough_date && <small> ({equity.trough_date.slice(0,10)})</small>}
            </span>
            <span className="stat">
              Current: {formatCurrency(equity.current)}
            </span>
          </div>
        </div>
        <div className="equity-chart-container">
          {equity.equity.length > 0 ? (
            <Line data={equityChartData} options={chartOptions} />
          ) : (
            <div className="no-data">No equity data available</div>
          )}
        </div>
      </div>

      {/* Recovery Progress Bar */}
      <div className="history-section recovery-section">
        <div className="section-header">
          <h3>Recovery Progress</h3>
          <span className="recovery-target">Target: $100,000</span>
        </div>
        <div className="recovery-bar-container">
          <div className="recovery-bar">
            <div
              className="recovery-fill"
              style={{ width: `${Math.min(100, Math.max(0, equity.recovery_pct))}%` }}
            />
            <div className="recovery-markers">
              <span className="marker trough">${(equity.trough / 1000).toFixed(1)}k</span>
              <span className="marker current">${(equity.current / 1000).toFixed(1)}k</span>
              <span className="marker target">$100k</span>
            </div>
          </div>
          <div className="recovery-labels">
            <span>Lowest: {equity.trough_date ? equity.trough_date.slice(0,10) : '-'}</span>
            <span className="recovery-pct">{equity.recovery_pct.toFixed(1)}% Recovered</span>
            <span>Break-even</span>
          </div>
        </div>
      </div>

      {/* Two Column Layout: Daily P/L and Symbol Performance */}
      <div className="history-two-column">
        {/* Daily Activity */}
        <div className="history-section">
          <div className="section-header">
            <h3>Daily Activity</h3>
            <span className="section-subtitle">{pnl.by_day.length} days</span>
          </div>
          <div className="daily-pnl-table">
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Trades</th>
                  <th>Buys</th>
                  <th>Sells</th>
                </tr>
              </thead>
              <tbody>
                {pnl.by_day.slice(0, 10).map((day, idx) => (
                  <tr key={idx}>
                    <td className="date">{day.date}</td>
                    <td className="trades">{day.trades}</td>
                    <td className="buys">{day.buys}</td>
                    <td className="sells">{day.sells}</td>
                  </tr>
                ))}
                {pnl.by_day.length === 0 && (
                  <tr>
                    <td colSpan="4" className="no-data">No trading activity</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Symbol Performance */}
        <div className="history-section">
          <div className="section-header">
            <h3>By Symbol</h3>
            <span className="section-subtitle">Realized P/L</span>
          </div>
          <div className="symbol-performance">
            {Object.entries(pnl.by_symbol)
              .sort((a, b) => b[1].pnl - a[1].pnl)
              .map(([symbol, data]) => (
                <div key={symbol} className="symbol-row">
                  <div className="symbol-info">
                    <span className="symbol-name">{symbol}</span>
                    <span className="symbol-trades">{data.trades} trades</span>
                  </div>
                  <div className="symbol-pnl-bar">
                    <div
                      className={`pnl-fill ${data.pnl >= 0 ? 'positive' : 'negative'}`}
                      style={{
                        width: `${Math.min(100, Math.abs(data.pnl) / Math.max(...Object.values(pnl.by_symbol).map(s => Math.abs(s.pnl))) * 100)}%`
                      }}
                    />
                  </div>
                  <div className={`symbol-pnl ${data.pnl >= 0 ? 'positive' : 'negative'}`}>
                    {formatCurrency(data.pnl)}
                  </div>
                </div>
              ))}
            {Object.keys(pnl.by_symbol).length === 0 && (
              <div className="no-data">No symbol data</div>
            )}
          </div>
        </div>
      </div>

      {/* Trade History Table */}
      <div className="history-section trade-history-section">
        <div className="section-header">
          <h3>Trade History</h3>
          <div className="trade-filters">
            <select
              value={symbolFilter}
              onChange={(e) => setSymbolFilter(e.target.value)}
              className="history-select"
            >
              {symbols.map(s => (
                <option key={s} value={s}>{s === 'all' ? 'All Symbols' : s}</option>
              ))}
            </select>
            <span className="trade-count">{filteredTrades.length} trades</span>
          </div>
        </div>
        <div className="trade-history-table">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Symbol</th>
                <th>Side</th>
                <th>Quantity</th>
                <th>Price</th>
                <th>Value</th>
                <th>P/L</th>
              </tr>
            </thead>
            <tbody>
              {filteredTrades.slice(0, 50).map((trade, idx) => (
                <tr key={idx} className={trade.side}>
                  <td className="time">
                    <span className="date">{trade.date}</span>
                    <span className="clock">{trade.time}</span>
                  </td>
                  <td className="symbol">{trade.symbol}</td>
                  <td className={`side ${trade.side}`}>
                    {trade.side.toUpperCase()}
                  </td>
                  <td className="qty">{trade.qty.toFixed(6)}</td>
                  <td className="price">{formatCurrency(trade.price)}</td>
                  <td className="value">{formatCurrency(trade.value)}</td>
                  <td className={`pnl ${trade.pnl !== null ? (trade.pnl >= 0 ? 'positive' : 'negative') : ''}`}>
                    {trade.pnl !== null ? formatCurrency(trade.pnl) : '-'}
                  </td>
                </tr>
              ))}
              {filteredTrades.length === 0 && (
                <tr>
                  <td colSpan="7" className="no-data">No trades found</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Footer */}
      <div className="history-footer">
        <span>Last updated: {lastUpdate ? lastUpdate.toLocaleTimeString() : '-'}</span>
        <span>Data from Alpaca API</span>
      </div>
    </div>
  );
}

export default HistoryDashboard;
