import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchDashboard } from '../api/client';

export function useDashboardData(intervalMs = 60000) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(null);
  const timer = useRef(null);

  const load = useCallback(async () => {
    try {
      const result = await fetchDashboard();
      setData(result);
      setError(null);
      setLastUpdated(new Date());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    timer.current = setInterval(load, intervalMs);
    return () => clearInterval(timer.current);
  }, [load, intervalMs]);

  return { data, error, loading, lastUpdated, refetch: load };
}
