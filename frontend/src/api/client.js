const API_BASE = 'http://localhost:8002';

export async function fetchDashboard() {
  const res = await fetch(`${API_BASE}/api/data`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchHistory() {
  const res = await fetch(`${API_BASE}/api/history`);
  if (!res.ok) throw new Error(`History API error: ${res.status}`);
  return res.json();
}

export async function fetchScenario(adjustedForces) {
  const res = await fetch(`${API_BASE}/api/scenario`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ adjusted_forces: adjustedForces }),
  });
  if (!res.ok) throw new Error(`Scenario API error: ${res.status}`);
  return res.json();
}
