import { TelemetryData, SubsystemDetail, AnomalyItem, FailurePredictionItem, RootCauseAnalysisData, Recommendation, RULData, MissionEventItem } from '../types/telemetry';

const API_BASE = (import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '');

export const api = {
  async getSpacecraft(): Promise<any> {
    const res = await fetch(`${API_BASE}/spacecraft`);
    if (!res.ok) throw new Error('Failed to fetch spacecraft');
    return res.json();
  },

  async getLatestTelemetry(): Promise<TelemetryData> {
    const res = await fetch(`${API_BASE}/telemetry/latest`);
    if (!res.ok) throw new Error('Failed to fetch latest telemetry');
    return res.json();
  },

  async getTelemetryHistory(points: number = 60): Promise<TelemetryData[]> {
    const res = await fetch(`${API_BASE}/telemetry/history?points=${points}`);
    if (!res.ok) throw new Error('Failed to fetch history');
    return res.json();
  },

  async getSubsystems(): Promise<SubsystemDetail[]> {
    const res = await fetch(`${API_BASE}/subsystems`);
    if (!res.ok) throw new Error('Failed to fetch subsystems');
    return res.json();
  },

  async getAnomalies(): Promise<AnomalyItem[]> {
    const res = await fetch(`${API_BASE}/anomalies`);
    if (!res.ok) throw new Error('Failed to fetch anomalies');
    return res.json();
  },

  async getPredictions(): Promise<FailurePredictionItem[]> {
    const res = await fetch(`${API_BASE}/predictions`);
    if (!res.ok) throw new Error('Failed to fetch predictions');
    return res.json();
  },

  async getRootCause(): Promise<RootCauseAnalysisData> {
    const res = await fetch(`${API_BASE}/root-cause/latest`);
    if (!res.ok) throw new Error('Failed to fetch root cause');
    return res.json();
  },

  async getRecommendations(): Promise<Recommendation[]> {
    const res = await fetch(`${API_BASE}/recommendations`);
    if (!res.ok) throw new Error('Failed to fetch recommendations');
    return res.json();
  },

  async getRUL(): Promise<RULData> {
    const res = await fetch(`${API_BASE}/rul`);
    if (!res.ok) throw new Error('Failed to fetch RUL');
    return res.json();
  },

  async getTimeline(): Promise<MissionEventItem[]> {
    const res = await fetch(`${API_BASE}/timeline`);
    if (!res.ok) throw new Error('Failed to fetch timeline');
    return res.json();
  },

  async getModelStatus(): Promise<any[]> {
    const res = await fetch(`${API_BASE}/models/status`);
    if (!res.ok) throw new Error('Failed to fetch model status');
    return res.json();
  },

  async getNasaChannels(): Promise<any[]> {
    const res = await fetch(`${API_BASE}/nasa/channels`);
    if (!res.ok) throw new Error('Failed to fetch NASA channels');
    return res.json();
  },

  async getNasaChannelData(chan_id: string): Promise<any> {
    const res = await fetch(`${API_BASE}/nasa/channels/${chan_id}`);
    if (!res.ok) throw new Error(`Failed to fetch NASA channel ${chan_id}`);
    return res.json();
  },

  async triggerScenario(scenario: string): Promise<any> {
    const res = await fetch(`${API_BASE}/demo/trigger`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenario })
    });
    return res.json();
  },

  async startSimulation(): Promise<any> {
    return (await fetch(`${API_BASE}/simulation/start`, { method: 'POST' })).json();
  },

  async pauseSimulation(): Promise<any> {
    return (await fetch(`${API_BASE}/simulation/pause`, { method: 'POST' })).json();
  },

  async resetSimulation(): Promise<any> {
    return (await fetch(`${API_BASE}/simulation/reset`, { method: 'POST' })).json();
  },

  async setSpeed(speed: number): Promise<any> {
    return (await fetch(`${API_BASE}/simulation/speed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ speed })
    })).json();
  },

  async runWhatIf(params: any): Promise<any> {
    const res = await fetch(`${API_BASE}/simulation/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    });
    return res.json();
  },

  async chat(message: string, history: any[] = []): Promise<any> {
    const res = await fetch(`${API_BASE}/ai/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, conversation_history: history })
    });
    return res.json();
  }
};
