import React, { createContext, useContext, useEffect, useState, useRef } from 'react';
import { TelemetryData, AnomalyItem, FailurePredictionItem, RULData, MissionEventItem } from '../types/telemetry';
import { api } from '../services/api';

interface TelemetryContextType {
  telemetry: TelemetryData | null;
  history: TelemetryData[];
  anomalies: AnomalyItem[];
  predictions: FailurePredictionItem[];
  rul: RULData | null;
  latestEvent: MissionEventItem | null;
  isConnected: boolean;
  selectedSubsystem: string | null;
  setSelectedSubsystem: (sub: string | null) => void;
  triggerScenario: (name: string) => Promise<void>;
  startSimulation: () => Promise<void>;
  pauseSimulation: () => Promise<void>;
  resetSimulation: () => Promise<void>;
  setSpeed: (speed: number) => Promise<void>;
  speedMultiplier: number;
  isSimRunning: boolean;
}

const TelemetryContext = createContext<TelemetryContextType | undefined>(undefined);

export const TelemetryProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [telemetry, setTelemetry] = useState<TelemetryData | null>(null);
  const [history, setHistory] = useState<TelemetryData[]>([]);
  const [anomalies, setAnomalies] = useState<AnomalyItem[]>([]);
  const [predictions, setPredictions] = useState<FailurePredictionItem[]>([]);
  const [rul, setRul] = useState<RULData | null>(null);
  const [latestEvent, setLatestEvent] = useState<MissionEventItem | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [selectedSubsystem, setSelectedSubsystem] = useState<string | null>(null);
  const [speedMultiplier, setSpeedState] = useState(1.0);
  const [isSimRunning, setIsSimRunning] = useState(true);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<any>(null);

  // Initial fetch of historical buffer
  useEffect(() => {
    api.getTelemetryHistory(60)
      .then(hist => {
        if (hist && hist.length > 0) {
          setHistory(hist);
          setTelemetry(hist[hist.length - 1]);
        }
      })
      .catch(() => {});
  }, []);

  // WebSocket Connection
  useEffect(() => {
    let unmounted = false;

    const connectWebSocket = () => {
      if (unmounted) return;
      const customWs = import.meta.env.VITE_WS_URL;
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = customWs
        ? (customWs.startsWith('ws') ? customWs : `${protocol}//${customWs}`).replace(/\/$/, '') + '/ws/telemetry'
        : `${protocol}//${window.location.host}/ws/telemetry`;

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!unmounted) {
          setIsConnected(true);
        }
      };

      ws.onmessage = (event) => {
        if (unmounted) return;
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'INITIAL_STATE' || msg.type === 'TELEMETRY_UPDATE') {
            const data: TelemetryData = msg.telemetry;
            setTelemetry(data);
            setHistory(prev => {
              const updated = [...prev, data];
              return updated.length > 120 ? updated.slice(-120) : updated;
            });
            if (msg.anomalies) setAnomalies(msg.anomalies);
            if (msg.predictions) setPredictions(msg.predictions);
            if (msg.rul) setRul(msg.rul);
            if (msg.latest_event) setLatestEvent(msg.latest_event);
          }
        } catch (e) {
          console.error('Error parsing WS message', e);
        }
      };

      ws.onclose = () => {
        if (!unmounted) {
          setIsConnected(false);
          reconnectTimeoutRef.current = setTimeout(connectWebSocket, 2000);
        }
      };

      ws.onerror = () => {
        ws.close();
      };
    };

    connectWebSocket();

    return () => {
      unmounted = true;
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
    };
  }, []);

  // Fallback Polling if WebSocket is disconnected
  useEffect(() => {
    if (isConnected) return;
    const interval = setInterval(async () => {
      try {
        const latest = await api.getLatestTelemetry();
        setTelemetry(latest);
        setHistory(prev => [...prev.slice(-119), latest]);
        const anom = await api.getAnomalies();
        setAnomalies(anom);
        const pred = await api.getPredictions();
        setPredictions(pred);
      } catch (e) {}
    }, 2000);

    return () => clearInterval(interval);
  }, [isConnected]);

  const triggerScenario = async (name: string) => {
    setTelemetry(prev => (prev ? { ...prev, operating_mode: name } : null));
    await api.triggerScenario(name);
  };

  const startSimulation = async () => {
    await api.startSimulation();
    setIsSimRunning(true);
  };

  const pauseSimulation = async () => {
    await api.pauseSimulation();
    setIsSimRunning(false);
  };

  const resetSimulation = async () => {
    await api.resetSimulation();
    setHistory([]);
  };

  const setSpeed = async (speed: number) => {
    await api.setSpeed(speed);
    setSpeedState(speed);
  };

  return (
    <TelemetryContext.Provider
      value={{
        telemetry,
        history,
        anomalies,
        predictions,
        rul,
        latestEvent,
        isConnected,
        selectedSubsystem,
        setSelectedSubsystem,
        triggerScenario,
        startSimulation,
        pauseSimulation,
        resetSimulation,
        setSpeed,
        speedMultiplier,
        isSimRunning
      }}
    >
      {children}
    </TelemetryContext.Provider>
  );
};

export const useTelemetry = () => {
  const context = useContext(TelemetryContext);
  if (!context) throw new Error('useTelemetry must be used within a TelemetryProvider');
  return context;
};
