import asyncio
import copy
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import deque

from backend.app.simulation.physics_engine import PhysicsEngine
from backend.app.simulation.scenario_manager import ScenarioManager
from backend.app.ml.anomaly_detector import anomaly_detector
from backend.app.ml.failure_predictor import failure_predictor
from backend.app.ml.rul_estimator import rul_estimator
from backend.app.websocket.connection_manager import connection_manager

class TelemetryService:
    """
    Central telemetry coordinator. Drives physics ticks, runs ML pipelines,
    manages history ring buffers, handles scenario transitions, and broadcasts to WebSockets.
    """
    def __init__(self):
        self.physics = PhysicsEngine()
        self.scenario_mgr = ScenarioManager()
        
        # State control
        self.is_running = True
        self.speed_multiplier = 1.0
        self.mission_elapsed_seconds = 3648240  # 42 days, 5 hours, etc.
        self.is_loop_running = False
        self._last_tick_time = datetime.utcnow()
        
        # Buffers
        self.history: deque = deque(maxlen=300)
        self.active_anomalies: List[Dict[str, Any]] = []
        self.latest_predictions: List[Dict[str, Any]] = []
        self.latest_rul: Dict[str, Any] = {}
        self.timeline_events: List[Dict[str, Any]] = []
        self.last_stage = "NORMAL"
        self.event_counter = 1
        
        # Initial nominal event
        self._record_event("INFO", "MISSION START", "Orbital Twin simulation engine initialized in nominal baseline mode.", "SYSTEM")
        
        # Immediate initial tick so state and ML predictions are pre-populated on cold start
        self._process_tick(1.0)

    def _record_event(self, severity: str, title: str, description: str, subsystem: str = "SYSTEM"):
        event = {
            "id": self.event_counter,
            "timestamp": datetime.utcnow().isoformat(),
            "severity": severity,
            "title": title,
            "description": description,
            "subsystem": subsystem
        }
        self.event_counter += 1
        self.timeline_events.insert(0, event)
        # Keep latest 60 events
        if len(self.timeline_events) > 60:
            self.timeline_events.pop()

    def set_running(self, running: bool):
        self.is_running = running
        status_str = "RESUMED" if running else "PAUSED"
        self._record_event("INFO", f"SIMULATION {status_str}", f"Operator toggled simulation state to {status_str}.", "CONTROL")

    def set_speed(self, speed: float):
        self.speed_multiplier = max(0.5, min(10.0, speed))
        self._record_event("INFO", "TIME ACCELERATION", f"Simulation time step set to {self.speed_multiplier}x.", "CONTROL")

    def reset(self):
        self.physics = PhysicsEngine()
        self.scenario_mgr = ScenarioManager()
        self.history.clear()
        self.active_anomalies.clear()
        self.latest_predictions.clear()
        self.latest_rul.clear()
        self.last_stage = "NORMAL"
        self._record_event("SUCCESS", "SYSTEM RESET", "Spacecraft digital twin state restored to nominal factory baseline.", "SYSTEM")
        self._process_tick(0.0)

    def trigger_scenario(self, scenario_name: str) -> Dict[str, Any]:
        scenario_clean = scenario_name.strip().upper()
        self.scenario_mgr.set_scenario(scenario_clean, self.physics.time_seconds)
        
        subsystem_map = {
            "NORMAL": "SYSTEM",
            "THERMAL_DEGRADATION": "Thermal",
            "BATTERY_FAILURE": "Battery",
            "COMMUNICATION_FAILURE": "Communication",
            "PROPULSION_ANOMALY": "Propulsion",
            "SOLAR_POWER_DROP": "Power"
        }
        sub = subsystem_map.get(scenario_clean, "SYSTEM")
        sev = "INFO" if scenario_clean == "NORMAL" else "WARNING"
        self._record_event(sev, f"SCENARIO INJECTED: {scenario_clean}", f"Operator triggered {scenario_clean} degradation sequence.", sub)
        return {"status": "SUCCESS", "active_scenario": scenario_clean}

    def _process_tick(self, dt: float) -> Dict[str, Any]:
        """Synchronously step simulation, execute ML pipelines, and update state buffers."""
        self.mission_elapsed_seconds += int(dt)
        
        # 1. Physics Engine Step
        overrides, stage, progress = self.scenario_mgr.get_physics_overrides(dt)
        raw_telemetry = self.physics.step(dt=dt, overrides=overrides)
        
        # 2. Stage Escalation Logging
        if stage != self.last_stage:
            if stage in ["WARNING", "DEGRADATION"]:
                self._record_event("WARNING", f"ESCALATION: {stage}", f"Telemetry progression reached {stage} phase (progress {int(progress*100)}%).", "SYSTEM")
            elif stage in ["CRITICAL", "FAILURE IMMINENT"]:
                self._record_event("CRITICAL", f"CRITICAL HAZARD: {stage}", f"Subsystem failure sequence reached critical threshold.", "SYSTEM")
            elif stage == "NORMAL":
                self._record_event("SUCCESS", "NOMINAL STABILIZATION", "Telemetry normalized back within nominal bounds.", "SYSTEM")
            self.last_stage = stage

        # 3. ML Anomaly Detection (Isolation Forest)
        anomaly_res = anomaly_detector.predict(raw_telemetry)
        raw_telemetry["anomaly_score"] = anomaly_res["anomaly_score"]
        raw_telemetry["anomaly_detected"] = anomaly_res["anomaly_detected"]
        raw_telemetry["anomaly_severity"] = anomaly_res["severity"]
        
        # Manage active anomaly list
        if anomaly_res["anomaly_detected"]:
            if not any(a["subsystem"] == anomaly_res["subsystem"] for a in self.active_anomalies):
                new_anomaly = {
                    "id": len(self.active_anomalies) + 1,
                    "timestamp": datetime.utcnow().isoformat(),
                    "subsystem": anomaly_res["subsystem"],
                    "channel": f"T-1" if anomaly_res["subsystem"] == "Thermal" else ("P-1" if anomaly_res["subsystem"] == "Power" else "S-1"),
                    "score": anomaly_res["anomaly_score"],
                    "severity": anomaly_res["severity"],
                    "affected_parameters": [f["parameter"] for f in anomaly_res["deviant_features"]],
                    "description": f"Significant variance detected in {anomaly_res['subsystem']} telemetry (Isolation Forest score {int(anomaly_res['anomaly_score']*100)}%).",
                    "is_nasa_benchmark": True,
                    "status": "ACTIVE"
                }
                self.active_anomalies.insert(0, new_anomaly)
                self._record_event(anomaly_res["severity"], f"ANOMALY DETECTED: {anomaly_res['subsystem']}", new_anomaly["description"], anomaly_res["subsystem"])
        else:
            if self.active_anomalies and self.scenario_mgr.active_scenario == "NORMAL":
                self.active_anomalies.clear()

        # 4. ML Failure Prediction (XGBoost)
        history_list = list(self.history)
        self.latest_predictions = failure_predictor.predict_all(raw_telemetry, history_list)
        
        highest_p = 0.0
        highest_sub = "NONE"
        for p in self.latest_predictions:
            if p["failure_probability"] > highest_p:
                highest_p = p["failure_probability"]
                highest_sub = p["subsystem"]
                
        raw_telemetry["highest_failure_risk"] = highest_sub
        raw_telemetry["max_failure_prob"] = highest_p

        # 5. RUL Estimation
        self.latest_rul = rul_estimator.estimate_components(raw_telemetry)
        raw_telemetry["rul_hours"] = self.latest_rul.get("components", {}).get("Battery", {}).get("estimated_rul_hours", 480.0)
        
        # Metadata
        raw_telemetry["timestamp"] = datetime.utcnow().isoformat()
        raw_telemetry["operating_mode"] = self.scenario_mgr.active_scenario
        raw_telemetry["degradation_stage"] = stage
        raw_telemetry["degradation_progress"] = round(progress, 2)
        raw_telemetry["mission_elapsed_seconds"] = self.mission_elapsed_seconds
        
        # Save to buffer
        self.history.append(raw_telemetry)
        return raw_telemetry

    async def tick(self):
        """Advances simulation by 1 step, executes ML models, and broadcasts telemetry."""
        if not self.is_running:
            return

        dt = 1.0 * self.speed_multiplier
        raw_telemetry = self._process_tick(dt)

        # Broadcast over WebSocket if active connections exist
        if connection_manager.active_connections:
            packet = {
                "type": "TELEMETRY_UPDATE",
                "telemetry": raw_telemetry,
                "anomalies": self.active_anomalies[:5],
                "predictions": self.latest_predictions,
                "rul": self.latest_rul,
                "latest_event": self.timeline_events[0] if self.timeline_events else None
            }
            await connection_manager.broadcast(packet)

    def get_latest_state(self) -> Dict[str, Any]:
        """Returns the full unified digital twin state, advancing on-demand if serverless."""
        if not self.history:
            self._process_tick(1.0)
            
        # In serverless execution where no background loop runs, advance step if sufficient time elapsed
        if not self.is_loop_running and self.is_running:
            now = datetime.utcnow()
            dt_elapsed = (now - self._last_tick_time).total_seconds()
            step_threshold = max(0.2, 1.0 / self.speed_multiplier)
            if dt_elapsed >= step_threshold:
                self._process_tick(min(5.0, dt_elapsed * self.speed_multiplier))
                self._last_tick_time = now
            
        latest = copy.deepcopy(self.history[-1])
        latest["anomalies"] = self.active_anomalies
        latest["predictions"] = self.latest_predictions
        latest["rul"] = self.latest_rul
        latest["recent_events"] = self.timeline_events[:10]
        return latest

telemetry_service = TelemetryService()
