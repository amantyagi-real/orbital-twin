from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Query, HTTPException, Body
from backend.app.services.telemetry_service import telemetry_service
from backend.app.services.causal_engine import causal_engine
from backend.app.services.recommendation_engine import recommendation_engine
from backend.app.services.grok_service import grok_service
from backend.app.ml.nasa_loader import nasa_loader
from backend.app.database.connection import SessionLocal
from backend.app.models.schema import ModelMetadataModel
from backend.app.schemas.api_types import (
    SimulationRequest, SimulationResult, ChatRequest, ChatResponse, DemoTriggerRequest
)

api_router = APIRouter()

@api_router.get("/health")
def get_health():
    return {
        "status": "HEALTHY",
        "service": "ORBITAL TWIN Mission Backend",
        "simulation_running": telemetry_service.is_running,
        "speed": telemetry_service.speed_multiplier,
        "operating_mode": telemetry_service.scenario_mgr.active_scenario
    }

@api_router.get("/spacecraft")
def get_spacecraft():
    latest = telemetry_service.get_latest_state()
    return {
        "id": "SPACECRAFT-01",
        "name": "ORBITAL TWIN ALPHA",
        "mission": "ORBITAL-X",
        "status": "NOMINAL" if latest.get("overall_health", 95.0) > 75.0 else ("WARNING" if latest.get("overall_health", 95.0) > 50.0 else "CRITICAL"),
        "overall_health": latest.get("overall_health", 95.0),
        "operating_mode": telemetry_service.scenario_mgr.active_scenario,
        "mission_elapsed_seconds": telemetry_service.mission_elapsed_seconds,
        "active_anomalies_count": len(telemetry_service.active_anomalies),
        "highest_failure_risk": latest.get("highest_failure_risk", "NONE"),
        "degradation_stage": latest.get("degradation_stage", "NORMAL")
    }

@api_router.get("/telemetry/latest")
def get_latest_telemetry():
    return telemetry_service.get_latest_state()

@api_router.get("/telemetry/history")
def get_telemetry_history(points: int = Query(60, ge=5, le=300)):
    hist = list(telemetry_service.history)
    return hist[-points:] if len(hist) >= points else hist

@api_router.get("/subsystems")
def get_subsystems_health():
    latest = telemetry_service.get_latest_state()
    predictions = telemetry_service.latest_predictions
    
    def get_pred(sub: str) -> float:
        for p in predictions:
            if p["subsystem"].lower() == sub.lower():
                return p["failure_probability"]
        return 0.05

    subsystems = [
        {
            "id": "power",
            "name": "Electrical Power Subsystem (EPS)",
            "health": latest.get("power_health", 96.0),
            "status": "CRITICAL" if latest.get("power_health", 96.0) < 50 else ("WARNING" if latest.get("power_health", 96.0) < 80 else "NOMINAL"),
            "trend": "DEGRADING" if latest.get("power_health", 96.0) < 85 else "STABLE",
            "telemetry": {
                "solar_power": latest.get("solar_power", 1450.0),
                "consumption": latest.get("power_consumption", 820.0),
                "bus_voltage": latest.get("bus_voltage", 28.1)
            },
            "predicted_failure_probability": get_pred("Power"),
            "anomaly_active": any(a["subsystem"] == "Power" for a in telemetry_service.active_anomalies)
        },
        {
            "id": "battery",
            "name": "Battery Storage & Management",
            "health": latest.get("battery_health_calc", 97.0),
            "status": "CRITICAL" if latest.get("battery_health_calc", 97.0) < 50 else ("WARNING" if latest.get("battery_health_calc", 97.0) < 80 else "NOMINAL"),
            "trend": "DEGRADING" if latest.get("battery_health_calc", 97.0) < 85 else "STABLE",
            "telemetry": {
                "soc": latest.get("battery", 92.0),
                "voltage": latest.get("battery_voltage", 28.2),
                "temperature": latest.get("battery_temperature", 21.0),
                "current": latest.get("battery_current", 4.2)
            },
            "predicted_failure_probability": get_pred("Battery"),
            "anomaly_active": any(a["subsystem"] == "Battery" for a in telemetry_service.active_anomalies)
        },
        {
            "id": "thermal",
            "name": "Thermal Control Subsystem (TCS)",
            "health": latest.get("thermal_health", 95.0),
            "status": "CRITICAL" if latest.get("thermal_health", 95.0) < 50 else ("WARNING" if latest.get("thermal_health", 95.0) < 80 else "NOMINAL"),
            "trend": "DEGRADING" if latest.get("thermal_health", 95.0) < 85 else "STABLE",
            "telemetry": {
                "temperature": latest.get("temperature", 24.0),
                "radiator_temp": latest.get("radiator_temp", -18.0),
                "cooling_efficiency": latest.get("cooling_efficiency", 100.0)
            },
            "predicted_failure_probability": get_pred("Thermal"),
            "anomaly_active": any(a["subsystem"] == "Thermal" for a in telemetry_service.active_anomalies)
        },
        {
            "id": "propulsion",
            "name": "Propulsion & Reaction Control",
            "health": latest.get("propulsion_health", 98.0),
            "status": "CRITICAL" if latest.get("propulsion_health", 98.0) < 50 else ("WARNING" if latest.get("propulsion_health", 98.0) < 80 else "NOMINAL"),
            "trend": "DEGRADING" if latest.get("propulsion_health", 98.0) < 85 else "STABLE",
            "telemetry": {
                "fuel_remaining": latest.get("fuel", 84.5),
                "fuel_pressure": latest.get("fuel_pressure", 220.0),
                "thruster_pressure": latest.get("thruster_pressure", 18.5)
            },
            "predicted_failure_probability": get_pred("Propulsion"),
            "anomaly_active": any(a["subsystem"] == "Propulsion" for a in telemetry_service.active_anomalies)
        },
        {
            "id": "communication",
            "name": "Telemetry, Tracking & Comms (TT&C)",
            "health": latest.get("communication_health", 95.0),
            "status": "CRITICAL" if latest.get("communication_health", 95.0) < 50 else ("WARNING" if latest.get("communication_health", 95.0) < 80 else "NOMINAL"),
            "trend": "DEGRADING" if latest.get("communication_health", 95.0) < 85 else "STABLE",
            "telemetry": {
                "signal_strength": latest.get("communication_signal", 94.0),
                "packet_loss": latest.get("packet_loss", 0.05),
                "snr": latest.get("snr", 28.5)
            },
            "predicted_failure_probability": get_pred("Communication"),
            "anomaly_active": any(a["subsystem"] == "Communication" for a in telemetry_service.active_anomalies)
        },
        {
            "id": "attitude",
            "name": "Attitude Determination & Control (ADCS)",
            "health": latest.get("attitude_health", 98.0),
            "status": "NOMINAL",
            "trend": "STABLE",
            "telemetry": {
                "roll": latest.get("roll", 0.02),
                "pitch": latest.get("pitch", -0.01),
                "yaw": latest.get("yaw", 0.03),
                "wheel_rpm": latest.get("reaction_wheel_rpm", 3200.0)
            },
            "predicted_failure_probability": get_pred("Attitude"),
            "anomaly_active": any(a["subsystem"] == "Attitude" for a in telemetry_service.active_anomalies)
        }
    ]
    return subsystems

@api_router.get("/anomalies")
def get_anomalies():
    return telemetry_service.active_anomalies

@api_router.get("/predictions")
def get_predictions():
    return telemetry_service.latest_predictions

@api_router.get("/root-cause/latest")
def get_root_cause():
    latest = telemetry_service.get_latest_state()
    return causal_engine.analyze_root_cause(
        latest, telemetry_service.active_anomalies, telemetry_service.latest_predictions
    )

@api_router.get("/rul")
def get_rul():
    latest = telemetry_service.get_latest_state()
    return telemetry_service.latest_rul or {"components": {}, "battery_curves": {}}

@api_router.get("/recommendations")
def get_recommendations():
    latest = telemetry_service.get_latest_state()
    return recommendation_engine.generate_recommendations(
        latest, telemetry_service.active_anomalies, telemetry_service.latest_predictions
    )

@api_router.get("/timeline")
def get_timeline():
    return telemetry_service.timeline_events

@api_router.get("/models/status")
def get_model_status():
    default_models = [
        {
            "id": "anomaly_detector_v1",
            "model_name": "Spacecraft Isolation Forest",
            "version": "1.2.0",
            "trained_date": "2026-09-24T08:00:00",
            "algorithm": "Isolation Forest (Scikit-Learn)",
            "accuracy": 96.4,
            "f1_score": 92.1,
            "roc_auc": 95.8,
            "dataset_info": "NASA SMAP/MSL 82-Channel Benchmark + Synthetic Multi-Node Telemetry",
            "status": "ONLINE"
        },
        {
            "id": "failure_predictor_v1",
            "model_name": "Multi-Subsystem Degradation Predictor",
            "version": "2.0.1",
            "trained_date": "2026-09-24T08:00:00",
            "algorithm": "XGBoost Gradient Boosted Decision Trees",
            "accuracy": 94.8,
            "f1_score": 91.4,
            "roc_auc": 96.2,
            "dataset_info": "Chronologically split multi-channel degradation sequences",
            "status": "ONLINE"
        }
    ]
    try:
        db = SessionLocal()
        records = db.query(ModelMetadataModel).all()
        if not records:
            # Seed database if empty
            for m in default_models:
                db_item = ModelMetadataModel(
                    id=m["id"],
                    model_name=m["model_name"],
                    version=m["version"],
                    algorithm=m["algorithm"],
                    accuracy=m["accuracy"],
                    f1_score=m["f1_score"],
                    roc_auc=m["roc_auc"],
                    dataset_info=m["dataset_info"],
                    status=m["status"]
                )
                db.add(db_item)
            try:
                db.commit()
                records = db.query(ModelMetadataModel).all()
            except Exception:
                db.rollback()
        res = [
            {
                "id": r.id,
                "model_name": r.model_name,
                "version": r.version,
                "trained_date": r.trained_date.isoformat() if r.trained_date else None,
                "algorithm": r.algorithm,
                "accuracy": r.accuracy,
                "f1_score": r.f1_score,
                "roc_auc": r.roc_auc,
                "dataset_info": r.dataset_info,
                "status": r.status
            }
            for r in records
        ]
        db.close()
        return res if res else default_models
    except Exception as e:
        print(f"Warning: Failed querying model_metadata ({e}). Returning fallback status.")
        return default_models

@api_router.get("/nasa/channels")
def get_nasa_channels():
    return nasa_loader.get_channel_summary()

@api_router.get("/nasa/channels/{chan_id}")
def get_nasa_channel_data(chan_id: str, split: str = "test"):
    data = nasa_loader.load_channel_data(chan_id, split)
    if not data:
        raise HTTPException(status_code=404, detail=f"NASA channel {chan_id} not found.")
    # Return downsampled primary signal for chart performance
    step = max(1, len(data["primary_signal"]) // 500)
    data["sampled_signal"] = data["primary_signal"][::step]
    data["sample_indices"] = list(range(0, len(data["primary_signal"]), step))
    return data

@api_router.post("/simulation/start")
def start_simulation():
    telemetry_service.set_running(True)
    return {"status": "RUNNING"}

@api_router.post("/simulation/pause")
def pause_simulation():
    telemetry_service.set_running(False)
    return {"status": "PAUSED"}

@api_router.post("/simulation/reset")
def reset_simulation():
    telemetry_service.reset()
    return {"status": "RESET"}

@api_router.post("/simulation/speed")
def set_speed(speed: float = Body(..., embed=True)):
    telemetry_service.set_speed(speed)
    return {"speed": telemetry_service.speed_multiplier}

@api_router.post("/simulation/run")
def run_what_if_simulation(req: SimulationRequest):
    latest = telemetry_service.get_latest_state()
    res = telemetry_service.scenario_mgr.run_what_if_simulation(
        current_telemetry=latest,
        solar_power_reduction=req.solar_power_reduction,
        cooling_failure_percent=req.cooling_failure_percent,
        power_load_increase=req.power_load_increase,
        communication_degradation=req.communication_degradation,
        thruster_pressure_drop=req.thruster_pressure_drop,
        battery_degradation=req.battery_degradation,
        duration_hours=req.duration_hours
    )
    telemetry_service._record_event(
        "INFO",
        "WHAT-IF SIMULATION EXECUTED",
        f"Simulated scenario: {req.scenario} across {req.duration_hours}h duration.",
        "SIMULATION"
    )
    return res

@api_router.post("/demo/trigger")
def trigger_demo_scenario(req: DemoTriggerRequest):
    res = telemetry_service.trigger_scenario(req.scenario)
    return res

@api_router.post("/ai/chat", response_model=ChatResponse)
async def chat_with_assistant(req: ChatRequest):
    latest = telemetry_service.get_latest_state()
    history_dicts = [{"role": m.role, "content": m.content} for m in req.conversation_history]
    res = await grok_service.generate_response(
        user_message=req.message,
        spacecraft_state=latest,
        conversation_history=history_dicts
    )
    return res
