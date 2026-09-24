import os
import joblib
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from backend.app.config import settings
from backend.app.ml.nasa_loader import nasa_loader

class AnomalyDetector:
    """
    ML Anomaly Detection Pipeline powered by Scikit-Learn Isolation Forest.
    Trained on NASA SMAP/MSL benchmark telemetry and synthetic multi-channel baselines.
    """
    FEATURE_NAMES = [
        "temperature",
        "battery_voltage",
        "battery_current",
        "power_consumption",
        "solar_power",
        "fuel_pressure",
        "thruster_pressure",
        "communication_signal",
        "packet_loss",
        "cpu_temp"
    ]
    
    # Nominal baseline statistics (mean, std) for feature normalization and deviation calculation
    NOMINAL_BASELINE = {
        "temperature": (24.0, 3.5),
        "battery_voltage": (28.2, 0.4),
        "battery_current": (4.2, 1.2),
        "power_consumption": (820.0, 45.0),
        "solar_power": (1450.0, 120.0),
        "fuel_pressure": (220.0, 6.0),
        "thruster_pressure": (18.5, 0.8),
        "communication_signal": (94.0, 3.0),
        "packet_loss": (0.05, 0.08),
        "cpu_temp": (42.0, 3.2)
    }

    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or os.path.join(settings.MODELS_DIR, "anomaly")
        try:
            os.makedirs(self.model_dir, exist_ok=True)
        except OSError:
            pass
        self.model_path = os.path.join(self.model_dir, "isolation_forest.joblib")
        self.scaler_path = os.path.join(self.model_dir, "scaler.joblib")
        
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_ready = False
        
        self._load_or_train()

    def _extract_feature_vector(self, telemetry: Dict[str, Any]) -> np.ndarray:
        return np.array([float(telemetry.get(name, self.NOMINAL_BASELINE[name][0])) for name in self.FEATURE_NAMES])

    def _load_or_train(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.is_ready = True
                print("AnomalyDetector: Loaded pre-trained Isolation Forest model from disk.")
                return
            except Exception as e:
                print(f"AnomalyDetector: Failed to load existing model ({e}). Re-training...")
        
        self.train_and_save()

    def train_and_save(self) -> Dict[str, Any]:
        """
        Trains Isolation Forest using combined NASA SMAP/MSL benchmark sequences
        and simulated nominal telemetry states.
        """
        print("AnomalyDetector: Training Isolation Forest on NASA + Spacecraft telemetry...")
        
        # 1. Synthesize 5000 nominal state vectors with realistic variance
        np.random.seed(42)
        n_samples = 4000
        data = []
        for _ in range(n_samples):
            vec = [
                np.random.normal(m, s) for _, (m, s) in self.NOMINAL_BASELINE.items()
            ]
            data.append(vec)
            
        # 2. Integrate NASA training channels if available
        nasa_matrix = nasa_loader.get_multi_channel_training_matrix()
        if nasa_matrix.shape[1] >= 5:
            # Map normalized NASA P-1, T-1, A-1, E-1, S-1 variations
            nasa_scaled = (nasa_matrix - np.mean(nasa_matrix, axis=0)) / (np.std(nasa_matrix, axis=0) + 1e-6)
            for i in range(min(1500, len(nasa_scaled))):
                row = data[i % len(data)].copy()
                # Modulate power and thermal features with real NASA variations
                row[0] += float(nasa_scaled[i, 1] * 1.5)  # Thermal variation from T-1
                row[3] += float(nasa_scaled[i, 0] * 20.0) # Power variation from P-1
                data.append(row)
                
        X = np.array(data)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Isolation Forest with contamination=0.03
        self.model = IsolationForest(
            n_estimators=120,
            contamination=0.03,
            max_samples="auto",
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled)
        
        # Serialize
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            print(f"AnomalyDetector: Successfully trained and saved model to {self.model_path}")
        except OSError as e:
            print(f"AnomalyDetector: Warning - could not write model to disk ({e}). Keeping model in-memory.")
        self.is_ready = True
        return {
            "status": "ONLINE",
            "samples": len(X),
            "features": len(self.FEATURE_NAMES),
            "algorithm": "Isolation Forest",
            "model_path": self.model_path
        }

    def predict(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates real-time anomaly score [0.0 - 1.0], severity level,
        and identifies deviant features.
        """
        if not self.is_ready or self.model is None:
            return {"anomaly_score": 0.05, "anomaly_detected": False, "severity": "NOMINAL", "deviant_features": []}
            
        vec = self._extract_feature_vector(telemetry).reshape(1, -1)
        vec_scaled = self.scaler.transform(vec)
        
        # Isolation Forest decision_function: lower score = more anomalous
        # Typically in range [-0.5, 0.5]. Normal is ~ +0.15, extreme anomaly is ~ -0.3
        raw_score = self.model.decision_function(vec_scaled)[0]
        
        # Normalize into probability-like score: 0.0 = completely normal, 1.0 = extreme anomaly
        # raw_score 0.15 -> 0.05; raw_score -0.20 -> 0.95
        anomaly_score = 1.0 / (1.0 + np.exp((raw_score + 0.02) * 14.0))
        anomaly_score = round(float(np.clip(anomaly_score, 0.02, 0.99)), 3)
        
        # Detect deviant features via Z-score
        deviant_features = []
        for name in self.FEATURE_NAMES:
            val = float(telemetry.get(name, self.NOMINAL_BASELINE[name][0]))
            m, s = self.NOMINAL_BASELINE[name]
            z = abs((val - m) / s)
            if z > 2.2:
                pct_delta = round(((val - m) / m) * 100.0, 1)
                deviant_features.append({
                    "parameter": name,
                    "current_value": round(val, 2),
                    "baseline_mean": round(m, 2),
                    "z_score": round(z, 2),
                    "percentage_delta": pct_delta
                })
                
        # Classify severity
        if anomaly_score >= 0.78 or len(deviant_features) >= 3:
            severity = "CRITICAL"
            anomaly_detected = True
        elif anomaly_score >= 0.52 or len(deviant_features) >= 1:
            severity = "WARNING"
            anomaly_detected = True
        else:
            severity = "NOMINAL"
            anomaly_detected = False
            
        # Determine primary subsystem
        subsystem = "SYSTEM"
        if deviant_features:
            param = deviant_features[0]["parameter"]
            if "temp" in param:
                subsystem = "Thermal"
            elif "battery" in param:
                subsystem = "Battery"
            elif "power" in param:
                subsystem = "Power"
            elif "pressure" in param or "fuel" in param:
                subsystem = "Propulsion"
            elif "signal" in param or "packet" in param:
                subsystem = "Communication"
            elif "cpu" in param:
                subsystem = "Computing"
                
        return {
            "anomaly_score": anomaly_score,
            "anomaly_detected": anomaly_detected,
            "severity": severity,
            "subsystem": subsystem,
            "deviant_features": deviant_features
        }

anomaly_detector = AnomalyDetector()
