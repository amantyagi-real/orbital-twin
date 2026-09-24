import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings
from backend.app.api.routes import api_router
from backend.app.websocket.connection_manager import connection_manager
from backend.app.services.telemetry_service import telemetry_service
from backend.app.database.connection import Base, engine

# Initialize database schema safely
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"Database schema initialization warning: {e}")

simulation_task = None

async def simulation_loop():
    """Continuous background loop stepping simulation and broadcasting telemetry."""
    print("ORBITAL TWIN: Continuous telemetry streaming loop started.")
    telemetry_service.is_loop_running = True
    try:
        while True:
            try:
                await telemetry_service.tick()
            except Exception as e:
                print(f"Error in telemetry loop: {e}")
            # Sleep 1.0s adjusted for speed multiplier
            sleep_time = max(0.1, 1.0 / telemetry_service.speed_multiplier)
            await asyncio.sleep(sleep_time)
    finally:
        telemetry_service.is_loop_running = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global simulation_task
    simulation_task = asyncio.create_task(simulation_loop())
    yield
    # Shutdown
    if simulation_task:
        simulation_task.cancel()
        try:
            await simulation_task
        except asyncio.CancelledError:
            pass
    print("ORBITAL TWIN: Telemetry loop stopped.")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-Powered Spacecraft Digital Twin & Mission Intelligence Platform",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root service status endpoint
@app.get("/")
def get_service_root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "ONLINE",
        "description": "AI-Powered Spacecraft Digital Twin & Mission Intelligence Platform",
        "docs": "/docs",
        "health": "/api/health"
    }

# Mount REST routes:
# 1. Under '/api' for standard client calls and reverse proxies
# 2. At root for serverless/servlet gateways where '/api' path may be stripped
app.include_router(api_router, prefix="/api")
app.include_router(api_router)

# Mount WebSocket endpoint
@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await connection_manager.connect(websocket)
    try:
        # Immediately send current state on connect
        latest = telemetry_service.get_latest_state()
        await websocket.send_json({
            "type": "INITIAL_STATE",
            "telemetry": latest,
            "anomalies": telemetry_service.active_anomalies[:5],
            "predictions": telemetry_service.latest_predictions,
            "rul": telemetry_service.latest_rul,
            "events": telemetry_service.timeline_events[:15]
        })
        while True:
            # Keep connection alive & handle incoming client messages (e.g. ping)
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception:
        connection_manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.HOST, port=settings.PORT, reload=False)
