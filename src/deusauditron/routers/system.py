from deusauditron.engine import Auditron
from fastapi import APIRouter

system_router = APIRouter(
    prefix="/system",
    tags=["system"],
    responses={404: {"description": "Not found"}},
)


@system_router.get("/health")
async def health_check():
    """Simple health endpoint."""
    try:
        engine = Auditron.get_instance()
        return {
            "status": "ok",
            "engine_running": engine._running if engine else False,
            "service_name": "Deusauditron"
        }
    except RuntimeError:
        return {
            "status": "error",
            "engine_running": False,
            "message": "Engine not initialized",
        }
