import litellm
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from deusauditron.config import get_config, Config, TracingManager
from deusauditron.llm_abstraction.litellm_adapter import LiteLLMAdapter
from deusauditron.llm_abstraction.llm_helper import LLMInvoker
from deusauditron.lock.local import LocalLockManager
from deusauditron.lock.redis import RedisLockManager
from deusauditron.qbackends.local import LocalQueueBackend
from deusauditron.qbackends.redis import RedisQueueBackend
from deusauditron.app_logging.logger import logger
from deusauditron.util.handler import get_authorization

from deusauditron.engine import Auditron
from deusauditron.routers.evaluation import evaluation_router
from deusauditron.routers.phoenix import phoenix_router
from deusauditron.routers.system import system_router
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the evaluation engine on startup and stop it on shutdown."""
    config = get_config()
    try:
        if config.redis.use_redis:
            q_backend = RedisQueueBackend(config.redis.redis_url)
            await q_backend.connect()
            lock_mgr = RedisLockManager(q_backend.redis)
            logger.info(
                f"## Starting Deusauditron with Redis configuration: {config.redis.redis_url}"
            )
        else:
            logger.info("## Starting Deusauditron with local in-memory configuration")
            q_backend = LocalQueueBackend()
            lock_mgr = LocalLockManager()

        # Initialize LLM interface singleton (same abstraction as DeusMachina)
        logger.info("Using LiteLLM for LLM abstraction (Deusauditron).")
        LLMInvoker(llm_adapter=LiteLLMAdapter())

        # Initialize tracing if enabled
        if config.tracing.enabled:
            TracingManager().get_tracer()
            litellm.callbacks = ["arize"]
            logger.info("Tracing initialized with Phoenix")

        engine = Auditron(queue_backend=q_backend, lock_manager=lock_mgr)
        engine.start()
        await engine.wait_for_ready()
        yield
    except Exception as e:
        logger.error(f"## Failed to initialize Deusauditron: {e}")
        raise
    finally:
        try:
            engine = Auditron.get_instance()
            engine.stop()
        except RuntimeError:
            pass


def create_app(config: Optional[Config] = None) -> FastAPI:
    if config is None:
        config = get_config()

    app = FastAPI(
        title="Deusauditron API",
        description="Dedicated evaluation service (mirrors DeusMachina evaluation).",
        version="1.0.1",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors.allowed_origins,
        allow_credentials=config.cors.allow_credentials,
        allow_methods=config.cors.allow_methods,
        allow_headers=config.cors.allow_headers,
    )

    prefix = "/api/v1"
    internal_prefix = "/internal/api/v1"
    routers = [
        evaluation_router,
        phoenix_router,
        system_router,
    ]
    for router in routers:
        app.include_router(
            router, 
            prefix=prefix, 
            dependencies=[Depends(get_authorization)], 
            tags=["External"]
        )
        app.include_router(router, prefix=internal_prefix, tags=["Internal"])

    return app


app = create_app()

# Run with: uvicorn app:app --host 0.0.0.0 --port 8081

