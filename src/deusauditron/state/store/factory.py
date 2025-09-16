from deusauditron.config import get_config
from deusauditron.app_logging.logger import logger
from deusauditron.state.store.base import BaseStateStore
from deusauditron.state.store.local import LocalStateStore
from deusauditron.state.store.redis import RedisStateStore


def get_state_store() -> BaseStateStore:
    config = get_config()
    if config.redis.use_redis:
        logger.info("Using redis state store (Deusauditron)")
        if not config.redis.redis_url:
            raise ValueError("REDIS_URL is required for redis store")
        return RedisStateStore(redis_url=config.redis.redis_url)
    return LocalStateStore()

