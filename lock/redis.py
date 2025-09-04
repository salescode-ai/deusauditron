import asyncio

from lock.base import BaseLockManager
from app_logging.logger import logger


class RedisLockManager(BaseLockManager):
    def __init__(self, redis_client, lock_expiry_sec: int = 30):
        self.redis = redis_client
        self.lock_expiry_sec = lock_expiry_sec

    async def acquire_lock(self, key: str) -> bool:
        result = await self.redis.set(name=key, value="1", nx=True, ex=self.lock_expiry_sec)
        return bool(result)

    async def release_lock(self, key: str) -> None:
        await self.redis.delete(key)

    async def wait_for_lock(self, key: str) -> None:
        while True:
            got_it = await self.acquire_lock(key)
            if got_it:
                return
            await asyncio.sleep(0.05)

    async def close(self) -> None:
        logger.debug("Redis lock manager closed")

