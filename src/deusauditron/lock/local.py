import asyncio

from deusauditron.lock.base import BaseLockManager
from deusauditron.app_logging.logger import logger


class LocalLockManager(BaseLockManager):
    def __init__(self):
        self._locks = {}

    async def acquire_lock(self, key: str) -> bool:
        lock = self._locks.setdefault(key, asyncio.Lock())
        if lock.locked():
            return False
        await lock.acquire()
        return True

    async def release_lock(self, key: str) -> None:
        lock = self._locks.get(key)
        if lock and lock.locked():
            lock.release()

    async def wait_for_lock(self, key: str) -> None:
        while True:
            got_it = await self.acquire_lock(key)
            if got_it:
                return
            await asyncio.sleep(0.05)

    async def close(self) -> None:
        for key, lock in self._locks.items():
            if lock.locked():
                lock.release()
                logger.debug(f"Released lock for key: {key}")
        self._locks.clear()
        logger.debug("Local lock manager closed")

