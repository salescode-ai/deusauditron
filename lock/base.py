from abc import ABC, abstractmethod


class BaseLockManager(ABC):
    @abstractmethod
    async def acquire_lock(self, key: str) -> bool:
        pass

    @abstractmethod
    async def release_lock(self, key: str) -> None:
        pass

    @abstractmethod
    async def wait_for_lock(self, key: str) -> None:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

