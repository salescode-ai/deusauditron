import asyncio
from typing import Optional

from deusauditron.qbackends.base import BaseQueueBackend
from deusauditron.schemas.shared_models.models import AgentEvalRequest, AgentRunRequest
from loguru import logger


class LocalQueueBackend(BaseQueueBackend):
    def __init__(self) -> None:
        self._run_queue = asyncio.Queue()
        self._eval_queue = asyncio.Queue()

    async def enqueue_run_request(self, request: AgentRunRequest) -> None:
        await self._run_queue.put(request)

    async def dequeue_run_request(self) -> Optional[AgentRunRequest]:
        try:
            return await self._run_queue.get()
        except asyncio.CancelledError:
            return None

    async def enqueue_eval_request(self, request: AgentEvalRequest) -> None:
        await self._eval_queue.put(request)

    async def dequeue_eval_request(self) -> Optional[AgentEvalRequest]:
        try:
            return await self._eval_queue.get()
        except asyncio.CancelledError:
            return None

    async def close(self) -> None:
        while not self._run_queue.empty():
            try:
                self._run_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        while not self._eval_queue.empty():
            try:
                self._eval_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.debug("Cleared local queues")

