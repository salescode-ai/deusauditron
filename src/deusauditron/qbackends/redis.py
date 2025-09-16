from typing import Optional, Tuple
import json

from loguru import logger
from redis.asyncio import Redis

from deusauditron.qbackends.base import BaseQueueBackend
from deusauditron.schemas.shared_models.models import AgentEvalRequest, AgentRunRequest


RUN_REQUEST_QUEUE_NAME = "globalq_agent_run_requests"
EVAL_REQUEST_QUEUE_NAME = "globalq_agent_eval_requests"


class RedisQueueBackend(BaseQueueBackend):
    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url
        self.redis: Optional[Redis] = None

    async def _connect_if_needed(self):
        if not self.redis:
            try:
                self.redis = Redis.from_url(self.redis_url)
                logger.debug(f"## Successfully connected to Redis at {self.redis_url}")
            except Exception as e:
                logger.error(f"## Failed to connect to Redis: {e}")
                raise

    async def connect(self):
        await self._connect_if_needed()

    async def enqueue_run_request(self, request: AgentRunRequest) -> None:
        await self._connect_if_needed()
        if not self.redis:
            raise RuntimeError("Redis not connected")
        if request.audio_data:
            audio_key = f"audio:{request.internal_request_id}"
            await self.redis.set(audio_key, request.audio_data, ex=60)  # 1 min TTL
            request_dict = request.model_dump()
            request_dict["_audio_key"] = audio_key
            json_data = json.dumps(request_dict, default=str)
        else:
            json_data = request.model_dump_json()
        await self.redis.rpush(RUN_REQUEST_QUEUE_NAME, json_data)  # type: ignore

    async def dequeue_run_request(self) -> Optional[AgentRunRequest]:
        await self._connect_if_needed()
        if not self.redis:
            raise RuntimeError("Redis not connected")
        data: Optional[Tuple[bytes, bytes]] = await self.redis.blpop([RUN_REQUEST_QUEUE_NAME])  # type: ignore
        if not data:
            return None
        _, json_data = data
        request_dict = json.loads(json_data.decode("utf-8"))
        audio_key = request_dict.pop("_audio_key", None)
        audio_data = None
        if audio_key:
            audio_data = await self.redis.get(audio_key)
            await self.redis.delete(audio_key)
        request = AgentRunRequest.model_validate(request_dict)
        if audio_data:
            request.audio_data = audio_data
        return request

    async def enqueue_eval_request(self, request: AgentEvalRequest) -> None:
        await self._connect_if_needed()
        if not self.redis:
            raise RuntimeError("Redis not connected")
        json_data = request.model_dump_json()
        await self.redis.rpush(EVAL_REQUEST_QUEUE_NAME, json_data)  # type: ignore

    async def dequeue_eval_request(self) -> Optional[AgentEvalRequest]:
        await self._connect_if_needed()
        if not self.redis:
            raise RuntimeError("Redis not connected")
        data: Optional[Tuple[bytes, bytes]] = await self.redis.blpop([EVAL_REQUEST_QUEUE_NAME])  # type: ignore
        if not data:
            return None
        _, json_data = data
        return AgentEvalRequest.model_validate_json(json_data.decode("utf-8"))

    async def close(self) -> None:
        if self.redis:
            await self.redis.close()
            self.redis = None
            logger.debug("Closed Redis connection")

