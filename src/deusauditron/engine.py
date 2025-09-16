import asyncio
from typing import Optional

from deusauditron.lock.base import BaseLockManager
from deusauditron.app_logging.context import set_logging_context
from deusauditron.app_logging.logger import logger
from deusauditron.qbackends.base import BaseQueueBackend
from deusauditron.state.manager import StateManager
from deusauditron.state.object.eval import EvalState
from deusauditron.schemas.autogen.references.evaluation_result_schema import Status
from deusauditron.schemas.shared_models.models import AgentEvalRequest

from deusauditron.eval.eval_worker import LLMEvaluator


class Auditron:
    """Minimal evaluation engine for Deusauditron.

    Processes `AgentEvalRequest` items from the queue and updates `EvalState`.
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Auditron, cls).__new__(cls)
        return cls._instance

    def __init__(self, queue_backend: BaseQueueBackend, lock_manager: BaseLockManager):
        if Auditron._initialized:
            return
        self.queue_backend = queue_backend
        self.lock_manager = lock_manager
        self._eval_consumer_task: Optional[asyncio.Task] = None
        self._running = False
        Auditron._initialized = True

    @classmethod
    def get_instance(cls) -> "Auditron":
        if cls._instance is None:
            raise RuntimeError("Auditron instance has not been initialized.")
        return cls._instance

    def start(self):
        if not self._running:
            self._running = True
            self._eval_consumer_task = asyncio.create_task(self._consume_eval_requests())

    def stop(self):
        if self._running:
            if self._eval_consumer_task:
                self._eval_consumer_task.cancel()
            self._running = False

    async def wait_for_ready(self):
        if not self._running:
            raise RuntimeError("Auditron is not running. Call start() first.")
        await asyncio.sleep(0.1)
        if not self._eval_consumer_task or self._eval_consumer_task.cancelled():
            raise RuntimeError("Evaluation consumer failed to start or was cancelled")
        logger.info("Deusauditron engine is ready and operational")

    async def submit_eval_request(self, request: AgentEvalRequest, eval_state: EvalState):
        """Enqueue an evaluation request and set initial eval state."""
        if not self._running:
            raise RuntimeError("Auditron is not running. Call start().")
        ckey = request.composite_key
        await self.lock_manager.wait_for_lock(ckey)
        try:
            await self.queue_backend.enqueue_eval_request(request)
            await StateManager().set_eval_state(
                request.tenant_id, request.agent_id, request.run_id, eval_state
            )
        finally:
            await self.lock_manager.release_lock(ckey)

    async def delete_eval_request(self, request: AgentEvalRequest):
        """Delete evaluation state (idempotent)."""
        if not self._running:
            raise RuntimeError("Auditron is not running. Call start().")
        ckey = request.composite_key
        await self.lock_manager.wait_for_lock(ckey)
        try:
            await StateManager().delete_eval_state(
                request.tenant_id, request.agent_id, request.run_id
            )
        finally:
            await self.lock_manager.release_lock(ckey)

    async def _consume_eval_requests(self):
        logger.info("## Deusauditron: starting eval request consumer loop")
        while True:
            try:
                request = await self.queue_backend.dequeue_eval_request()
                if not request:
                    continue
                asyncio.create_task(self._handle_eval_request(request))
            except asyncio.CancelledError:
                logger.info("## Deusauditron: eval request consumer loop cancelled")
                break
            except Exception as ex:
                logger.exception(f"## ERROR in Deusauditron _consume_eval_requests: {ex}")

    async def _handle_eval_request(self, request: AgentEvalRequest):
        set_logging_context(request.tenant_id, request.agent_id, request.run_id)
        logger.info(f"## Acquiring lock for {request.composite_key}")
        await self.lock_manager.wait_for_lock(request.composite_key)
        logger.info(
            f"## Handling evaluation request {request.composite_key} in background"
        )
        try:
            evaluator = LLMEvaluator(request)
            await evaluator.evaluate()
            logger.info(f"## Deusauditron: evaluation completed {request.composite_key}")
        except Exception as ex:
            logger.error(f"[Deusauditron ERROR] in _handle_eval_request: {ex}")
            try:
                eval_state = await StateManager().get_eval_state(
                    request.tenant_id, request.agent_id, request.run_id
                )
                if eval_state and eval_state.evaluation_result:
                    eval_state.evaluation_result.status = Status.Error
                    await StateManager().set_eval_state(
                        request.tenant_id, request.agent_id, request.run_id, eval_state
                    )
            except Exception as state_ex:
                logger.error(f"Deusauditron: failed to update error state: {state_ex}")
        finally:
            logger.info(f"## Releasing lock for {request.composite_key}")
            await self.lock_manager.release_lock(request.composite_key)

