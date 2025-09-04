import asyncio
from datetime import datetime, timezone
from typing import Callable, List, Optional

from app_logging.logger import logger
from state.manager import StateManager
from schemas.autogen.references.auto_refine_schema import AutoRefine
from schemas.autogen.references.evaluation_result_schema import EvaluationResult, Status
from schemas.autogen.references.granular_evaluation_result_schema import (
    GranularEvaluationResults,
)


class EvalStateManager:
    def __init__(self, tenant_id: str, agent_id: str, run_id: str):
        self.tenant_id = tenant_id
        self.agent_id = agent_id
        self.run_id = run_id
        self._update_lock = asyncio.Lock()

    async def update_eval_state_atomic(self, update_func: Callable, *args, **kwargs) -> None:
        async with self._update_lock:
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    eval_state = await StateManager().get_eval_state(self.tenant_id, self.agent_id, self.run_id)
                    if not eval_state or not eval_state.get_evaluation_result():
                        logger.warning("Eval state or result not found during atomic update")
                        return
                    eval_result = eval_state.get_evaluation_result()
                    update_func(eval_result, *args, **kwargs)
                    await StateManager().set_eval_state(self.tenant_id, self.agent_id, self.run_id, eval_state)
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error(f"Failed to update eval state after {max_retries} retries: {e}")
                        raise
                    logger.warning(f"Retry {retry_count}/{max_retries} for eval state update: {e}")
                    await asyncio.sleep(0.1 * retry_count)

    def update_eval_result_fields(
        self,
        eval_result: EvaluationResult,
        progress: Optional[float] = None,
        status: Optional[Status] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        evaluated_nodes: Optional[List[str]] = None,
    ) -> None:
        if progress is not None:
            eval_result.progress = progress
        if status is not None:
            eval_result.status = status
        if start_time is not None:
            eval_result.start_time = start_time
        if end_time is not None:
            eval_result.end_time = end_time
        if evaluated_nodes is not None:
            eval_result.evaluated_nodes = evaluated_nodes

    def update_eval_result_evaluations(
        self, eval_result: EvaluationResult, field_name: str, evaluations: List[GranularEvaluationResults]
    ) -> None:
        setattr(eval_result, field_name, evaluations)

    def update_eval_result_auto_refinements(self, eval_result: EvaluationResult, auto_refinements: List[AutoRefine]) -> None:
        eval_result.auto_refinements = auto_refinements

    async def update_eval_state(
        self,
        progress: float,
        status: Status,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        evaluated_nodes: Optional[List[str]] = None,
    ) -> None:
        await self.update_eval_state_atomic(
            self.update_eval_result_fields,
            progress=progress,
            status=status,
            start_time=start_time,
            end_time=end_time,
            evaluated_nodes=evaluated_nodes,
        )

    async def update_evaluations(self, field_name: str, evaluations: List[GranularEvaluationResults]) -> None:
        await self.update_eval_state_atomic(self.update_eval_result_evaluations, field_name, evaluations)

    async def update_auto_refinements(self, auto_refinements: List[AutoRefine]) -> None:
        await self.update_eval_state_atomic(self.update_eval_result_auto_refinements, auto_refinements)

    async def update_final_state(self, evaluated_nodes: Optional[List[str]] = None) -> None:
        end_time = datetime.now(timezone.utc)
        await self.update_eval_state(100.0, Status.Completed, end_time=end_time, evaluated_nodes=evaluated_nodes)

    async def update_error_state(self, evaluated_nodes: Optional[List[str]] = None) -> None:
        end_time = datetime.now(timezone.utc)
        await self.update_eval_state(0.0, Status.Error, end_time=end_time, evaluated_nodes=evaluated_nodes)

    async def update_start_state(self, start_time: Optional[datetime] = None) -> None:
        if start_time is None:
            start_time = datetime.now(timezone.utc)
        await self.update_eval_state(0.0, Status.In_Progress, start_time=start_time)

    async def update_progress(self, progress: float) -> None:
        await self.update_eval_state_atomic(self.update_eval_result_fields, progress=progress)

