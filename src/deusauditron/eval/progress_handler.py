import asyncio

from deusauditron.app_logging.logger import logger

from .eval_state_manager import EvalStateManager
from .progress_tracker import EvaluationProgressTracker, MilestoneType


class EvaluationProgressHandler:
    def __init__(self, eval_state_manager: EvalStateManager, auto_refine_enabled: bool = False):
        self.eval_state_manager = eval_state_manager
        self.progress_tracker = EvaluationProgressTracker(auto_refine_enabled)
        self._update_lock = asyncio.Lock()
        logger.info(f"Initialized progress handler with auto_refine={auto_refine_enabled}")

    async def mark_milestone_complete(self, milestone_type: MilestoneType) -> None:
        async with self._update_lock:
            try:
                new_progress = await self.progress_tracker.mark_milestone_complete(milestone_type)
                await self.eval_state_manager.update_progress(new_progress)
                logger.info(f"Progress updated to {new_progress:.1f}% after completing {milestone_type.value}")
            except Exception as e:
                logger.error(f"Failed to update progress for milestone {milestone_type.value}: {e}")

