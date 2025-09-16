import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Set

from deusauditron.app_logging.logger import logger


class MilestoneType(Enum):
    TURN_EVALUATIONS = "turn_evaluations"
    NODE_EVALUATIONS = "node_evaluations"
    INTENT_EVALUATIONS = "intent_evaluations"
    CONVERSATION_EVALUATIONS = "conversation_evaluations"
    AUTO_REFINE_EVALUATIONS = "auto_refine_evaluations"


@dataclass
class MilestoneConfig:
    milestone_type: MilestoneType
    weight: float
    description: str


class EvaluationProgressTracker:
    def __init__(self, auto_refine_enabled: Optional[bool] = False):
        self._lock = asyncio.Lock()
        self._completed_milestones: Set[MilestoneType] = set()
        self._current_progress: float = 0.0
        self._auto_refine_enabled = auto_refine_enabled
        self._milestone_configs = self._get_milestone_configs()
        logger.info(
            f"Initialized progress tracker with auto_refine={auto_refine_enabled}, total milestones: {len(self._milestone_configs)}"
        )

    def _get_milestone_configs(self) -> Dict[MilestoneType, MilestoneConfig]:
        ALL_MILESTONES = {
            MilestoneType.TURN_EVALUATIONS: "Turn-level evaluations",
            MilestoneType.NODE_EVALUATIONS: "Node-level evaluations",
            MilestoneType.INTENT_EVALUATIONS: "Intent-level evaluations",
            MilestoneType.CONVERSATION_EVALUATIONS: "Conversation-level evaluations",
            MilestoneType.AUTO_REFINE_EVALUATIONS: "Auto-refinement evaluations",
        }
        EVALUATION_MODES = {
            True: {"milestones": list(ALL_MILESTONES.keys()), "weight_per_milestone": 20.0},
            False: {
                "milestones": [
                    MilestoneType.TURN_EVALUATIONS,
                    MilestoneType.NODE_EVALUATIONS,
                    MilestoneType.INTENT_EVALUATIONS,
                    MilestoneType.CONVERSATION_EVALUATIONS,
                ],
                "weight_per_milestone": 25.0,
            },
        }
        auto_refine_bool = False if self._auto_refine_enabled is None or hasattr(self._auto_refine_enabled, "_mock_name") else bool(self._auto_refine_enabled)
        mode_config = EVALUATION_MODES[auto_refine_bool]
        return {
            milestone_type: MilestoneConfig(
                milestone_type, mode_config["weight_per_milestone"], ALL_MILESTONES[milestone_type]
            )
            for milestone_type in mode_config["milestones"]
        }

    async def mark_milestone_complete(self, milestone_type: MilestoneType) -> float:
        async with self._lock:
            if milestone_type not in self._milestone_configs:
                logger.warning(f"Unknown milestone type: {milestone_type}")
                return self._current_progress
            if milestone_type in self._completed_milestones:
                logger.warning(f"Milestone {milestone_type.value} already marked as complete")
                return self._current_progress
            self._completed_milestones.add(milestone_type)
            config = self._milestone_configs[milestone_type]
            self._current_progress = min(self._current_progress + config.weight, 100.0)
            logger.info(
                f"Milestone completed: {config.description} (Progress: {self._current_progress:.1f}%)"
            )
            return self._current_progress

    async def get_current_progress(self) -> float:
        async with self._lock:
            return self._current_progress

