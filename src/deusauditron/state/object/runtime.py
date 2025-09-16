from typing import List, Optional, Sequence

from pydantic import BaseModel, Field, RootModel
from deusauditron.schemas.shared_models.models import InteractionLog, Message


class InteractionLogList(RootModel):
    root: list[InteractionLog] = Field(default_factory=list)
    def unwrap(self) -> list[InteractionLog]:
        return self.root
    @classmethod
    def wrap(cls, interaction_log: Sequence[InteractionLog]) -> "InteractionLogList":
        return cls(root=list(interaction_log))


class MessagesList(RootModel):
    root: list[Message] = Field(default_factory=list)
    def unwrap(self) -> list[Message]:
        return self.root
    @classmethod
    def wrap(cls, messages: Sequence[Message]) -> "MessagesList":
        return cls(root=list(messages))


class PathList(BaseModel):
    root: List[str] = Field(default_factory=list)
    def unwrap(self) -> List[str]:
        return self.root
    @classmethod
    def wrap(cls, path: List[str]) -> "PathList":
        return cls(root=path)


class CurrentNode(BaseModel):
    root: Optional[str] = None
    def unwrap(self) -> Optional[str]:
        return self.root
    @classmethod
    def wrap(cls, node: str) -> "CurrentNode":
        return cls(root=node)

