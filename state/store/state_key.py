from pydantic import BaseModel, Field


class StateKey(BaseModel):
    tenant_id: str = Field(...)
    agent_id: str = Field(...)
    run_id: str = Field(...)

