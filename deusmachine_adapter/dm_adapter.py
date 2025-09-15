import os
from typing import Optional, Dict, Any, List
from httpx import AsyncClient

from config import get_config
from app_logging.logger import logger
from schemas.shared_models.models import Message

config = get_config()


class DMAdapter:
    def __init__(self):
        self.deusmachina_url = config.deusmachina_url
        self._client: Optional[AsyncClient] = None

    async def ensure_client(self) -> AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = AsyncClient(
                base_url=self.deusmachina_url,
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def create_agent(
        self,
        tenant_id: str,
        agent_id: str,
        run_id: str,
        metadata: Dict[str, Any],
        blueprint: str,
        dynamic_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        client = await self.ensure_client()
        url = f"{self.deusmachina_url}/agents/{tenant_id}/{agent_id}/{run_id}"
        payload = {
            "metadata": metadata,
            "blueprint": blueprint,
            "api_keys": {
                "groq": os.getenv("GROQ_API_KEY"),
                "openai": os.getenv("OPENAI_API_KEY"),
            },
        }
        if dynamic_data:
            payload["dynamic_data"] = dynamic_data

        try:
            response = await client.post(
                url,
                json=payload,
            )
            logger.info(f"Response: {response.json()}")
            response.raise_for_status()
            if response.status_code == 201:
                logger.info(
                    f"Agent {tenant_id}/{agent_id}/{run_id} created successfully: {response.json()}"
                )
            else:
                error_text = response.text
                logger.warning(
                    f"Failed to create agent {tenant_id}/{agent_id}/{run_id}: {error_text}"
                )
        except Exception as e:
            logger.warning(f"Failed to create agent: {e}")
            raise

    async def run_agent(
        self,
        tenant_id: str,
        agent_id: str,
        run_id: str,
        user_input: str,
        messages: List[Message] = [],
        dynamic_data: Dict[str, Any] | None = None,
        streaming: bool = False,
        num2words: bool = False,
        vocab_replace: bool = False,
        image_url: str = "",
        entry_node_name: str = "",
    ) -> List[Dict[str, Any]]:
        client = await self.ensure_client()
        url = f"{self.deusmachina_url}/agents/{tenant_id}/{agent_id}/{run_id}/run"
        payload = {
            "user_input": user_input,
            "streaming": streaming,
            "variables": {"num2words": num2words, "vocab_replace": vocab_replace},
        }
        if image_url:
            payload["image_url"] = image_url
        if entry_node_name:
            payload["entry_node_name"] = entry_node_name
        if len(messages) > 0:
            payload["messages"] = [
                m.model_dump(by_alias=True) if hasattr(m, "model_dump") else m
                for m in messages
            ]
        if dynamic_data:
            payload["dynamic_data"] = dynamic_data

        try:
            response = await client.post(
                url,
                json=payload,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to run agent: {e}")
            raise

    async def delete_agent(self, tenant_id: str, agent_id: str, run_id: str) -> None:
        client = await self.ensure_client()
        url = f"{self.deusmachina_url}/agents/{tenant_id}/{agent_id}/{run_id}"

        try:
            response = await client.delete(url)
            response.raise_for_status()
            if response.status_code == 200:
                logger.info(
                    f"Agent {tenant_id}/{agent_id}/{run_id} deleted successfully"
                )
            else:
                error_text = response.text
                logger.warning(
                    f"Failed to delete agent {tenant_id}/{agent_id}/{run_id}: {error_text}"
                )
        except Exception as e:
            logger.warning(f"Failed to delete agent: {e}")
            raise

    async def completions_api(
        self, user_input: str, model: Dict[str, Any], streaming: bool = False
    ) -> str:
        client = await self.ensure_client()
        url = f"{self.deusmachina_url}/features/completions"
        payload = {
            "user_input": user_input,
            "model": model,
            "streaming": streaming,
            "api_keys": {
                "groq": os.getenv("GROQ_API_KEY"),
                "openai": os.getenv("OPENAI_API_KEY"),
            },
        }

        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to run completions api: {e}")
            raise
