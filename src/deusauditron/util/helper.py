import json
import os
import re
import textwrap
import redis
from typing import Any, Dict, List, Optional
import importlib.resources as pkg_resources

from deusauditron.config.config import get_config
from deusauditron.app_logging.logger import logger

from deusauditron.schemas.shared_models.models import Message, MessageRole, MiniInteractionLog


class Trinity:
    @staticmethod
    def get_prompt_file(filename):
        return pkg_resources.files("deusauditron.prompts").joinpath(filename)

    @staticmethod
    def get_prompt_file_content(filename: str) -> str:
        with Trinity.get_prompt_file(filename).open("r", encoding="utf-8") as f:
            prompt_text = f.read()
        return prompt_text

    @staticmethod
    def replace_variables(template_string: str, global_data: dict) -> str:
        def replacer(match):
            var_name = match.group(1)
            value = global_data.get(var_name, f"{{{var_name}}}")
            if isinstance(value, list):
                value = ", ".join(map(str, value))
            elif not isinstance(value, str):
                value = str(value)
            return value
        return re.sub(r"\{(.*?)\}", replacer, template_string)

    @staticmethod
    def get_conversation_str(messages: List[Message]) -> str:
        conversation_lines = []
        if messages:
            for msg in messages:
                if msg.role == MessageRole.USER:
                    conversation_lines.append(f"User: {msg.content}")
                elif msg.role == MessageRole.ASSISTANT:
                    conversation_lines.append(f"Assistant: {msg.content}")
        conversation_str = "\n".join(conversation_lines)
        return textwrap.indent(conversation_str, "            ")

    @staticmethod
    def get_interaction_str(messages: List[MiniInteractionLog]) -> str:
        """
        Converts a list of provider-agnostic messages into a formatted string representation
        of the conversation, with "User: " and "Assistant: " prefixes.

        Args:
            messages: A list of provider-agnostic message dictionaries.

        Returns:
            A string representing the conversation, with each message indented.
        """
        conversation_lines = []
        if messages:
            for msg in messages:
                conversation_lines.append(f"User: {msg.user_message}")
                conversation_lines.append(f"Assistant: {msg.llm_response}")

        conversation_str = "\n".join(conversation_lines)
        return textwrap.indent(conversation_str, "            ")  # 12 spaces
    
    @staticmethod
    async def aread_from_redis(redis_source) -> str:
        return Trinity.read_from_redis(redis_source)

    # Fetch data from Redis
    @staticmethod
    def read_from_redis(redis_source) -> str:
        try:
            config = get_config()
            redis_client = redis.StrictRedis.from_url(
                config.redis.redis_url, decode_responses=True
            )
            # Parse the source string: "namespace:index:key"
            parts = redis_source.split(":")
            if len(parts) != 3:
                # TODO: handle this in linting
                logger.error(
                    "Invalid Redis source format. Expected 'namespace:index:key'."
                )
                return ""
            namespace, index, key = parts
            full_key = f"{namespace}:{index}:{key}"  # Example: "order_agent_history_kb:index:12345"
            # Fetch data from Redis (assuming it's stored as a JSON string)
            value = redis_client.get(full_key)
            if value is None:
                print(f"Warning: No data found for Redis key {full_key}")
                return ""
            return str(
                value
            )  # Return raw data (or use json.loads(value) if JSON is stored)
        except Exception as e:
            logger.error(f"Error fetching from Redis: {e}")
            return ""


    @staticmethod
    def get_turn_evaluation_prompt() -> str:
        return Trinity.get_prompt_file_content("turn.evaulation.prompt")

    @staticmethod
    def get_node_evaluation_prompt() -> str:
        return Trinity.get_prompt_file_content("node.evaulation.prompt")

    @staticmethod
    def get_intent_evaluation_prompt() -> str:
        return Trinity.get_prompt_file_content("intent.evaulation.prompt")

    @staticmethod
    def get_conversation_evaluation_prompt() -> str:
        return Trinity.get_prompt_file_content("conversation.evaulation.prompt")

    @staticmethod
    def get_llm_response_conversion_prompt() -> str:
        return Trinity.get_prompt_file_content("llm_response_conversion.prompt")

    @staticmethod
    def get_auto_refine_turn_node_prompt() -> str:
        return Trinity.get_prompt_file_content("auto_refine_turn_node.prompt")

    @staticmethod
    def get_auto_refine_flow_prompt() -> str:
        return Trinity.get_prompt_file_content("auto_refine_flow.prompt")

    @staticmethod
    def get_auto_refine_response_conversion_prompt() -> str:
        return Trinity.get_prompt_file_content("auto_refine_response_conversion.prompt")

    @staticmethod
    def get_auto_refine_flow_response_conversion_prompt() -> str:
        return Trinity.get_prompt_file_content("auto_refine_flow_response_conversion.prompt")

    @staticmethod
    def get_auto_refine_intent_prompt() -> str:
        return Trinity.get_prompt_file_content("auto_refine_intent.prompt")

    @staticmethod
    def get_auto_refine_intent_response_conversion_prompt() -> str:
        return Trinity.get_prompt_file_content("auto_refine_intent_response_conversion.prompt")

    @staticmethod
    def write_to_s3(bucket_name: str, object_key: str, data: str) -> None:
        # Optional: add real S3 here if desired; keep a stub for compatibility
        raise RuntimeError("S3 write not implemented in Deusauditron helper")

