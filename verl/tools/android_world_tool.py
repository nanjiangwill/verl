import json
import logging
import os
from typing import Any, Optional, Tuple
from uuid import uuid4

import requests

from .base_tool import BaseTool
from .sandbox_fusion_tools import PoolMode, init_execution_pool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AndroidWorldTool(BaseTool):
    """A tool for interacting with an Android environment via HTTP APIs."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        default_schema = OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "android_action",
                    "description": "Perform an action in the Android world.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "The action to perform, e.g. tap settings icon.",
                            }
                        },
                        "required": ["action"],
                    },
                },
            }
        )
        super().__init__(config, tool_schema or default_schema)
        self._instance_dict = {}
        self.num_workers = config.get("num_workers", 10)
        self.rate_limit = config.get("rate_limit", 10)
        self.default_timeout = config.get("default_timeout", 30)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )
        self.android_world_url = config.get("android_world_url", "")
        if self.android_world_url == "":
            raise ValueError("android_world_url is not set")
        logger.info("Init AndroidWorldTool with config: %s", config)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": [],
        }
        return instance_id

    def _execute_action(self, action: str, timeout: int) -> Tuple[str, dict]:
        payload = {"action": action}
        try:
            resp = requests.post(self.android_world_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:  # broad catch to log remote failures
            logger.warning("AndroidWorld API error: %s", e)
            data = {"result": "error", "error": str(e)}
        return json.dumps(data.get("result", "")), data

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        action = parameters.get("action", "")
        timeout = parameters.get("timeout", self.default_timeout)
        if not isinstance(action, str):
            action = str(action)
        result_text, metadata = await self.execution_pool.execute.remote(self._execute_action, action, timeout)
        self._instance_dict[instance_id]["reward"].append(result_text.strip())
        return result_text, 0.0, metadata

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
