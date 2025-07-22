# Copyright 2025 Amazon.com Inc and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import copy
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, TypeVar

import ray
import ray.actor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema
from verl.workers.rollout.schemas import AsyncRolloutRequest

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")

BASE_CHAT_HISTORY = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]

# Adapted from verl/tools/sandbox_fusion_tools.py
class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class SearchExecutionWorker:
    """Worker for executing search operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """Execute function with optional rate limiting."""
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    # TODO we should make this available to the tool caller
                    logger.warning(f"Error when executing search: {e}")
        else:
            return fn(*fn_args, **fn_kwargs)


def init_self_context_management_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    """Initialize search execution pool."""
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(SearchExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")

class SelfMemoryManagementAction(str, Enum):
    PIN_MEMORY = "pin_memory"
    UNPIN_MEMORY = "unpin_memory"
    CLEAR_CONTEXT = "clear_context"


class SelfContextManagementTool(BaseTool):
    """A tool for executing the code using sanbox fusion image.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "self_context_management",
                "description": "A tool for managing self-context memory operations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pin_memory": {
                            "type": "string",
                            "description": "Memory content to pin",
                        },
                        "unpin_memory": {
                            "type": "string",
                            "description": "Memory index to unpin (as string)",
                        },
                        "clear_context": {
                            "type": "string",
                            "description": "Clear context (empty string)",
                        },
                    },
                    "required": [],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        # TODO: better documentation for the config
        self.num_workers = config.get("num_workers", 10)
        self.rate_limit = config.get("rate_limit", 10)
        self.default_timeout = config.get("default_timeout", 30)
        self.default_language = config.get("default_language", "python")
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_self_context_management_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )
        self.valid_memory_management_actions = [action.value for action in SelfMemoryManagementAction]
        self.processing_class = None

        logger.info(f"Init SelfContextManagementTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, req: AsyncRolloutRequest, processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin, **kwargs) -> str:
        self.processing_class = processing_class
        self._instance_dict[req.request_id] = {
            "req": req,
            "reward": None,
            "memories": {},
            "memory_boundaries": {},
            "active_memory_idx": [],
        }
        return req.request_id

    # @rollout_trace_op
    async def execute(self, req: AsyncRolloutRequest, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        if len(parameters) != 1:
            return f"Only one action per turn supported, got {len(parameters)}", 0.0, {}

        if SelfMemoryManagementAction.PIN_MEMORY.value in parameters:
            memory = parameters.get(SelfMemoryManagementAction.PIN_MEMORY.value, "")
            return self.execute_pin_memory(req, memory)
            
        elif SelfMemoryManagementAction.UNPIN_MEMORY.value in parameters:
            try:
                memory_idx = int(parameters.get(SelfMemoryManagementAction.UNPIN_MEMORY.value, ""))
            except ValueError:
                available = self._instance_dict[req.request_id]['active_memory_idx']
                return f"Memory index must be integer from {available}", 0.0, {}
                
            if memory_idx not in self._instance_dict[req.request_id]["active_memory_idx"]:
                return f"Memory {memory_idx} not in active list", 0.0, {}
            return self.execute_unpin_memory(req, memory_idx)
            
        elif SelfMemoryManagementAction.CLEAR_CONTEXT.value in parameters:
            return self.execute_clear_context(req)
        else:
            return f"Invalid action, supported: {self.valid_memory_management_actions}", 0.0, {}

    def execute_pin_memory(self, req: AsyncRolloutRequest, memory: str):
        # We start memory idx from 1 instead of 0.
        memory_idx = len(self._instance_dict[req.request_id]["memories"]) + 1

        formatted_memory = f"[memory {memory_idx}] {memory}"
        boundaries = self._get_memory_boundaries(req, self.processing_class, formatted_memory)
            
        self._instance_dict[req.request_id]["memories"][memory_idx] = formatted_memory
        self._instance_dict[req.request_id]["memory_boundaries"][memory_idx] = boundaries
        self._instance_dict[req.request_id]["active_memory_idx"].append(memory_idx)
        return formatted_memory, 0.0, {}
    
    def execute_unpin_memory(self, req: AsyncRolloutRequest, memory_idx: int):
        self._instance_dict[req.request_id]["active_memory_idx"].remove(memory_idx)
        # Remove the memory boundaries from valid boundaries
        start_idx, end_idx = self._instance_dict[req.request_id]["memory_boundaries"][memory_idx]
        req.valid_boundaries_for_input_ids = [x for x in req.valid_boundaries_for_input_ids if x < start_idx or x >= end_idx]
        return f"", 0.0, {}
    
    def execute_clear_context(self, req: AsyncRolloutRequest):
        valid_boundaries = list(range(req.prompt_ids.shape[-1]))
        invalid_boundaries = []

        for memory_idx in self._instance_dict[req.request_id]["active_memory_idx"]:
            start_idx, end_idx = self._instance_dict[req.request_id]["memory_boundaries"][memory_idx]
            invalid_boundaries.append((req.input_ids.shape[-1], valid_boundaries[-1], start_idx))
            valid_boundaries = valid_boundaries + list(range(start_idx, end_idx))

        # add the last boundary
        if valid_boundaries[-1] != req.input_ids.shape[-1]:
            invalid_boundaries.append((req.input_ids.shape[-1], valid_boundaries[-1], req.input_ids.shape[-1]))

        # set the next turn input ids
        req.valid_boundaries_for_input_ids = valid_boundaries

        # set the invalid boundaries
        for i_boundary in invalid_boundaries:
            append_flag = True
            for existing_boundary in req.invalid_boundaries_for_attention_mask:
                if existing_boundary[1] == i_boundary[1] and existing_boundary[2] == i_boundary[2]:
                    append_flag = False
                    break
            if append_flag:
                req.invalid_boundaries_for_attention_mask.append(i_boundary)

        return f"", 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

    def _get_memory_boundaries(self, req: AsyncRolloutRequest, processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin, memory: str):
        start_idx = req.input_ids.shape[-1]
        _req = copy.deepcopy(req)
        content_info = _req.add_tool_response_messages(processing_class, [memory])
        
        if content_info is None or "input_ids" not in content_info:
            return [start_idx, start_idx]
        
        memory_ids_len = content_info["input_ids"][..., _req.base_conv_wo_gen_prompt_end_pos :].shape[-1]
        return [start_idx, start_idx + memory_ids_len]