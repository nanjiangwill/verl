import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from tensordict import TensorDict
from transformers import AutoConfig, AutoTokenizer
from utils_sglang import get_rollout_config, prepare_inputs

from verl.protocol import DataProto
from verl.tools.android_world_tool import AndroidWorldTool
from verl.workers.rollout.schemas import AsyncRolloutRequest, AsyncRolloutRequestStateEnum
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout


def get_android_messages():
    user_prompt = {"role": "user", "content": "Open the settings app."}
    expect_turn_0_msg = {
        "role": "assistant",
        "content": "Tapping on the settings icon.",
        "tool_calls": [{"type": "function", "function": {"name": "android_action", "arguments": {"action": "open_settings"}}}],
    }
    expect_turn_1_msg = {"role": "assistant", "content": "<answer>Done.</answer>"}

    tool_return_0_msg = {"role": "tool", "content": "success"}
    return [user_prompt], [expect_turn_0_msg, expect_turn_1_msg], [tool_return_0_msg]


class TestRolloutWithAndroidWorldTool:
    @pytest.fixture
    def qwen_tokenizer(self):
        local_model_path = "Qwen/Qwen2.5-0.5B"
        tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @pytest.fixture
    def qwen_model_config(self):
        local_model_path = "Qwen/Qwen2.5-0.5B"
        return AutoConfig.from_pretrained(local_model_path)

    @pytest.fixture
    def android_data(self, qwen_tokenizer):
        user_prompt, expect_turn_array, tool_return_array = get_android_messages()
        prompts = [[msg] for msg in user_prompt]
        preencode_turn_array = [qwen_tokenizer.apply_chat_template([turn], tokenize=False, add_generation_prompt=False) for turn in expect_turn_array]
        preencode_tool_return_array = [qwen_tokenizer.apply_chat_template([turn], tokenize=False, add_generation_prompt=True) for turn in tool_return_array]
        return prompts, preencode_turn_array, preencode_tool_return_array

    @pytest.fixture
    def android_rollout_config(self):
        max_prompt_length = 4096
        max_response_length = 300
        dtype = "bfloat16"
        tensor_parallel_size = 1
        tool_path = "./resource/tool_configs/android_world_tool_config"
        return get_rollout_config(max_response_length, max_prompt_length, dtype, tensor_parallel_size, tool_path)

    @pytest.fixture
    def android_data_proto(self, android_data, qwen_tokenizer):
        preencode_prompts, _, _ = android_data
        prompts = [qwen_tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in preencode_prompts]
        input_ids, attention_mask, position_ids = prepare_inputs(qwen_tokenizer, prompts, 1000)
        prompt_dict = TensorDict(
            {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids},
            batch_size=input_ids.shape[0],
        )
        messages = np.asarray(preencode_prompts)
        tools_kwargs = np.array([{"android_action": {"create_kwargs": {"ground_truth": "success"}}}], dtype=object)
        index = np.array([0], dtype=object)
        return DataProto(batch=prompt_dict, non_tensor_batch={"raw_prompt": messages, "tools_kwargs": tools_kwargs, "index": index})

    @patch.object(SGLangRollout, "_init_distributed_env", return_value=None)
    @patch.object(SGLangRollout, "_init_inference_engine", return_value=None)
    @patch.object(SGLangRollout, "_init_sampling_params", return_value=None)
    def test_tools_registration(self, mock_env, mock_engine, mock_sampling, android_rollout_config, qwen_tokenizer, qwen_model_config):
        rollout = SGLangRollout(actor_module="", config=android_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
        assert len(rollout._tool_schemas) == 1
        assert "android_action" in rollout._tool_map
        assert isinstance(rollout._tool_map["android_action"], AndroidWorldTool)
        assert rollout._tool_call_parser_type == "qwen25"

    @patch.object(AndroidWorldTool, "execute", new_callable=AsyncMock)
    @patch.object(SGLangRollout, "_init_distributed_env", return_value=None)
    @patch.object(SGLangRollout, "_init_inference_engine", return_value=None)
    @patch.object(SGLangRollout, "_init_sampling_params", return_value=None)
    def test_tool_call_basic_case(
        self,
        mock_sampling,
        mock_engine,
        mock_env,
        mock_execute,
        android_rollout_config,
        qwen_tokenizer,
        qwen_model_config,
        android_data_proto,
        android_data,
    ):
        _, expect_turn_array, tool_return_array = android_data
        mock_execute.return_value = (tool_return_array[0], 0.0, {"status": "success"})
        android_rollout_config.multi_turn.max_turns = 5
        rollout = SGLangRollout(actor_module="", config=android_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)

        req = rollout._preprocess_prompt_to_async_rollout_requests(android_data_proto, n=1)[0]
        req = MagicMock(wraps=req, spec=AsyncRolloutRequest)
        req.finalize = MagicMock()
        req_list = [req]

        rollout._handle_engine_call = MagicMock()
        futures = [asyncio.Future() for _ in expect_turn_array]
        for idx, (fut, turn) in enumerate(zip(futures, expect_turn_array)):
            fut.set_result(
                {
                    "text": turn,
                    "meta_info": {
                        "id": "dummy",
                        "finish_reason": {"type": "tool_calls" if idx < len(expect_turn_array) - 1 else "stop"},
                        "prompt_tokens": len(turn),
                        "completion_tokens": 100,
                    },
                }
            )
            if idx < len(expect_turn_array) - 1:
                assert rollout._function_call_parser.has_tool_call(turn)
                assert rollout._function_call_parser.parse_non_stream(turn)
        rollout._handle_engine_call.side_effect = futures
        rollout._tp_rank = 0
        loop = asyncio.get_event_loop()
        output_req_list = loop.run_until_complete(asyncio.gather(*[rollout._async_rollout_a_request(r, True, False) for r in req_list]))
        assert len(output_req_list) == 1
        output_req = output_req_list[0]
        assert output_req.state == AsyncRolloutRequestStateEnum.COMPLETED
        assert "android_action" in output_req.metrics
        assert output_req.metrics["android_action"][0]["status"] == "success"
        assert mock_execute.await_count == 1
        assert len(output_req.messages) == 4  # user + 2 assistant + 1 tool
        tool_counter = sum(1 for m in output_req.messages if m.role == "tool")
        assert tool_counter == 1
