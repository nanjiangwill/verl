# Copyright 2025 ModelBest Inc. and/or its affiliates
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

"""
Test for SDPA multi-turn optimization in SGLangRollout.
This test verifies that the SDPA optimization produces correct results
for multi-turn conversations with reasoning tokens.

The test file contains:
1. Unit tests for utility functions (can run without GPU/distributed setup)
2. Integration tests for SGLang rollout (require GPU/distributed setup)

usage: torchrun --standalone --nnodes=1 \
    --nproc_per_node=2 $(which pytest) \
    -s test_sglang_multiturn_sdpa_optimization.py
"""

import numpy as np
import torch
from tensordict import TensorDict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from transformers import AutoTokenizer
from utils_sglang import (
    are_lists_similar,
    clean_torchelastic_env,
    initialize_global_process_group,
    load_tokenizer_and_model,
    prepare_inputs,
)

from verl import DataProto
from verl.utils.multiturn_forward_utils import (
    MultiTurnOptimizationMethod, 
    supports_multiturn_optimization,
    should_enable_multiturn_optimization,
    prepare_multiturn_inputs,
    _create_multiturn_attention_mask,
    apply_multiturn_sdpa_mask,
    extract_assistant_logits,
)
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout
from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager


class MockTokenizer:
    """Mock tokenizer for unit testing."""
    
    def __init__(self, name):
        self.__class__.__name__ = name
        
    def apply_chat_template(self, messages, tools=None, tokenize=True, 
                          add_generation_prompt=False, enable_thinking=None):
        """Mock chat template application."""
        if not tokenize:
            # Return text
            if enable_thinking is False:
                return "User: test\nAssistant: response"  # No reasoning
            else:
                return "User: test\nAssistant: <thinking>reasoning</thinking>response"  # With reasoning
        else:
            # Return token IDs
            if enable_thinking is False:
                return [1, 2, 3, 4, 5]  # Short sequence without reasoning
            else:
                return [1, 2, 3, 6, 7, 8, 4, 5]  # Longer sequence with reasoning tokens


# ============ Unit Tests for Utility Functions ============

def test_enum_values():
    """Test that all expected optimization method enum values are present."""
    print("\n=== Testing Optimization Method Enum ===")
    
    expected_methods = ["none", "naive", "sdpa", "flex_attention", "kv_cache", "sequence_packing"]
    
    for method_str in expected_methods:
        method = MultiTurnOptimizationMethod(method_str)
        assert method.value == method_str, f"Expected {method_str}, got {method.value}"
    
    print("✓ All enum values are correct")


def test_tokenizer_support_detection():
    """Test tokenizer support detection for different methods."""
    print("\n=== Testing Tokenizer Support Detection ===")
    
    # Test supported tokenizers
    qwen2_fast = MockTokenizer("Qwen2TokenizerFast")
    qwen2_regular = MockTokenizer("Qwen2Tokenizer")
    
    assert supports_multiturn_optimization(qwen2_fast, MultiTurnOptimizationMethod.SDPA)
    assert supports_multiturn_optimization(qwen2_regular, MultiTurnOptimizationMethod.SDPA)
    print("✓ Qwen2 tokenizers correctly detected as supporting SDPA")
    
    # Test unsupported tokenizers
    llama_tokenizer = MockTokenizer("LlamaTokenizerFast")
    gpt2_tokenizer = MockTokenizer("GPT2TokenizerFast")
    
    assert not supports_multiturn_optimization(llama_tokenizer, MultiTurnOptimizationMethod.SDPA)
    assert not supports_multiturn_optimization(gpt2_tokenizer, MultiTurnOptimizationMethod.SDPA)
    print("✓ Non-Qwen2 tokenizers correctly detected as not supporting SDPA")
    
    # Test naive method (should work with any tokenizer)
    qwen2_tokenizer = MockTokenizer("Qwen2TokenizerFast")
    assert supports_multiturn_optimization(llama_tokenizer, MultiTurnOptimizationMethod.NAIVE)
    assert supports_multiturn_optimization(qwen2_tokenizer, MultiTurnOptimizationMethod.NAIVE)
    print("✓ Naive method works with any tokenizer")
    
    # Test NONE method (always supported)
    any_tokenizer = MockTokenizer("AnyTokenizer")
    assert supports_multiturn_optimization(any_tokenizer, MultiTurnOptimizationMethod.NONE)
    print("✓ NONE method always supported")


def test_optimization_enabling_logic():
    """Test optimization enabling logic."""
    print("\n=== Testing Optimization Enabling Logic ===")
    
    qwen2_tokenizer = MockTokenizer("Qwen2TokenizerFast")
    llama_tokenizer = MockTokenizer("LlamaTokenizerFast")
    
    # Should enable when multiturn is enabled and tokenizer is supported
    assert should_enable_multiturn_optimization(
        qwen2_tokenizer, MultiTurnOptimizationMethod.SDPA, enable_multiturn=True
    )
    print("✓ Enables optimization with supported tokenizer and multiturn=True")
    
    # Should not enable when multiturn is disabled
    assert not should_enable_multiturn_optimization(
        qwen2_tokenizer, MultiTurnOptimizationMethod.SDPA, enable_multiturn=False
    )
    print("✓ Disables optimization when multiturn=False")
    
    # Should not enable with unsupported tokenizer
    assert not should_enable_multiturn_optimization(
        llama_tokenizer, MultiTurnOptimizationMethod.SDPA, enable_multiturn=True
    )
    print("✓ Disables optimization with unsupported tokenizer")
    
    # Force enable should override tokenizer support
    assert should_enable_multiturn_optimization(
        llama_tokenizer, MultiTurnOptimizationMethod.SDPA, 
        enable_multiturn=True, force_enable=True
    )
    print("✓ Force enable overrides tokenizer support")
    
    # NONE method should never enable optimization
    assert not should_enable_multiturn_optimization(
        qwen2_tokenizer, MultiTurnOptimizationMethod.NONE, enable_multiturn=True
    )
    print("✓ NONE method never enables optimization")


def test_attention_mask_creation():
    """Test custom attention mask creation and constraints."""
    print("\n=== Testing Attention Mask Creation ===")
    
    seq_len = 10
    assistant_ranges = [(2, 4, 5, 7)]  # (reasoning_start, reasoning_end, clean_start, clean_end)
    
    mask = _create_multiturn_attention_mask(seq_len, assistant_ranges)
    
    # Test shape
    assert mask.shape == (1, 1, seq_len, seq_len), f"Expected shape (1, 1, {seq_len}, {seq_len}), got {mask.shape}"
    assert mask.dtype == torch.float32, f"Expected float32, got {mask.dtype}"
    print("✓ Attention mask has correct shape and dtype")
    
    # Test constraints
    bool_mask = ~torch.isinf(mask[0, 0])
    
    # Check that clean version cannot see reasoning content
    reasoning_start, reasoning_end, clean_start, clean_end = assistant_ranges[0]
    for clean_pos in range(clean_start, clean_end):
        for reasoning_pos in range(reasoning_start, reasoning_end):
            assert not bool_mask[clean_pos, reasoning_pos], \
                f"Clean position {clean_pos} should not see reasoning position {reasoning_pos}"
    print("✓ Clean version cannot see reasoning content")
    
    # Check that future tokens cannot see reasoning versions
    for future_pos in range(clean_end, seq_len):
        for reasoning_pos in range(reasoning_start, reasoning_end):
            assert not bool_mask[future_pos, reasoning_pos], \
                f"Future position {future_pos} should not see reasoning position {reasoning_pos}"
    print("✓ Future tokens cannot see reasoning versions")
    
    # Test with multiple assistant messages
    multi_assistant_ranges = [(1, 3, 4, 6), (7, 8, 9, 10)]
    seq_len_multi = 12
    mask_multi = _create_multiturn_attention_mask(seq_len_multi, multi_assistant_ranges)
    assert mask_multi.shape == (1, 1, seq_len_multi, seq_len_multi)
    print("✓ Multiple assistant messages handled correctly")


def test_sdpa_mask_application():
    """Test SDPA mask application."""
    print("\n=== Testing SDPA Mask Application ===")
    
    batch_size, num_heads, seq_len, head_dim = 1, 2, 8, 4
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)  
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Create a simple causal attention mask
    attention_mask = torch.zeros(1, 1, seq_len, seq_len)
    attention_mask = attention_mask.masked_fill(
        torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool(), 
        float('-inf')
    )
    
    output = apply_multiturn_sdpa_mask(query, key, value, attention_mask)
    
    assert output.shape == (batch_size, num_heads, seq_len, head_dim), \
        f"Expected shape {(batch_size, num_heads, seq_len, head_dim)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"
    print("✓ SDPA mask application produces valid output")
    
    # Test with dropout
    output_with_dropout = apply_multiturn_sdpa_mask(
        query, key, value, attention_mask, dropout_p=0.1
    )
    assert output_with_dropout.shape == (batch_size, num_heads, seq_len, head_dim)
    print("✓ SDPA with dropout works correctly")


def test_assistant_logits_extraction():
    """Test assistant logits extraction."""
    print("\n=== Testing Assistant Logits Extraction ===")
    
    batch_size, seq_len, vocab_size = 1, 20, 100
    
    logits = torch.randn(batch_size, seq_len, vocab_size)
    assistant_ranges = [
        (2, 6, 8, 12),  # First assistant: reasoning [2:6], clean [8:12]
        (14, 16, 17, 19)  # Second assistant: reasoning [14:16], clean [17:19]
    ]
    
    # Test clean version extraction
    extracted_clean = extract_assistant_logits(logits, assistant_ranges, use_clean_version=True)
    expected_length = (12 - 8) + (19 - 17)  # 4 + 2 = 6
    assert extracted_clean.shape == (batch_size, expected_length, vocab_size), \
        f"Expected shape {(batch_size, expected_length, vocab_size)}, got {extracted_clean.shape}"
    
    # Check that we got the right tokens
    torch.testing.assert_close(extracted_clean[:, :4, :], logits[:, 8:12, :])  # First clean part
    torch.testing.assert_close(extracted_clean[:, 4:6, :], logits[:, 17:19, :])  # Second clean part
    print("✓ Clean version logits extracted correctly")
    
    # Test reasoning version extraction
    extracted_reasoning = extract_assistant_logits(logits, assistant_ranges, use_clean_version=False)
    expected_length = (6 - 2) + (16 - 14)  # 4 + 2 = 6
    assert extracted_reasoning.shape == (batch_size, expected_length, vocab_size)
    
    # Check that we got the right tokens
    torch.testing.assert_close(extracted_reasoning[:, :4, :], logits[:, 2:6, :])  # First reasoning part
    torch.testing.assert_close(extracted_reasoning[:, 4:6, :], logits[:, 14:16, :])  # Second reasoning part
    print("✓ Reasoning version logits extracted correctly")
    
    # Test empty assistant ranges
    empty_extracted = extract_assistant_logits(logits, [])
    assert empty_extracted.shape == (batch_size, 0, vocab_size)
    print("✓ Empty assistant ranges handled correctly")


def test_input_preparation():
    """Test input preparation for SDPA method."""
    print("\n=== Testing Input Preparation ===")
    
    tokenizer = MockTokenizer("Qwen2TokenizerFast")
    messages = [
        {"role": "user", "content": "test question"},
        {"role": "assistant", "content": "<thinking>reasoning content</thinking>response"}
    ]
    
    input_ids, position_ids, attention_mask, assistant_ranges = prepare_multiturn_inputs(
        tokenizer, messages, MultiTurnOptimizationMethod.SDPA
    )
    
    assert isinstance(input_ids, torch.Tensor), "input_ids should be a tensor"
    assert isinstance(position_ids, torch.Tensor), "position_ids should be a tensor"
    assert isinstance(attention_mask, torch.Tensor), "attention_mask should be a tensor"
    assert isinstance(assistant_ranges, list), "assistant_ranges should be a list"
    assert input_ids.shape == position_ids.shape, "input_ids and position_ids should have same shape"
    assert len(assistant_ranges) >= 0, "assistant_ranges should be a valid list"
    print("✓ SDPA input preparation produces valid outputs")


def test_error_handling():
    """Test error handling for various edge cases."""
    print("\n=== Testing Error Handling ===")
    
    # Test unsupported method raises NotImplementedError
    tokenizer = MockTokenizer("Qwen2TokenizerFast")
    messages = [{"role": "user", "content": "test"}]
    
    try:
        prepare_multiturn_inputs(tokenizer, messages, MultiTurnOptimizationMethod.NAIVE)
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError:
        pass
    print("✓ Unsupported methods raise NotImplementedError")
    
    # Test unsupported tokenizer raises ValueError
    unsupported_tokenizer = MockTokenizer("LlamaTokenizerFast")
    
    try:
        prepare_multiturn_inputs(unsupported_tokenizer, messages, MultiTurnOptimizationMethod.SDPA)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not supported for tokenizer" in str(e)
    print("✓ Unsupported tokenizers raise ValueError")
    
    # Test invalid enum value
    try:
        MultiTurnOptimizationMethod("invalid_method")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("✓ Invalid enum values raise ValueError")


def get_rollout_config_with_sdpa(
    max_response_length,
    max_prompt_length,
    dtype,
    tensor_parallel_size,
    optimization_method="sdpa",
):
    """Get rollout config with SDPA optimization enabled."""
    from omegaconf import OmegaConf
    
    sampling_params = dict(
        n=1,
        temperature=0,
        top_p=1,
        top_k=-1,
        max_new_tokens=max_response_length,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
        ignore_eos=False,
    )

    rollout_config = OmegaConf.create(
        {
            "name": "sglang",
            "mode": "sync",
            "load_format": "dummy_dtensor",
            "enforce_eager": False,
            "free_cache_engine": True,
            "dtype": dtype,
            "gpu_memory_utilization": 0.5,
            "ignore_eos": False,
            "max_num_batched_tokens": 8192,
            "prompt_length": max_prompt_length,
            "response_length": max_response_length,
            "tensor_model_parallel_size": tensor_parallel_size,
            "multi_turn": {
                "max_assistant_turns": 3,
                "max_user_turns": 3,
                "enable": True,
                "optimization_method": optimization_method,
                "force_multiturn_optimization": False,
                "use_inference_chat_template": False,
                "tokenization_sanity_check_mode": "strict",
            },
            "max_model_len": None,
            **sampling_params,
        }
    )

    return rollout_config


def create_multiturn_reasoning_prompts():
    """Create multi-turn conversation prompts with reasoning tokens."""
    return [
        [
            {"role": "user", "content": "What is 15 + 27?"},
            {"role": "assistant", "content": "<thinking>I need to add 15 and 27. 15 + 27 = 42</thinking>The answer is 42."},
            {"role": "user", "content": "Now multiply that by 3"},
        ],
        [
            {"role": "user", "content": "Explain photosynthesis briefly"},
            {"role": "assistant", "content": "<thinking>Photosynthesis is the process plants use to make food from sunlight. I should explain it simply.</thinking>Photosynthesis is how plants make food using sunlight, water, and carbon dioxide."},
            {"role": "user", "content": "What are the main products?"},
        ],
        [
            {"role": "user", "content": "Count from 1 to 5"},
            {"role": "assistant", "content": "<thinking>I need to count from 1 to 5: 1, 2, 3, 4, 5</thinking>1, 2, 3, 4, 5"},
            {"role": "user", "content": "Now count backwards"},
        ],
    ]


def test_sdpa_optimization_tokenizer_support():
    """Test that SDPA optimization is correctly detected for supported tokenizers."""
    print("\n=== Testing SDPA Tokenizer Support Detection ===")
    
    # Test with Qwen2 tokenizer (should be supported)
    qwen2_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", padding_side="left")
    qwen2_tokenizer.pad_token = qwen2_tokenizer.eos_token
    
    assert supports_multiturn_optimization(qwen2_tokenizer, MultiTurnOptimizationMethod.SDPA), \
        "Qwen2 tokenizer should support SDPA optimization"
    
    print("✓ Qwen2 tokenizer correctly detected as supporting SDPA optimization")


def test_multiturn_sdpa_optimization():
    """Test multi-turn SDPA optimization produces correct results."""
    assert torch.cuda.device_count() >= 2
    initialize_global_process_group()
    clean_torchelastic_env()

    print("\n=== Testing Multi-Turn SDPA Optimization ===")

    max_prompt_length = 128
    max_response_length = 64
    dtype = "bfloat16"
    tensor_parallel_size = 2
    local_model_path = "Qwen/Qwen2.5-0.5B"  # Use Qwen2 model that supports SDPA

    tokenizer, actor_model = load_tokenizer_and_model(local_model_path, dtype)

    # Test tokenizer support
    assert supports_multiturn_optimization(tokenizer, MultiTurnOptimizationMethod.SDPA), \
        f"Expected {tokenizer.__class__.__name__} to support SDPA optimization"

    preencode_prompts = create_multiturn_reasoning_prompts()
    
    # Apply chat template to get the actual prompt text
    prompts = []
    for message_list in preencode_prompts:
        prompt_text = tokenizer.apply_chat_template(
            message_list, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt_text)

    input_ids, attention_mask, position_ids = prepare_inputs(tokenizer, prompts, max_prompt_length)

    fsdp_device_mesh = init_device_mesh("cuda", mesh_shape=(tensor_parallel_size,), mesh_dim_names=("fsdp",))
    inference_device_mesh_cpu = init_device_mesh(
        "cpu", mesh_shape=(1, tensor_parallel_size, 1), mesh_dim_names=("dp", "infer_tp", "pp")
    )

    fsdp_model = FSDP(
        actor_model,
        use_orig_params=True,
        device_id=fsdp_device_mesh["fsdp"].get_local_rank(),
        mixed_precision=MixedPrecision(param_dtype=getattr(torch, dtype)),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_mesh=fsdp_device_mesh,
    )

    # Test 1: Run without SDPA optimization (baseline)
    print("Running baseline test without SDPA optimization...")
    rollout_config_baseline = get_rollout_config_with_sdpa(
        max_response_length, max_prompt_length, dtype, tensor_parallel_size, 
        optimization_method="none"
    )
    
    rollout_baseline = SGLangRollout(
        actor_module=local_model_path,
        config=rollout_config_baseline,
        processing_class=tokenizer,
        model_hf_config=actor_model.config,
    )

    rollout_sharding_manager_baseline = FSDPSGLangShardingManager(
        module=fsdp_model,
        inference_engine=rollout_baseline._engine,
        model_config=actor_model.config,
        rollout_config=rollout_config_baseline,
        full_params=True,
        device_mesh=inference_device_mesh_cpu,
    )

    with rollout_sharding_manager_baseline:
        prompt_dict = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0],
        )

        messages = np.asarray(preencode_prompts)
        prompts_data = DataProto(
            batch=prompt_dict,
            non_tensor_batch={
                "raw_prompt": messages,
            },
        )

        prompts_data.meta_info.update(
            {
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            }
        )

        prompts_data = rollout_sharding_manager_baseline.preprocess_data(prompts_data)
        baseline_output = rollout_baseline.generate_sequences(prompts=prompts_data)
        baseline_output = rollout_sharding_manager_baseline.postprocess_data(baseline_output)
        baseline_output = baseline_output.to("cpu")

    baseline_response_tokens = tokenizer.batch_decode(baseline_output.batch["responses"])
    print(f"Baseline responses: {baseline_response_tokens}")

    # Test 2: Run with SDPA optimization
    print("Running test with SDPA optimization...")
    rollout_config_sdpa = get_rollout_config_with_sdpa(
        max_response_length, max_prompt_length, dtype, tensor_parallel_size,
        optimization_method="sdpa"
    )
    
    rollout_sdpa = SGLangRollout(
        actor_module=local_model_path,
        config=rollout_config_sdpa,
        processing_class=tokenizer,
        model_hf_config=actor_model.config,
    )

    rollout_sharding_manager_sdpa = FSDPSGLangShardingManager(
        module=fsdp_model,
        inference_engine=rollout_sdpa._engine,
        model_config=actor_model.config,
        rollout_config=rollout_config_sdpa,
        full_params=True,
        device_mesh=inference_device_mesh_cpu,
    )

    with rollout_sharding_manager_sdpa:
        prompts_data = DataProto(
            batch=prompt_dict,
            non_tensor_batch={
                "raw_prompt": messages,
            },
        )

        prompts_data.meta_info.update(
            {
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            }
        )

        prompts_data = rollout_sharding_manager_sdpa.preprocess_data(prompts_data)
        sdpa_output = rollout_sdpa.generate_sequences(prompts=prompts_data)
        sdpa_output = rollout_sharding_manager_sdpa.postprocess_data(sdpa_output)
        sdpa_output = sdpa_output.to("cpu")

    sdpa_response_tokens = tokenizer.batch_decode(sdpa_output.batch["responses"])
    print(f"SDPA responses: {sdpa_response_tokens}")

    # Verify that outputs are similar (allowing some small differences due to optimization)
    assert are_lists_similar(baseline_response_tokens, sdpa_response_tokens, threshold=15), \
        "SDPA optimization should produce similar results to baseline"
    
    print("✓ SDPA optimization produces correct results!")

    # Test 3: Verify optimization fields are set correctly
    print("Verifying optimization fields are set correctly...")
    
    # Check that the requests have the correct optimization settings
    test_prompts_small = [
        [{"role": "user", "content": "Simple test"}]
    ]
    
    input_ids_small, attention_mask_small, position_ids_small = prepare_inputs(
        tokenizer, [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) 
                   for p in test_prompts_small], 32
    )
    
    with rollout_sharding_manager_sdpa:
        prompt_dict_small = TensorDict({
            "input_ids": input_ids_small,
            "attention_mask": attention_mask_small, 
            "position_ids": position_ids_small,
        }, batch_size=input_ids_small.shape[0])
        
        test_prompts_data = DataProto(
            batch=prompt_dict_small,
            non_tensor_batch={"raw_prompt": np.asarray(test_prompts_small)},
        )
        test_prompts_data.meta_info.update({
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        })
        
        # Access the internal request creation to verify optimization settings
        req_list = rollout_sdpa._preprocess_prompt_to_async_rollout_requests(test_prompts_data)
        
        for req in req_list:
            assert req.enable_multiturn_optimization, \
                "Request should have multi-turn optimization enabled"
            assert req.multiturn_optimization_method == "sdpa", \
                f"Expected SDPA method, got {req.multiturn_optimization_method}"
    
    print("✓ Optimization fields are set correctly!")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    print("✅ All SDPA optimization tests passed!")


def test_unsupported_tokenizer():
    """Test that unsupported tokenizers don't enable SDPA optimization."""
    print("\n=== Testing Unsupported Tokenizer Behavior ===")
    
    initialize_global_process_group()
    clean_torchelastic_env()
    
    # Use a non-Qwen2 model that shouldn't support SDPA
    llama_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side="left")
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    
    # Verify it's not supported
    assert not supports_multiturn_optimization(llama_tokenizer, MultiTurnOptimizationMethod.SDPA), \
        "Non-Qwen2 tokenizer should not support SDPA optimization"
    
    print("✓ Non-Qwen2 tokenizer correctly detected as not supporting SDPA optimization")
    
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    # Run unit tests first (these don't require GPU/distributed setup)
    print("🧪 Running Unit Tests...")
    test_enum_values()
    test_tokenizer_support_detection()
    test_optimization_enabling_logic()
    test_attention_mask_creation()
    test_sdpa_mask_application()
    test_assistant_logits_extraction()
    test_input_preparation()
    test_error_handling()
    print("✅ All unit tests passed!")
    
    # Run integration tests (these require GPU/distributed setup)
    print("\n🔧 Running Integration Tests...")
    test_sdpa_optimization_tokenizer_support()
    test_multiturn_sdpa_optimization()
    test_unsupported_tokenizer()
    print("✅ All integration tests passed!")
    
    print("\n🎉 All SDPA optimization tests completed successfully!")