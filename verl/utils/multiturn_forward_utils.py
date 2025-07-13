# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Multi-turn forward pass optimization utilities.

This module implements various optimization strategies for efficient single-pass training
of multi-turn conversations while maintaining inference-like context visibility.
Only enabled for specific tokenizers that support reasoning token masking.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin

logger = logging.getLogger(__name__)


class MultiTurnOptimizationMethod(str, Enum):
    """Enum for multi-turn optimization methods."""
    
    NONE = "none"  # No optimization, standard processing
    NAIVE = "naive"  # Turn-by-turn forward passes
    SDPA = "sdpa"  # Custom attention masks with SDPA
    FLEX_ATTENTION = "flex_attention"  # FlexAttention for ultra-long conversations  
    KV_CACHE = "kv_cache"  # KV cache acceleration with cropping
    SEQUENCE_PACKING = "sequence_packing"  # Sequence packing optimization


def supports_multiturn_optimization(
    processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin],
    method: MultiTurnOptimizationMethod
) -> bool:
    """
    Check if the tokenizer/processor supports the specified multi-turn optimization method.
    
    Args:
        processing_class: Tokenizer or processor to check
        method: Optimization method to check support for
        
    Returns:
        True if the optimization method is supported, False otherwise
    """
    if method == MultiTurnOptimizationMethod.NONE:
        return True  # No optimization is always supported
    
    # Check if it's a processor with a tokenizer
    if isinstance(processing_class, ProcessorMixin):
        if hasattr(processing_class, 'tokenizer'):
            tokenizer = processing_class.tokenizer
        else:
            return False
    else:
        tokenizer = processing_class
    
    tokenizer_class_name = tokenizer.__class__.__name__
    
    if method == MultiTurnOptimizationMethod.SDPA:
        # SDPA optimization currently only supports Qwen2 tokenizers
        supported_tokenizers = ["Qwen2TokenizerFast", "Qwen2Tokenizer"]
        return tokenizer_class_name in supported_tokenizers
    
    elif method == MultiTurnOptimizationMethod.NAIVE:
        # Naive method works with any tokenizer
        return True
    
    elif method in [
        MultiTurnOptimizationMethod.FLEX_ATTENTION,
        MultiTurnOptimizationMethod.KV_CACHE,
        MultiTurnOptimizationMethod.SEQUENCE_PACKING
    ]:
        # Other methods not yet implemented - could support same tokenizers as SDPA
        supported_tokenizers = ["Qwen2TokenizerFast", "Qwen2Tokenizer"]
        return tokenizer_class_name in supported_tokenizers
    
    return False


def should_enable_multiturn_optimization(
    processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin],
    method: MultiTurnOptimizationMethod,
    enable_multiturn: bool,
    force_enable: bool = False
) -> bool:
    """
    Determine if multi-turn optimization should be enabled.
    
    Args:
        processing_class: Tokenizer or processor
        method: Optimization method to use
        enable_multiturn: Whether multi-turn mode is enabled
        force_enable: Force enable optimization regardless of tokenizer support
        
    Returns:
        True if optimization should be enabled
    """
    if not enable_multiturn or method == MultiTurnOptimizationMethod.NONE:
        return False
        
    if force_enable:
        logger.warning(
            f"Multi-turn optimization method '{method}' force enabled. This may not work properly "
            "with unsupported tokenizers."
        )
        return True
    
    is_supported = supports_multiturn_optimization(processing_class, method)
    
    if enable_multiturn and not is_supported:
        logger.info(
            f"Multi-turn mode enabled but tokenizer {processing_class.__class__.__name__} "
            f"does not support optimization method '{method}'. Using standard multi-turn processing."
        )
    
    return is_supported




def prepare_multiturn_inputs(
    processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin],
    messages: List[Dict],
    method: MultiTurnOptimizationMethod,
    tools: Optional[List[Dict]] = None,
    max_model_len: int = 32768,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """
    Prepare inputs for multi-turn optimization based on the specified method.
    
    Args:
        processing_class: Tokenizer or processor for the model
        messages: List of conversation messages (dict format)
        method: Optimization method to use
        tools: Optional list of tool schemas  
        max_model_len: Maximum model context length
        **kwargs: Additional method-specific arguments
        
    Returns:
        Tuple of (input_ids, position_ids, attention_mask, assistant_ranges)
        - input_ids: The optimized token sequence
        - position_ids: Position IDs for the sequence
        - attention_mask: Custom attention mask (if applicable)
        - assistant_ranges: List of token ranges for assistant messages
    """
    if not supports_multiturn_optimization(processing_class, method):
        raise ValueError(
            f"Multi-turn optimization method '{method}' not supported for tokenizer: "
            f"{processing_class.__class__.__name__}"
        )
    
    if method == MultiTurnOptimizationMethod.SDPA:
        return _prepare_multiturn_sdpa_inputs(processing_class, messages, tools, max_model_len)
    elif method == MultiTurnOptimizationMethod.NAIVE:
        return _prepare_multiturn_naive_inputs(processing_class, messages, tools, max_model_len)
    elif method == MultiTurnOptimizationMethod.FLEX_ATTENTION:
        return _prepare_multiturn_flex_attention_inputs(processing_class, messages, tools, max_model_len)
    elif method == MultiTurnOptimizationMethod.KV_CACHE:
        return _prepare_multiturn_kv_cache_inputs(processing_class, messages, tools, max_model_len)
    elif method == MultiTurnOptimizationMethod.SEQUENCE_PACKING:
        return _prepare_multiturn_sequence_packing_inputs(processing_class, messages, tools, max_model_len)
    else:
        raise ValueError(f"Unknown optimization method: {method}")


def _prepare_multiturn_sdpa_inputs(
    processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin],
    messages: List[Dict],
    tools: Optional[List[Dict]] = None,
    max_model_len: int = 32768,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """
    Prepare inputs for multi-turn SDPA optimization.
    
    This function creates a strategically duplicated sequence where assistant messages
    appear twice - once with reasoning tokens and once without. The custom attention
    mask ensures proper visibility patterns for training while maintaining inference
    consistency.
    
    Only works with supported tokenizers (currently Qwen2).
    
    Args:
        processing_class: Tokenizer or processor for the model
        messages: List of conversation messages (dict format)
        tools: Optional list of tool schemas  
        max_model_len: Maximum model context length
        
    Returns:
        Tuple of (input_ids, position_ids, attention_mask, assistant_ranges)
        - input_ids: The strategically duplicated token sequence
        - position_ids: Position IDs for the sequence
        - attention_mask: Custom 4D attention mask for SDPA
        - assistant_ranges: List of (start, end) indices for assistant messages
    """
    
    # Build the strategically duplicated sequence
    input_ids = []
    assistant_ranges = []
    current_pos = 0
    
    # Process each message in the conversation
    for i, message in enumerate(messages):
        if message.get("role") == "assistant":
            # For assistant messages, we need both reasoning and non-reasoning versions
            
            # First, get the message with reasoning (for training)
            messages_with_reasoning = messages[:i+1]
            reasoning_tokens = processing_class.apply_chat_template(
                messages_with_reasoning,
                tools=tools,
                tokenize=True,
                add_generation_prompt=False
            )
            
            # Get the message without reasoning (for inference consistency)
            # Use enable_thinking=False to strip reasoning content
            clean_message = dict(message)
            messages_clean = messages[:i] + [clean_message]
            clean_tokens = processing_class.apply_chat_template(
                messages_clean,
                tools=tools,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False  # This strips reasoning tokens for Qwen2
            )
            
            # Calculate the delta tokens for this assistant message
            prev_tokens = processing_class.apply_chat_template(
                messages[:i],
                tools=tools,
                tokenize=True,
                add_generation_prompt=False
            ) if i > 0 else []
            
            reasoning_delta = reasoning_tokens[len(prev_tokens):]
            clean_delta = clean_tokens[len(prev_tokens):]
            
            # Add both versions to the sequence
            reasoning_start = current_pos
            input_ids.extend(reasoning_delta)
            current_pos += len(reasoning_delta)
            reasoning_end = current_pos
            
            clean_start = current_pos  
            input_ids.extend(clean_delta)
            current_pos += len(clean_delta)
            clean_end = current_pos
            
            assistant_ranges.append((reasoning_start, reasoning_end, clean_start, clean_end))
            
        else:
            # For non-assistant messages, add normally
            messages_so_far = messages[:i+1]
            tokens = processing_class.apply_chat_template(
                messages_so_far,
                tools=tools,
                tokenize=True,
                add_generation_prompt=False
            )
            
            prev_tokens = processing_class.apply_chat_template(
                messages[:i],
                tools=tools,
                tokenize=True,
                add_generation_prompt=False
            ) if i > 0 else []
            
            delta_tokens = tokens[len(prev_tokens):]
            input_ids.extend(delta_tokens)
            current_pos += len(delta_tokens)
    
    # Convert to tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    seq_len = len(input_ids)
    
    # Create position IDs
    position_ids = torch.arange(seq_len, dtype=torch.long)
    
    # Create custom attention mask
    attention_mask = _create_multiturn_attention_mask(seq_len, assistant_ranges)
    
    return input_ids, position_ids, attention_mask, assistant_ranges


def _create_multiturn_attention_mask(
    seq_len: int, 
    assistant_ranges: List[Tuple[int, int, int, int]]
) -> torch.Tensor:
    """
    Create a custom 4D attention mask for multi-turn SDPA optimization.
    
    The mask ensures that:
    1. Reasoning versions of assistant messages can see all previous context
    2. Clean versions can only see non-reasoning context
    3. Future messages can only see clean versions of past assistant messages
    
    Args:
        seq_len: Total sequence length
        assistant_ranges: List of (reasoning_start, reasoning_end, clean_start, clean_end) tuples
        
    Returns:
        4D attention mask tensor of shape (1, 1, seq_len, seq_len)
    """
    # Start with causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    mask = ~mask  # Invert to get lower triangular (can attend to previous positions)
    
    # Apply multi-turn constraints
    for reasoning_start, reasoning_end, clean_start, clean_end in assistant_ranges:
        # Clean version should not see reasoning content from any assistant messages
        for other_r_start, other_r_end, _, _ in assistant_ranges:
            mask[clean_start:clean_end, other_r_start:other_r_end] = False
            
        # Future tokens should not see reasoning versions
        mask[clean_end:, reasoning_start:reasoning_end] = False
    
    # Convert to SDPA format: (batch, heads, seq_len, seq_len)
    # Use float with -inf for masked positions (SDPA convention)
    float_mask = torch.zeros(seq_len, seq_len)
    float_mask = float_mask.masked_fill(~mask, float('-inf'))
    
    return float_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions


def apply_multiturn_sdpa_mask(
    query: torch.Tensor,
    key: torch.Tensor, 
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Apply multi-turn optimized SDPA with custom attention mask.
    
    Args:
        query: Query tensor of shape (batch, heads, seq_len, head_dim)
        key: Key tensor of shape (batch, heads, seq_len, head_dim)  
        value: Value tensor of shape (batch, heads, seq_len, head_dim)
        attention_mask: Custom attention mask from prepare_multiturn_sdpa_inputs
        dropout_p: Dropout probability
        is_causal: Whether to apply causal masking (should be False since we have custom mask)
        
    Returns:
        Attention output tensor
    """
    return torch.nn.functional.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attention_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,  # Set to False since we provide custom mask
    )


def extract_assistant_logits(
    logits: torch.Tensor,
    assistant_ranges: List[Tuple[int, int, int, int]],
    use_clean_version: bool = True
) -> torch.Tensor:
    """
    Extract logits for assistant messages from the multi-turn optimized sequence.
    
    Args:
        logits: Full sequence logits of shape (batch, seq_len, vocab_size)
        assistant_ranges: Assistant message ranges from prepare_multiturn_sdpa_inputs
        use_clean_version: Whether to use clean (non-reasoning) version of assistant messages
        
    Returns:
        Assistant message logits
    """
    assistant_logits = []
    
    for reasoning_start, reasoning_end, clean_start, clean_end in assistant_ranges:
        if use_clean_version:
            # Use clean version for consistency with inference
            assistant_logits.append(logits[:, clean_start:clean_end, :])
        else:
            # Use reasoning version for training
            assistant_logits.append(logits[:, reasoning_start:reasoning_end, :])
    
    if assistant_logits:
        return torch.cat(assistant_logits, dim=1)
    else:
        return torch.empty(logits.shape[0], 0, logits.shape[2], device=logits.device)


# ============ Placeholder implementations for other optimization methods ============

def _prepare_multiturn_naive_inputs(
    processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin],
    messages: List[Dict],
    tools: Optional[List[Dict]] = None,
    max_model_len: int = 32768,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """
    Prepare inputs for naive turn-by-turn forward passes.
    
    This method processes each turn individually with separate forward passes,
    applying the same context modifications used during inference.
    
    TODO: Implement the naive approach that processes turns individually.
    """
    # Placeholder implementation
    raise NotImplementedError("Naive multi-turn optimization not yet implemented")


def _prepare_multiturn_flex_attention_inputs(
    processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin],
    messages: List[Dict],
    tools: Optional[List[Dict]] = None,
    max_model_len: int = 32768,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """
    Prepare inputs for FlexAttention-based multi-turn optimization.
    
    This method uses FlexAttention for handling ultra-long conversations
    with custom attention patterns.
    
    TODO: Implement FlexAttention support for ultra-long conversations.
    """
    # Placeholder implementation
    raise NotImplementedError("FlexAttention multi-turn optimization not yet implemented")


def _prepare_multiturn_kv_cache_inputs(
    processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin],
    messages: List[Dict],
    tools: Optional[List[Dict]] = None,
    max_model_len: int = 32768,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """
    Prepare inputs for KV cache acceleration with cropping.
    
    This method leverages key-value caching to reduce redundant computation,
    caches KV states, crops them appropriately, and rebuilds without reasoning.
    
    TODO: Implement KV cache acceleration with proper cropping logic.
    """
    # Placeholder implementation
    raise NotImplementedError("KV cache multi-turn optimization not yet implemented")


def _prepare_multiturn_sequence_packing_inputs(
    processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin],
    messages: List[Dict],
    tools: Optional[List[Dict]] = None,
    max_model_len: int = 32768,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """
    Prepare inputs for sequence packing optimization.
    
    This method concatenates samples into a single sequence to eliminate padding
    and improve computational efficiency.
    
    TODO: Implement sequence packing with proper attention mask handling.
    """
    # Placeholder implementation
    raise NotImplementedError("Sequence packing multi-turn optimization not yet implemented")


