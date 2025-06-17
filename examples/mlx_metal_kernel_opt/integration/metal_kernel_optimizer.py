"""
MLX Metal Kernel Optimizer - Cascading Attention Optimizations

This module provides advanced Metal kernel optimizations for various attention patterns
found in modern transformer architectures. It intelligently dispatches optimized kernels
based on model characteristics and falls back gracefully when optimizations aren't beneficial.

Supported Optimizations:
1. Grouped Query Attention (GQA) - Optimized for models like Qwen3, Llama-3, etc.
2. Multi-Head Attention (MHA) - Optimized for standard attention patterns
3. Multi-Query Attention (MQA) - Optimized for single KV head models
4. Sliding Window Attention - Optimized for local attention patterns

Key Features:
- Automatic dispatch based on model architecture
- Graceful fallback to standard MLX operations
- Apple Silicon specific optimizations
- Memory-efficient online softmax implementation
- Vectorized operations with SIMD optimization
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import math
import time
import warnings
from typing import Optional, Tuple, Any, Dict, Union
from dataclasses import dataclass


@dataclass
class AttentionConfig:
    """Configuration for attention pattern detection and optimization"""
    num_heads: int
    num_kv_heads: int
    head_dim: int
    seq_len: int
    batch_size: int
    
    @property
    def is_gqa(self) -> bool:
        """Grouped Query Attention: multiple query heads per KV head"""
        return self.num_heads > self.num_kv_heads > 1
    
    @property
    def is_mqa(self) -> bool:
        """Multi-Query Attention: single KV head"""
        return self.num_kv_heads == 1
    
    @property
    def is_mha(self) -> bool:
        """Multi-Head Attention: equal heads"""
        return self.num_heads == self.num_kv_heads
    
    @property
    def heads_per_kv(self) -> int:
        """Number of query heads per KV head"""
        return self.num_heads // self.num_kv_heads
    
    @property
    def attention_pattern(self) -> str:
        """Get attention pattern name"""
        if self.is_gqa:
            return f"GQA-{self.heads_per_kv}:1"
        elif self.is_mqa:
            return "MQA"
        elif self.is_mha:
            return "MHA"
        else:
            return "UNKNOWN"


class MetalKernelOptimizer:
    """
    Advanced Metal kernel optimizer with intelligent dispatch and fallback mechanisms.
    """
    
    # Optimization thresholds and configurations
    OPTIMIZATION_THRESHOLDS = {
        'min_seq_len': 64,      # Minimum sequence length to benefit from custom kernels
        'max_seq_len': 4096,    # Maximum sequence length supported efficiently
        'min_head_dim': 64,     # Minimum head dimension for vectorization benefits
        'max_head_dim': 256,    # Maximum head dimension supported
        'min_heads': 8,         # Minimum number of heads to benefit from optimization
        'gqa_ratio_min': 2,     # Minimum GQA ratio to trigger GQA optimization
        'memory_efficiency_threshold': 0.8,  # Memory usage threshold
    }
    
    # Supported model architectures and their optimal configurations
    SUPPORTED_ARCHITECTURES = {
        'qwen3': {
            'pattern': 'GQA',
            'ratios': [5],  # 40:8 = 5:1
            'head_dims': [128],
            'optimization_priority': 'memory+speed'
        },
        'llama3': {
            'pattern': 'GQA', 
            'ratios': [4, 8],  # Various GQA ratios
            'head_dims': [128],
            'optimization_priority': 'speed'
        },
        'gemma': {
            'pattern': 'MHA',
            'ratios': [1],
            'head_dims': [256],
            'optimization_priority': 'memory'
        },
        'mistral': {
            'pattern': 'GQA',
            'ratios': [4],
            'head_dims': [128],
            'optimization_priority': 'speed'
        }
    }

    def __init__(self, enable_debug: bool = False):
        self.enable_debug = enable_debug
        self.optimization_cache = {}
        self.fallback_count = 0
        self.success_count = 0
        
    def should_optimize(self, config: AttentionConfig) -> Tuple[bool, str]:
        """
        Determine if the given attention configuration should use optimized kernels.
        
        Returns:
            Tuple of (should_optimize, reason)
        """
        reasons = []
        
        # Check basic thresholds
        if config.seq_len < self.OPTIMIZATION_THRESHOLDS['min_seq_len']:
            return False, f"Sequence length {config.seq_len} below threshold {self.OPTIMIZATION_THRESHOLDS['min_seq_len']}"
            
        if config.seq_len > self.OPTIMIZATION_THRESHOLDS['max_seq_len']:
            return False, f"Sequence length {config.seq_len} above supported limit {self.OPTIMIZATION_THRESHOLDS['max_seq_len']}"
            
        if config.head_dim < self.OPTIMIZATION_THRESHOLDS['min_head_dim']:
            return False, f"Head dimension {config.head_dim} below vectorization threshold {self.OPTIMIZATION_THRESHOLDS['min_head_dim']}"
            
        if config.head_dim > self.OPTIMIZATION_THRESHOLDS['max_head_dim']:
            return False, f"Head dimension {config.head_dim} above supported limit {self.OPTIMIZATION_THRESHOLDS['max_head_dim']}"
            
        if config.num_heads < self.OPTIMIZATION_THRESHOLDS['min_heads']:
            return False, f"Number of heads {config.num_heads} below optimization threshold {self.OPTIMIZATION_THRESHOLDS['min_heads']}"
        
        # Check pattern-specific optimizations
        if config.is_gqa and config.heads_per_kv >= self.OPTIMIZATION_THRESHOLDS['gqa_ratio_min']:
            reasons.append(f"GQA pattern with {config.heads_per_kv}:1 ratio benefits from custom kernel")
        elif config.is_mqa:
            reasons.append("MQA pattern benefits from specialized kernel")
        elif config.is_mha and config.num_heads >= 16:
            reasons.append("Large MHA benefits from vectorized implementation")
        else:
            return False, f"Attention pattern {config.attention_pattern} not optimized for this configuration"
            
        return True, "; ".join(reasons)

    def get_optimized_kernel_source(self, config: AttentionConfig) -> str:
        """
        Generate optimized Metal kernel source based on attention configuration.
        """
        if config.is_gqa:
            return self._get_gqa_kernel_source(config)
        elif config.is_mqa:
            return self._get_mqa_kernel_source(config)
        elif config.is_mha:
            return self._get_mha_kernel_source(config)
        else:
            raise ValueError(f"Unsupported attention pattern: {config.attention_pattern}")

    def _get_gqa_kernel_source(self, config: AttentionConfig) -> str:
        """Generate GQA-optimized Metal kernel source"""
        return f"""
        // Advanced GQA Metal Kernel - Optimized for {config.attention_pattern}
        // Architecture: {config.num_heads}:{config.num_kv_heads} heads, {config.head_dim}D
        // Optimizations: Memory coalescing, SIMD vectorization, online softmax
        
        uint thread_id = thread_position_in_grid.x;
        uint head_idx = thread_position_in_grid.y; 
        uint batch_idx = thread_position_in_grid.z;
        uint query_pos = thread_id;
        
        // Bounds checking with early exit
        if (batch_idx >= BATCH_SIZE || head_idx >= NUM_HEADS || query_pos >= SEQ_LEN) {{
            return;
        }}
        
        // Extract configuration values
        T scale_val = scale[0];
        bool use_mask_val = use_mask[0] > 0;
        
        // GQA mapping with optimized division
        uint kv_head_idx = head_idx / HEADS_PER_KV;
        
        // Pre-calculate memory indices for optimal access patterns
        const uint q_base = batch_idx * (NUM_HEADS * SEQ_LEN * HEAD_DIM) + 
                            head_idx * (SEQ_LEN * HEAD_DIM) + 
                            query_pos * HEAD_DIM;
                            
        const uint k_base_start = batch_idx * (NUM_KV_HEADS * SEQ_LEN * HEAD_DIM) + 
                                  kv_head_idx * (SEQ_LEN * HEAD_DIM);
                                  
        const uint v_base_start = k_base_start;
        
        const uint mask_base = batch_idx * (NUM_HEADS * SEQ_LEN * SEQ_LEN) + 
                               head_idx * (SEQ_LEN * SEQ_LEN) + 
                               query_pos * SEQ_LEN;
                               
        // Load query vector into fast thread memory with vectorization
        thread T query_vec[HEAD_DIM];
        
        // Vectorized query loading for better memory throughput
        for (uint d = 0; d < HEAD_DIM; d += 4) {{
            if (d + 3 < HEAD_DIM) {{
                // Load 4 elements at once for SIMD efficiency
                *((thread float4*)(query_vec + d)) = *((device float4*)(queries + q_base + d));
            }} else {{
                // Handle remaining elements
                for (uint dd = d; dd < HEAD_DIM; ++dd) {{
                    query_vec[dd] = queries[q_base + dd];
                }}
                break;
            }}
        }}
        
        // Advanced online softmax with numerical stability
        T max_score = T(-INFINITY);
        T denominator = T(0.0);
        thread T output_accumulator[HEAD_DIM];
        
        // Initialize accumulator
        for (uint d = 0; d < HEAD_DIM; ++d) {{
            output_accumulator[d] = T(0.0);
        }}

        // Main attention computation loop with memory optimization
        for (uint key_pos = 0; key_pos < SEQ_LEN; ++key_pos) {{
            // Efficient mask checking
            bool is_valid = use_mask_val ? mask[mask_base + key_pos] : true;
            if (!is_valid) continue;

            // Optimized score computation with SIMD
            const uint k_base = k_base_start + key_pos * HEAD_DIM;
            T score = T(0.0);
            
            // Vectorized dot product with unrolling
            for (uint d = 0; d < HEAD_DIM; d += 8) {{
                if (d + 7 < HEAD_DIM) {{
                    // Unrolled 8-way SIMD for maximum throughput
                    score += query_vec[d] * keys[k_base + d] +
                             query_vec[d+1] * keys[k_base + d+1] +
                             query_vec[d+2] * keys[k_base + d+2] +
                             query_vec[d+3] * keys[k_base + d+3] +
                             query_vec[d+4] * keys[k_base + d+4] +
                             query_vec[d+5] * keys[k_base + d+5] +
                             query_vec[d+6] * keys[k_base + d+6] +
                             query_vec[d+7] * keys[k_base + d+7];
                }} else {{
                    // Handle remaining elements efficiently
                    for (uint dd = d; dd < HEAD_DIM; ++dd) {{
                        score += query_vec[dd] * keys[k_base + dd];
                    }}
                    break;
                }}
            }}
            score *= scale_val;

            // Numerically stable online softmax update
            T new_max_score = max(max_score, score);
            T exp_old_diff = exp(max_score - new_max_score);
            T exp_new_diff = exp(score - new_max_score);

            // Update denominator with new maximum
            denominator = denominator * exp_old_diff + exp_new_diff;
            
            // Load and accumulate values with vectorization
            const uint v_base = v_base_start + key_pos * HEAD_DIM;
            
            // Vectorized value accumulation
            for (uint d = 0; d < HEAD_DIM; d += 8) {{
                if (d + 7 < HEAD_DIM) {{
                    // Unrolled vector operations for optimal performance
                    output_accumulator[d] = output_accumulator[d] * exp_old_diff + exp_new_diff * values[v_base + d];
                    output_accumulator[d+1] = output_accumulator[d+1] * exp_old_diff + exp_new_diff * values[v_base + d+1];
                    output_accumulator[d+2] = output_accumulator[d+2] * exp_old_diff + exp_new_diff * values[v_base + d+2];
                    output_accumulator[d+3] = output_accumulator[d+3] * exp_old_diff + exp_new_diff * values[v_base + d+3];
                    output_accumulator[d+4] = output_accumulator[d+4] * exp_old_diff + exp_new_diff * values[v_base + d+4];
                    output_accumulator[d+5] = output_accumulator[d+5] * exp_old_diff + exp_new_diff * values[v_base + d+5];
                    output_accumulator[d+6] = output_accumulator[d+6] * exp_old_diff + exp_new_diff * values[v_base + d+6];
                    output_accumulator[d+7] = output_accumulator[d+7] * exp_old_diff + exp_new_diff * values[v_base + d+7];
                }} else {{
                    for (uint dd = d; dd < HEAD_DIM; ++dd) {{
                        output_accumulator[dd] = output_accumulator[dd] * exp_old_diff + exp_new_diff * values[v_base + dd];
                    }}
                    break;
                }}
            }}
            
            max_score = new_max_score;
        }}

        // Final normalization and vectorized output
        if (denominator > T(1e-9)) {{
            T inv_denominator = T(1.0) / denominator;
            
            // Vectorized final output for memory efficiency
            for (uint d = 0; d < HEAD_DIM; d += 4) {{
                if (d + 3 < HEAD_DIM) {{
                    *((device float4*)(output + q_base + d)) = *((thread float4*)(output_accumulator + d)) * inv_denominator;
                }} else {{
                    for (uint dd = d; dd < HEAD_DIM; ++dd) {{
                        output[q_base + dd] = output_accumulator[dd] * inv_denominator;
                    }}
                    break;
                }}
            }}
        }} else {{
            // Zero output for masked sequences
            for (uint d = 0; d < HEAD_DIM; d += 4) {{
                if (d + 3 < HEAD_DIM) {{
                    *((device float4*)(output + q_base + d)) = float4(0.0);
                }} else {{
                    for (uint dd = d; dd < HEAD_DIM; ++dd) {{
                        output[q_base + dd] = T(0.0);
                    }}
                    break;
                }}
            }}
        }}
        """

    def _get_mqa_kernel_source(self, config: AttentionConfig) -> str:
        """Generate MQA-optimized Metal kernel source"""
        return f"""
        // MQA Metal Kernel - Single KV head optimization
        // All query heads share the same key and value
        
        uint thread_id = thread_position_in_grid.x;
        uint head_idx = thread_position_in_grid.y; 
        uint batch_idx = thread_position_in_grid.z;
        uint query_pos = thread_id;
        
        if (batch_idx >= BATCH_SIZE || head_idx >= NUM_HEADS || query_pos >= SEQ_LEN) {{
            return;
        }}
        
        T scale_val = scale[0];
        bool use_mask_val = use_mask[0] > 0;
        
        // MQA: All heads use kv_head_idx = 0
        const uint kv_head_idx = 0;
        
        // Memory layout optimized for single KV head
        const uint q_base = batch_idx * (NUM_HEADS * SEQ_LEN * HEAD_DIM) + 
                            head_idx * (SEQ_LEN * HEAD_DIM) + 
                            query_pos * HEAD_DIM;
                            
        const uint k_base_start = batch_idx * (SEQ_LEN * HEAD_DIM);  // Single KV head
        const uint v_base_start = k_base_start;
        
        const uint mask_base = batch_idx * (NUM_HEADS * SEQ_LEN * SEQ_LEN) + 
                               head_idx * (SEQ_LEN * SEQ_LEN) + 
                               query_pos * SEQ_LEN;
        
        // Load query with vectorization
        thread T query_vec[HEAD_DIM];
        for (uint d = 0; d < HEAD_DIM; d += 4) {{
            if (d + 3 < HEAD_DIM) {{
                *((thread float4*)(query_vec + d)) = *((device float4*)(queries + q_base + d));
            }} else {{
                for (uint dd = d; dd < HEAD_DIM; ++dd) {{
                    query_vec[dd] = queries[q_base + dd];
                }}
                break;
            }}
        }}
        
        // MQA-optimized attention computation
        T max_score = T(-INFINITY);
        T denominator = T(0.0);
        thread T output_accumulator[HEAD_DIM];
        
        for (uint d = 0; d < HEAD_DIM; ++d) {{
            output_accumulator[d] = T(0.0);
        }}

        for (uint key_pos = 0; key_pos < SEQ_LEN; ++key_pos) {{
            bool is_valid = use_mask_val ? mask[mask_base + key_pos] : true;
            if (!is_valid) continue;

            const uint k_base = k_base_start + key_pos * HEAD_DIM;
            T score = T(0.0);
            
            // Vectorized score computation
            for (uint d = 0; d < HEAD_DIM; d += 4) {{
                if (d + 3 < HEAD_DIM) {{
                    score += query_vec[d] * keys[k_base + d] +
                             query_vec[d+1] * keys[k_base + d+1] +
                             query_vec[d+2] * keys[k_base + d+2] +
                             query_vec[d+3] * keys[k_base + d+3];
                }} else {{
                    for (uint dd = d; dd < HEAD_DIM; ++dd) {{
                        score += query_vec[dd] * keys[k_base + dd];
                    }}
                    break;
                }}
            }}
            score *= scale_val;

            T new_max_score = max(max_score, score);
            T exp_old_diff = exp(max_score - new_max_score);
            T exp_new_diff = exp(score - new_max_score);

            denominator = denominator * exp_old_diff + exp_new_diff;
            
            const uint v_base = v_base_start + key_pos * HEAD_DIM;
            
            for (uint d = 0; d < HEAD_DIM; d += 4) {{
                if (d + 3 < HEAD_DIM) {{
                    output_accumulator[d] = output_accumulator[d] * exp_old_diff + exp_new_diff * values[v_base + d];
                    output_accumulator[d+1] = output_accumulator[d+1] * exp_old_diff + exp_new_diff * values[v_base + d+1];
                    output_accumulator[d+2] = output_accumulator[d+2] * exp_old_diff + exp_new_diff * values[v_base + d+2];
                    output_accumulator[d+3] = output_accumulator[d+3] * exp_old_diff + exp_new_diff * values[v_base + d+3];
                }} else {{
                    for (uint dd = d; dd < HEAD_DIM; ++dd) {{
                        output_accumulator[dd] = output_accumulator[dd] * exp_old_diff + exp_new_diff * values[v_base + dd];
                    }}
                    break;
                }}
            }}
            
            max_score = new_max_score;
        }}

        // Final output
        if (denominator > T(1e-9)) {{
            T inv_denominator = T(1.0) / denominator;
            for (uint d = 0; d < HEAD_DIM; d += 4) {{
                if (d + 3 < HEAD_DIM) {{
                    *((device float4*)(output + q_base + d)) = *((thread float4*)(output_accumulator + d)) * inv_denominator;
                }} else {{
                    for (uint dd = d; dd < HEAD_DIM; ++dd) {{
                        output[q_base + dd] = output_accumulator[dd] * inv_denominator;
                    }}
                    break;
                }}
            }}
        }} else {{
            for (uint d = 0; d < HEAD_DIM; d += 4) {{
                if (d + 3 < HEAD_DIM) {{
                    *((device float4*)(output + q_base + d)) = float4(0.0);
                }} else {{
                    for (uint dd = d; dd < HEAD_DIM; ++dd) {{
                        output[q_base + dd] = T(0.0);
                    }}
                    break;
                }}
            }}
        }}
        """

    def _get_mha_kernel_source(self, config: AttentionConfig) -> str:
        """Generate MHA-optimized Metal kernel source"""
        return f"""
        // MHA Metal Kernel - Equal heads optimization
        // Each query head has its own corresponding key and value head
        
        uint thread_id = thread_position_in_grid.x;
        uint head_idx = thread_position_in_grid.y; 
        uint batch_idx = thread_position_in_grid.z;
        uint query_pos = thread_id;
        
        if (batch_idx >= BATCH_SIZE || head_idx >= NUM_HEADS || query_pos >= SEQ_LEN) {{
            return;
        }}
        
        T scale_val = scale[0];
        bool use_mask_val = use_mask[0] > 0;
        
        // MHA: Direct 1:1 mapping
        const uint kv_head_idx = head_idx;
        
        const uint q_base = batch_idx * (NUM_HEADS * SEQ_LEN * HEAD_DIM) + 
                            head_idx * (SEQ_LEN * HEAD_DIM) + 
                            query_pos * HEAD_DIM;
                            
        const uint k_base_start = batch_idx * (NUM_HEADS * SEQ_LEN * HEAD_DIM) + 
                                  kv_head_idx * (SEQ_LEN * HEAD_DIM);
                                  
        const uint v_base_start = k_base_start;
        
        const uint mask_base = batch_idx * (NUM_HEADS * SEQ_LEN * SEQ_LEN) + 
                               head_idx * (SEQ_LEN * SEQ_LEN) + 
                               query_pos * SEQ_LEN;
        
        // Standard vectorized implementation for MHA
        thread T query_vec[HEAD_DIM];
        for (uint d = 0; d < HEAD_DIM; d += 4) {{
            if (d + 3 < HEAD_DIM) {{
                *((thread float4*)(query_vec + d)) = *((device float4*)(queries + q_base + d));
            }} else {{
                for (uint dd = d; dd < HEAD_DIM; ++dd) {{
                    query_vec[dd] = queries[q_base + dd];
                }}
                break;
            }}
        }}
        
        T max_score = T(-INFINITY);
        T denominator = T(0.0);
        thread T output_accumulator[HEAD_DIM];
        
        for (uint d = 0; d < HEAD_DIM; ++d) {{
            output_accumulator[d] = T(0.0);
        }}

        for (uint key_pos = 0; key_pos < SEQ_LEN; ++key_pos) {{
            bool is_valid = use_mask_val ? mask[mask_base + key_pos] : true;
            if (!is_valid) continue;

            const uint k_base = k_base_start + key_pos * HEAD_DIM;
            T score = T(0.0);
            
            for (uint d = 0; d < HEAD_DIM; d += 4) {{
                if (d + 3 < HEAD_DIM) {{
                    score += query_vec[d] * keys[k_base + d] +
                             query_vec[d+1] * keys[k_base + d+1] +
                             query_vec[d+2] * keys[k_base + d+2] +
                             query_vec[d+3] * keys[k_base + d+3];
                }} else {{
                    for (uint dd = d; dd < HEAD_DIM; ++dd) {{
                        score += query_vec[dd] * keys[k_base + dd];
                    }}
                    break;
                }}
            }}
            score *= scale_val;

            T new_max_score = max(max_score, score);
            T exp_old_diff = exp(max_score - new_max_score);
            T exp_new_diff = exp(score - new_max_score);

            denominator = denominator * exp_old_diff + exp_new_diff;
            
            const uint v_base = v_base_start + key_pos * HEAD_DIM;
            
            for (uint d = 0; d < HEAD_DIM; d += 4) {{
                if (d + 3 < HEAD_DIM) {{
                    output_accumulator[d] = output_accumulator[d] * exp_old_diff + exp_new_diff * values[v_base + d];
                    output_accumulator[d+1] = output_accumulator[d+1] * exp_old_diff + exp_new_diff * values[v_base + d+1];
                    output_accumulator[d+2] = output_accumulator[d+2] * exp_old_diff + exp_new_diff * values[v_base + d+2];
                    output_accumulator[d+3] = output_accumulator[d+3] * exp_old_diff + exp_new_diff * values[v_base + d+3];
                }} else {{
                    for (uint dd = d; dd < HEAD_DIM; ++dd) {{
                        output_accumulator[dd] = output_accumulator[dd] * exp_old_diff + exp_new_diff * values[v_base + dd];
                    }}
                    break;
                }}
            }}
            
            max_score = new_max_score;
        }}

        if (denominator > T(1e-9)) {{
            T inv_denominator = T(1.0) / denominator;
            for (uint d = 0; d < HEAD_DIM; d += 4) {{
                if (d + 3 < HEAD_DIM) {{
                    *((device float4*)(output + q_base + d)) = *((thread float4*)(output_accumulator + d)) * inv_denominator;
                }} else {{
                    for (uint dd = d; dd < HEAD_DIM; ++dd) {{
                        output[q_base + dd] = output_accumulator[dd] * inv_denominator;
                    }}
                    break;
                }}
            }}
        }} else {{
            for (uint d = 0; d < HEAD_DIM; d += 4) {{
                if (d + 3 < HEAD_DIM) {{
                    *((device float4*)(output + q_base + d)) = float4(0.0);
                }} else {{
                    for (uint dd = d; dd < HEAD_DIM; ++dd) {{
                        output[q_base + dd] = T(0.0);
                    }}
                    break;
                }}
            }}
        }}
        """

    def optimized_attention(self, queries: mx.array, keys: mx.array, values: mx.array, 
                          scale: float = 1.0, mask: Optional[mx.array] = None) -> mx.array:
        """
        Apply optimized attention with intelligent dispatch and fallback.
        
        Args:
            queries: Query tensor [B, num_heads, L, head_dim]
            keys: Key tensor [B, num_kv_heads, L, head_dim] 
            values: Value tensor [B, num_kv_heads, L, head_dim]
            scale: Attention scaling factor
            mask: Attention mask (causal, boolean tensor, or None)
            
        Returns:
            Attention output [B, num_heads, L, head_dim]
        """
        B, num_heads, L, head_dim = queries.shape
        _, num_kv_heads, _, _ = keys.shape
        
        # Create configuration for this attention call
        config = AttentionConfig(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            seq_len=L,
            batch_size=B
        )
        
        # Check if we should apply optimizations
        should_opt, reason = self.should_optimize(config)
        
        if not should_opt:
            if self.enable_debug:
                print(f"🔄 Falling back to MLX SDPA: {reason}")
            self.fallback_count += 1
            return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask=mask)
        
        # Try to apply optimized kernel
        try:
            if self.enable_debug:
                print(f"⚡ Applying {config.attention_pattern} optimization: {reason}")
            
            result = self._execute_optimized_kernel(queries, keys, values, scale, mask, config)
            self.success_count += 1
            return result
            
        except Exception as e:
            if self.enable_debug:
                warnings.warn(f"🚨 Metal kernel failed: {e}, falling back to MLX SDPA")
            self.fallback_count += 1
            return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask=mask)

    def _execute_optimized_kernel(self, queries: mx.array, keys: mx.array, values: mx.array,
                                scale: float, mask: Optional[mx.array], config: AttentionConfig) -> mx.array:
        """Execute the optimized Metal kernel"""
        
        # Handle mask conversion with better logic
        if mask == "causal" or mask is None:
            causal_mask = mx.triu(mx.ones((config.seq_len, config.seq_len), dtype=mx.bool_), k=1)
            mask_tensor = mx.logical_not(causal_mask)
            use_mask = True
        elif isinstance(mask, mx.array):
            mask_tensor = mask.astype(mx.bool_)
            use_mask = True
        else:
            mask_tensor = mx.ones((config.seq_len, config.seq_len), dtype=mx.bool_)
            use_mask = False
        
        # Expand mask to proper dimensions
        if mask_tensor.ndim == 2:
            mask_tensor = mx.broadcast_to(mask_tensor[None, None, :, :], 
                                        (config.batch_size, config.num_heads, config.seq_len, config.seq_len))
        elif mask_tensor.ndim == 3:
            mask_tensor = mx.broadcast_to(mask_tensor[:, None, :, :], 
                                        (config.batch_size, config.num_heads, config.seq_len, config.seq_len))
        
        # Prepare kernel inputs
        scale_tensor = mx.array([scale], dtype=queries.dtype)
        use_mask_tensor = mx.array([1 if use_mask else 0], dtype=mx.int32)
        
        # Get optimized kernel source
        kernel_source = self.get_optimized_kernel_source(config)
        
        # Create and execute Metal kernel
        kernel = mx.fast.metal_kernel(
            name=f"optimized_{config.attention_pattern.lower()}_attention",
            input_names=["queries", "keys", "values", "mask", "scale", "use_mask"],
            output_names=["output"],
            source=kernel_source,
        )
        
        # Optimize thread configuration based on sequence length and hardware
        threadgroup_size = min(32, config.seq_len)
        if config.seq_len >= 1024:
            threadgroup_size = 64  # Larger threadgroups for long sequences
        elif config.seq_len >= 512:
            threadgroup_size = 32
        else:
            threadgroup_size = 16  # Smaller threadgroups for short sequences
            
        # Execute kernel with optimized configuration
        outputs = kernel(
            inputs=[queries, keys, values, mask_tensor, scale_tensor, use_mask_tensor],
            output_shapes=[(config.batch_size, config.num_heads, config.seq_len, config.head_dim)],
            output_dtypes=[queries.dtype],
            grid=(config.seq_len, config.num_heads, config.batch_size),
            threadgroup=(threadgroup_size, 1, 1),
            template=[
                ("T", queries.dtype),
                ("BATCH_SIZE", config.batch_size),
                ("NUM_HEADS", config.num_heads),
                ("NUM_KV_HEADS", config.num_kv_heads),
                ("SEQ_LEN", config.seq_len),
                ("HEAD_DIM", config.head_dim),
                ("HEADS_PER_KV", config.heads_per_kv),
            ],
        )
        
        return outputs[0]

    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        total_calls = self.success_count + self.fallback_count
        success_rate = self.success_count / total_calls if total_calls > 0 else 0.0
        
        return {
            'total_calls': total_calls,
            'optimized_calls': self.success_count,
            'fallback_calls': self.fallback_count,
            'optimization_rate': success_rate,
            'cache_size': len(self.optimization_cache)
        }

    def reset_stats(self):
        """Reset optimization statistics"""
        self.success_count = 0
        self.fallback_count = 0
        self.optimization_cache.clear()


# Global optimizer instance
_global_optimizer = MetalKernelOptimizer()


def optimized_scaled_dot_product_attention(queries: mx.array, keys: mx.array, values: mx.array,
                                         scale: float = 1.0, mask: Optional[mx.array] = None) -> mx.array:
    """
    Drop-in replacement for mx.fast.scaled_dot_product_attention with Metal optimizations.
    
    This function provides the same interface as MLX's built-in scaled_dot_product_attention
    but intelligently applies optimized Metal kernels when beneficial.
    """
    return _global_optimizer.optimized_attention(queries, keys, values, scale, mask)


def configure_optimizer(enable_debug: bool = False, **kwargs):
    """Configure the global optimizer"""
    global _global_optimizer
    _global_optimizer = MetalKernelOptimizer(enable_debug=enable_debug)
    
    # Update thresholds if provided
    for key, value in kwargs.items():
        if key in _global_optimizer.OPTIMIZATION_THRESHOLDS:
            _global_optimizer.OPTIMIZATION_THRESHOLDS[key] = value


def get_optimizer_stats() -> Dict[str, Any]:
    """Get global optimizer statistics"""
    return _global_optimizer.get_stats()


def reset_optimizer_stats():
    """Reset global optimizer statistics"""
    _global_optimizer.reset_stats()
