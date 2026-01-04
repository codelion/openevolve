# OpenEvolve Metal Kernel Optimization: Automated Discovery of Custom GPU Kernels for Transformer Attention

**Evolutionary Optimization of Apple Silicon Metal Kernels for Grouped Query Attention in Qwen3-0.6B**

## Abstract

This example demonstrates evolutionary code optimization for discovering custom Apple Silicon Metal GPU kernels for transformer attention. It targets Grouped Query Attention (GQA) in Qwen3-0.6B using MLX’s `metal_kernel` API, with performance evaluated via `mlx_lm.generate`.

> **Important**: Earlier versions of this example had evaluation validity issues (subprocess benchmarks not using the evolved kernel, correctness tests using float32 while the default model is bfloat16, and docs/tests assuming the wrong head configuration). These issues can lead to misleading “best program” results and invalid performance claims. The example has been updated to address these problems.

## 1. Introduction

### 1.1 Motivation

Modern transformer models rely heavily on optimized attention kernels for efficient inference. While frameworks like MLX provide highly optimized implementations, the rapid evolution of hardware architectures creates opportunities for specialized optimizations that general-purpose kernels cannot capture. This work explores whether evolutionary code optimization can automatically discover hardware-specific kernel optimizations that outperform expert-engineered baselines.

### 1.2 Target System

- **Model**: Qwen3-0.6B with Grouped Query Attention (16 query heads : 8 key-value heads)
- **Hardware**: Apple M-series GPUs with unified memory architecture  
- **Framework**: MLX with custom Metal kernel integration
- **Baseline**: `mx.fast.scaled_dot_product_attention`
- **Evolution Target**: Metal shader source code implementing GQA attention computation

## 2. Methodology

### 2.1 Evolution Framework

We employ OpenEvolve to automatically optimize the Metal kernel source code responsible for computing attention. The evolutionary process operates on a single code block (EVOLVE-BLOCK) containing approximately 150 lines of Metal C++ shader code while preserving the surrounding MLX integration infrastructure.

**Evolution Configuration**:
- **Population Size**: 25 programs
- **Generations**: 25 iterations  
- **Models**: Gemini 2.5 Flash (60%) + Gemini 2.5 Pro (40%)
- **Selection**: Multi-objective optimization balancing performance and correctness

### 2.2 Evaluation Methodology

Each evolved kernel undergoes comprehensive evaluation:

1. **Correctness Validation**: Functional/safety checks and dtype coverage consistent with the target model (bfloat16 by default).
2. **Performance Benchmarking**: Diverse inference scenarios covering:
   - Short context (16-64 tokens)
   - Long context (512-2048 tokens) 
   - Code generation
   - Sustained dialogue
   - Technical documentation
   - Memory stress tests

3. **Safety Validation**: GPU command buffer error detection and Metal memory violation checking

### 2.3 Optimization Constraints

**Preserved Elements**:
- Kernel function signature and I/O specifications
- Thread grid mapping and bounds checking
- Overall algorithm correctness (attention semantics)
- MLX integration interface

**Optimizable Elements**:
- Memory access patterns and vectorization
- Computation order and algorithmic efficiency
- Apple Silicon specific optimizations
- GQA-specific computation strategies

## 3. Technical Contributions

### 3.1 Discovered Optimizations

The evolutionary process discovered several key optimizations:

#### 3.1.1 Enhanced Vectorization
```metal
// Original: Scalar operations
for (uint d = 0; d < HEAD_DIM; d++) {
    score += query_vec[d] * keys[k_base + d];
}

// Evolved: Vector operations with optimal width
vec<T, 8> query_vec_v[HEAD_DIM / 8];  // 16 vectors for 128-dim heads
for (uint d_vec = 0; d_vec < HEAD_DIM / 8; d_vec++) {
    score += dot(query_vec_v[d_vec], ((device vec<T, 8>*)(keys + k_base))[d_vec]);
}
```

**Note**: Vectorized kernels must be validated under the target dtype (bfloat16 by default). Some vectorized patterns (e.g. `dot(vec<bfloat, N>)`) may not compile on Metal and should be caught by correctness gating.

#### 3.1.2 Online Softmax Algorithm
```metal
// Pass 1: Find maximum for numerical stability
T max_score = T(-INFINITY);
for (uint key_pos = 0; key_pos < SEQ_LEN; key_pos++) {
    T score = compute_attention_score(query_vec, key_vec) * scale_val;
    max_score = max(max_score, score);
}

// Pass 2: Combined softmax computation and value accumulation
T sum_exp = T(0.0);
vec<T, 8> output_acc_v[HEAD_DIM / 8];
for (uint key_pos = 0; key_pos < SEQ_LEN; key_pos++) {
    T exp_score = exp(current_score - max_score);
    sum_exp += exp_score;
    // Fused accumulation
    output_acc_v[d_vec] += exp_score * ((device vec<T, 8>*)(values + v_base))[d_vec];
}
```

**Innovation**: Reduced from three-pass to two-pass algorithm, fusing softmax normalization with value accumulation.

#### 3.1.3 Memory Access Optimization
```metal
// Pre-computed base indices for coalesced access
const uint q_base = batch_idx * (NUM_HEADS * SEQ_LEN * HEAD_DIM) + 
                    head_idx * (SEQ_LEN * HEAD_DIM) + 
                    query_pos * HEAD_DIM;
const uint kv_head_idx = head_idx / HEADS_PER_KV;  // Direct 2:1 mapping
```

**Innovation**: Leverages unified memory bandwidth through coalesced access patterns and direct GQA head mapping.

### 3.2 Apple Silicon Specialization

The evolved kernel exploits specific Apple Silicon features:
- **Unified Memory**: Optimized bandwidth utilization patterns
- **SIMD Width**: 8-element vectors matching GPU vector units  
- **Thread Group Size**: 32-thread groups optimal for Apple GPUs
- **Register Allocation**: Balanced computation vs. memory bandwidth

## 4. Experimental Results

### 4.1 Performance Benchmarking

We evaluate kernels against the MLX baseline across a benchmark suite representing real-world inference patterns.

**Note**: If you are comparing results across commits, ensure the benchmarks are actually exercising the custom kernel (subprocess hook) and that correctness covers the target dtype (bfloat16). Otherwise, reported speedups can be noise.

To reproduce results on your machine:

```bash
cd openevolve/examples/mlx_metal_kernel_opt
python run_benchmarks.py --mode compare --model mlx-community/Qwen3-0.6B-bf16 --output-dir results
```

This writes CSV/JSON comparison artifacts into `results/` for analysis.

### 4.4 Correctness Validation

Correctness checks should include:
- **Target dtype coverage** (bfloat16 by default for `mlx-community/Qwen3-0.6B-bf16`)
- **Numerical sanity** (no NaN/Inf)
- **Shape checks**
- **Safety checks** (GPU command buffer errors / memory violations)

## 5. Discussion

### 5.1 Performance Characteristics

Kernel performance is workload-dependent. In particular:

- **Short sequences**: may see limited gains due to fixed overheads.
- **Long sequences / sustained decode**: are typically where attention kernels matter most, but must be measured.

### 5.2 Technical Insights

**Vectorization Impact**: The discovery of `vec<T, 8>` operations as optimal for 128-dimensional heads represents a significant finding, suggesting that hardware-specific vector widths are crucial for performance.

**Algorithm Innovation**: The two-pass online softmax represents a novel contribution, demonstrating that evolutionary approaches can discover algorithmic improvements beyond simple micro-optimizations.

**GQA Specialization**: Direct exploitation of the 2:1 query-to-KV head ratio through specialized indexing patterns shows the value of architecture-specific optimizations.

### 5.3 Evolutionary Process Analysis

With realistic evaluation enabled (subprocess hook + bfloat16 correctness), it is expected that some evolved kernels will be rejected due to bfloat16 Metal compilation/runtime failures. The evaluator should treat these as ordinary candidate failures (not crashes) and continue evolution.

## 6. Related Work

This work extends prior research in automated kernel optimization:

- **AlphaTensor** [Fawzi et al., 2022]: Matrix multiplication algorithm discovery
- **TensorIR** [Feng et al., 2023]: Tensor compiler optimization  
- **Ansor** [Zheng et al., 2020]: Automated tensor program optimization

Our approach differs by applying evolutionary optimization directly to GPU shader source code rather than higher-level tensor algebra, enabling discovery of hardware-specific optimizations that would be difficult to express in tensor IRs.

## 7. Limitations and Future Work

### 7.1 Current Limitations

- **Workload Specificity**: Performance improvements are highly dependent on sequence patterns
- **Model Scope**: Results specific to Qwen3-0.6B's 16:8 GQA configuration
- **Hardware Scope**: Optimizations specific to Apple Silicon architecture

### 7.2 Future Directions

- **Multi-Architecture**: Extend to CUDA, ROCm, and other GPU architectures
- **Model Generalization**: Apply to different attention patterns and model sizes  
- **Algorithmic Expansion**: Explore evolution of other transformer components
- **Cross-Compilation**: Develop architecture-agnostic optimization strategies

## 8. Conclusion

We demonstrate how evolutionary code optimization can be applied to discover hardware-specific Metal kernels for transformer attention. Performance gains are workload-dependent; for credible results, rerun the benchmark suite on your machine with the evaluation validity fixes enabled.

This work establishes evolutionary optimization as a viable approach for automated GPU kernel discovery and suggests significant potential for applying similar techniques to other performance-critical computational kernels.


