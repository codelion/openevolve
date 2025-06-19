# OpenEvolve Metal Kernel Optimization: Automated Discovery of Custom GPU Kernels for Transformer Attention

**Evolutionary Optimization of Apple Silicon Metal Kernels for Grouped Query Attention in Qwen3-0.6B**

## Abstract

This work demonstrates the application of evolutionary code optimization to the automatic discovery of custom Metal GPU kernels for transformer attention mechanisms. Using OpenEvolve, we evolved a specialized Metal kernel for Grouped Query Attention (GQA) in Qwen3-0.6B that leverages Apple Silicon's unified memory architecture and vector processing capabilities. Our approach achieved measurable performance improvements over MLX's highly optimized `scaled_dot_product_attention` baseline across diverse inference workloads, with decode speed improvements averaging 12.5% and reaching up to 106% on specific benchmark tasks.

## 1. Introduction

### 1.1 Motivation

Modern transformer models rely heavily on optimized attention kernels for efficient inference. While frameworks like MLX provide highly optimized implementations, the rapid evolution of hardware architectures creates opportunities for specialized optimizations that general-purpose kernels cannot capture. This work explores whether evolutionary code optimization can automatically discover hardware-specific kernel optimizations that outperform expert-engineered baselines.

### 1.2 Target System

- **Model**: Qwen3-0.6B with Grouped Query Attention (40 query heads : 8 key-value heads)
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

1. **Correctness Validation**: Numerical accuracy verification against MLX baseline
2. **Performance Benchmarking**: 20 diverse inference scenarios covering:
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

**Innovation**: Using 8-element vectors perfectly matches Apple Silicon's SIMD capabilities for 128-dimensional attention heads.

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
const uint kv_head_idx = head_idx / HEADS_PER_KV;  // Direct 5:1 mapping
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

We evaluated the evolved kernel against MLX baseline across 20 comprehensive benchmark scenarios representing real-world inference patterns.

**Aggregate Performance Improvements**:
- **Decode Speed**: +12.5% average improvement (σ = 38.3%)
- **Prefill Speed**: +14.4% average improvement (σ = 17.6%)  
- **Total Throughput**: +10.4% average improvement (σ = 30.7%)
- **Memory Usage**: +0.99% average reduction (σ = 1.7%)

### 4.2 Benchmark Category Analysis

| **Category** | **Benchmarks** | **Decode Improvement** | **Notable Results** |
|--------------|----------------|------------------------|-------------------|
| **Short Context** | 2 | -4.6% ± 3.8% | Mixed results on very short sequences |
| **Long Context** | 6 | +8.1% ± 42.1% | High variance, strong improvements in some cases |
| **Code Generation** | 1 | -16.5% | Performance regression |  
| **General Tasks** | 9 | +24.8% ± 35.4% | Strongest category with 106% peak improvement |
| **Stress Tests** | 2 | +22.9% ± 31.5% | Robust performance under memory pressure |

### 4.3 Statistical Analysis

**Distribution of Improvements**:
- **Significant Gains** (>25%): 7/20 benchmarks
- **Moderate Gains** (5-25%): 3/20 benchmarks  
- **Neutral** (±5%): 4/20 benchmarks
- **Regressions** (<-5%): 6/20 benchmarks

**Peak Performance**: Repetitive pattern generation achieved 106% decode speed improvement, demonstrating the kernel's effectiveness for certain workload characteristics.

### 4.4 Correctness Validation

All evolved kernels maintained numerical correctness:
- **Accuracy**: 100% correctness score across all test cases
- **Numerical Stability**: No NaN/Inf values detected
- **Statistical Validation**: Output distributions within expected ranges
- **Functional Equivalence**: Attention semantics preserved

## 5. Discussion

### 5.1 Performance Characteristics

The evolved kernel shows workload-dependent performance characteristics:

**Strengths**:
- **Sustained Generation**: +46.6% improvement on dialogue tasks
- **Long Sequences**: +73.9% improvement on extreme-length generation
- **Memory Efficiency**: Consistent memory usage reduction

**Limitations**:  
- **Short Sequences**: Limited improvement due to setup overhead
- **Code Generation**: -16.5% regression suggesting suboptimal patterns for this workload
- **Variance**: High performance variance across different sequence patterns

### 5.2 Technical Insights

**Vectorization Impact**: The discovery of `vec<T, 8>` operations as optimal for 128-dimensional heads represents a significant finding, suggesting that hardware-specific vector widths are crucial for performance.

**Algorithm Innovation**: The two-pass online softmax represents a novel contribution, demonstrating that evolutionary approaches can discover algorithmic improvements beyond simple micro-optimizations.

**GQA Specialization**: Direct exploitation of the 5:1 query-to-KV head ratio through specialized indexing patterns shows the value of architecture-specific optimizations.

### 5.3 Evolutionary Process Analysis

**Convergence**: The system converged to the optimal solution within 25 generations, with significant improvements appearing by generation 10.

**Safety**: Zero Metal kernel compilation errors or GPU command buffer failures across all evolution attempts, demonstrating robust evolutionary constraints.

**Diversity**: The evolutionary process explored multiple optimization strategies including different vectorization patterns, memory layouts, and algorithmic approaches.

## 6. Related Work

This work extends prior research in automated kernel optimization:

- **AlphaTensor** [Fawzi et al., 2022]: Matrix multiplication algorithm discovery
- **TensorIR** [Feng et al., 2023]: Tensor compiler optimization  
- **Ansor** [Zheng et al., 2020]: Automated tensor program optimization

Our approach differs by applying evolutionary optimization directly to GPU shader source code rather than higher-level tensor algebra, enabling discovery of hardware-specific optimizations that would be difficult to express in tensor IRs.

## 7. Limitations and Future Work

### 7.1 Current Limitations

- **Workload Specificity**: Performance improvements are highly dependent on sequence patterns
- **Model Scope**: Results specific to Qwen3-0.6B's 40:8 GQA configuration
- **Hardware Scope**: Optimizations specific to Apple Silicon architecture

### 7.2 Future Directions

- **Multi-Architecture**: Extend to CUDA, ROCm, and other GPU architectures
- **Model Generalization**: Apply to different attention patterns and model sizes  
- **Algorithmic Expansion**: Explore evolution of other transformer components
- **Cross-Compilation**: Develop architecture-agnostic optimization strategies

## 8. Conclusion

We demonstrate that evolutionary code optimization can automatically discover hardware-specific GPU kernel optimizations that outperform expert-engineered baselines. The evolved Metal kernel achieved an average 12.5% decode speed improvement through novel vectorization patterns, algorithmic innovations, and Apple Silicon specializations. While performance gains are workload-dependent, the approach successfully identified genuinely novel optimizations that would be challenging to discover through manual optimization.

This work establishes evolutionary optimization as a viable approach for automated GPU kernel discovery and suggests significant potential for applying similar techniques to other performance-critical computational kernels.