# 🎯 Qwen3-0.6B Custom GQA Attention Optimization

**Evolving custom Grouped Query Attention kernels using MLX primitives for Qwen3-0.6B on Apple M4**

This example demonstrates AlphaEvolve's kernel optimization approach by implementing and evolving custom GQA attention computation using MLX primitives, targeting the specific 40:8 query-to-KV head pattern in Qwen3-0.6B.

## 🔄 **Updated Approach: Custom Kernel Implementation**

### **Why We Changed Strategy:**

**Previous Approach (High-level orchestration):**
- ❌ Only optimized around `mx.fast.scaled_dot_product_attention`
- ❌ Limited optimization opportunities
- ❌ Multiple EVOLVE-BLOCKS (OpenEvolve format violation)

**Current Approach (Custom kernel implementation):**
- ✅ **Custom GQA implementation** using MLX primitives 
- ✅ **Real optimization opportunities** at computation level
- ✅ **Single EVOLVE-BLOCK** with core attention computation
- ✅ **Follows AlphaEvolve methodology** of optimizing actual kernels

## 🎯 **Optimization Target**

- **Model**: mlx-community/Qwen3-0.6B-bf16
- **Architecture**: 40 query heads : 8 key/value heads (5:1 GQA ratio)
- **Hardware**: Apple M4 24GB unified memory
- **Baseline Performance**: 70.3 tokens/sec average decode speed
- **Goal**: 80+ tokens/sec (14%+ improvement)

## 🔧 **Custom GQA Implementation**

### **Core Evolution Area (Single EVOLVE-BLOCK):**

```python
def __call__(self, x, mask=None, cache=None):
    # Standard preprocessing...
    queries = self.q_proj(x)  # [B, L, 40*128]
    keys = self.k_proj(x)     # [B, L, 8*128] 
    values = self.v_proj(x)   # [B, L, 8*128]
    
    # EVOLVE-BLOCK-START
    # Custom GQA Attention Implementation using MLX primitives
    # This replaces mx.fast.scaled_dot_product_attention entirely
    
    # Current baseline: Manual broadcasting + standard computation
    keys_expanded = mx.repeat(keys, self.gqa_ratio, axis=1)     # [B, 40, L, 128]
    values_expanded = mx.repeat(values, self.gqa_ratio, axis=1) # [B, 40, L, 128]
    
    scores = mx.matmul(queries, keys_expanded.transpose(0, 1, 3, 2)) * self.scale
    attn_weights = mx.softmax(scores, axis=-1, precise=True)
    output = mx.matmul(attn_weights, values_expanded)
    
    # EVOLUTION OPPORTUNITIES:
    # 1. Better GQA broadcasting strategies (chunked computation)
    # 2. Fused operations (combined matmul+softmax)
    # 3. Memory layout optimization for Apple Silicon
    # 4. Optimized causal masking
    # EVOLVE-BLOCK-END
```

## 🚀 **Key Optimization Opportunities**

### **1. GQA Broadcasting Strategies:**
```python
# Current: Explicit broadcasting with mx.repeat
keys_expanded = mx.repeat(keys, 5, axis=1)  # Creates 5x memory usage

# Evolution options:
# - Chunked computation (process 5 query heads per KV head)
# - On-demand broadcasting (avoid materialized copies)
# - Strided access patterns (direct indexing)
```

### **2. Computation Fusion:**
```python
# Current: Separate operations
scores = mx.matmul(queries, keys_t) * scale
weights = mx.softmax(scores)
output = mx.matmul(weights, values)

# Evolution: Fused operations to reduce memory transfers
```

### **3. Apple Silicon Optimizations:**
- bfloat16 native operations
- Unified memory bandwidth optimization
- Cache-friendly memory access patterns
- SIMD-friendly computation layouts

## 📊 **Baseline vs Custom Implementation**

From your M4 benchmarks:
```
Baseline Performance (mx.fast.scaled_dot_product_attention):
- Average decode: 70.3 tokens/sec
- Range: 65.0 - 80.7 tokens/sec
- Memory: 1.24-1.69 GB
- Context degradation: ~7%

Custom Implementation Target:
- Average decode: 80+ tokens/sec (14%+ improvement)
- Better memory efficiency
- Improved context scaling
- Maintained numerical accuracy
```

## 🧪 **Evaluation System**

### **Comprehensive Testing:**
1. **Correctness Verification**: Custom implementation produces identical results
2. **Performance Benchmarking**: Real text generation on 5 key scenarios
3. **Memory Efficiency**: Track memory usage vs baseline
4. **Context Scaling**: Test performance across different sequence lengths

### **Success Metrics:**
- **Primary**: Average decode speed improvement (70.3 → 80+ tokens/sec)
- **Secondary**: Memory efficiency, context scaling
- **Critical**: Numerical correctness maintained

## 🚀 **Usage**

### **1. Test Initial Custom Implementation**
```bash
cd /Users/asankhaya/Documents/GitHub/openevolve/examples/mlx_metal_kernel_opt
python initial_program.py  # Test custom GQA implementation
```

### **2. Run Evaluator Test**
```bash
python evaluator.py  # Test evaluation system
```

### **3. Start Evolution**
```bash
cd /Users/asankhaya/Documents/GitHub/openevolve
python main.py --config examples/mlx_metal_kernel_opt/config.yaml
```

## 📈 **Expected Evolution Trajectory**

### **Generation 1-10: Broadcasting Optimizations**
- Chunked GQA computation strategies
- Memory-efficient broadcasting alternatives
- Target: 70.3 → 73-75 tokens/sec

### **Generation 11-20: Computation Fusion**
- Fused matmul + softmax operations
- Optimized causal masking integration
- Target: 75 → 78-82 tokens/sec

### **Generation 21-30: Apple Silicon Specialization**
- bfloat16 optimization
- Unified memory access patterns
- Advanced tensor layout optimization
- Target: 80+ tokens/sec (14%+ improvement)

## 🔍 **Key Advantages of Custom Implementation**

### **Real Optimization Potential:**
- **Kernel-level optimizations** using MLX primitives
- **GQA-specific strategies** for 40:8 pattern
- **Apple Silicon specialization** for M4 architecture
- **Measurable improvements** on real workloads

### **Realistic Scope:**
- Uses MLX's optimized primitives (not raw Metal)
- Maintains compatibility with mlx-lm ecosystem
- Achievable 14% improvement target
- Working baseline implementation

### **Evolution-Friendly:**
- Single EVOLVE-BLOCK with core computation
- Clear optimization opportunities
- Concrete performance targets
- Systematic testing framework

## 💡 **Why This Approach Will Work**

1. **Real baseline**: 70.3 tokens/sec from actual M4 measurements
2. **Custom implementation**: Full control over GQA computation  
3. **MLX primitives**: Optimized building blocks, not raw Metal
4. **Specific target**: Qwen3's exact 40:8 pattern, not generic attention
5. **Proven methodology**: Following AlphaEvolve's kernel optimization approach

This approach should evolve meaningful, measurable improvements for Qwen3-0.6B's specific GQA pattern while maintaining compatibility and correctness.

---

**🎯 Ready for custom kernel evolution!**
