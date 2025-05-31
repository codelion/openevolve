#!/usr/bin/env python3
"""
MLX Attention Grid Search

This script performs a comprehensive grid search to find optimal attention configurations
for different sequence lengths. It focuses on finding configurations that achieve
perfect accuracy (1.0 cosine similarity) while maximizing performance.

Grid Search Parameters:
- sequence_length: [128, 512, 1024, 4096]
- window_size: [None, 32, 64, 128, 256, 512]
- query_chunk_size: [64, 128, 256, 512]
- dilation_rate: [1, 2, 3, 4]

The script prioritizes numerical accuracy and identifies the fastest configurations
that maintain perfect compatibility with standard attention.
"""

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Tuple, Any
import importlib.util

import mlx.core as mx
import mlx.nn as nn
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass
class GridSearchConfig:
    """Configuration for grid search parameters"""
    
    # Grid search dimensions
    sequence_lengths: List[int]
    window_sizes: List[Optional[int]]
    query_chunk_sizes: List[int]
    dilation_rates: List[int]
    
    # Model architecture (fixed for search)
    hidden_size: int = 768
    num_heads: int = 12
    num_kv_heads: int = 12
    batch_size: int = 1
    
    # Evaluation parameters
    warmup_runs: int = 5
    benchmark_runs: int = 10
    accuracy_threshold: float = 0.9  # Threshold for "perfect" accuracy
    timeout_seconds: int = 60
    
    # Resource limits
    max_memory_gb: float = 21.0  # Skip configs that might use too much memory
    
    @classmethod
    def default(cls):
        """Create default grid search configuration"""
        return cls(
            sequence_lengths=[1024, 2048, 4096, 8192, 16384],
            window_sizes=[256, 512, 1024, 2048, 4096, 8192],
            query_chunk_sizes=[64, 128, 256, 512, 1024, 2048, 4096],
            dilation_rates=[1, 2, 3, 4],
        )
    
    def estimate_total_configs(self) -> int:
        """Estimate total number of configurations to test"""
        return len(self.sequence_lengths) * len(self.window_sizes) * len(self.query_chunk_sizes) * len(self.dilation_rates)
    
    def is_config_valid(self, seq_len: int, window_size: Optional[int], 
                       chunk_size: int, dilation: int) -> Tuple[bool, str]:
        """Check if a configuration is valid and provide reason if not"""
        
        # Window size validation
        if window_size is not None:
            if window_size >= seq_len:
                return False, f"window_size ({window_size}) >= seq_len ({seq_len})"
            if window_size < 2:
                return False, f"window_size ({window_size}) too small"
        
        # Chunk size validation
        if chunk_size > seq_len:
            return False, f"chunk_size ({chunk_size}) > seq_len ({seq_len})"
        
        # Dilation validation
        if window_size is not None and dilation > 1:
            effective_window = window_size * dilation
            if effective_window >= seq_len:
                return False, f"effective_window ({effective_window}) >= seq_len ({seq_len})"
        
        # Memory estimation (rough)
        attention_memory_gb = (seq_len ** 2 * self.batch_size * self.num_heads * 4) / (1024**3)  # 4 bytes per float32
        if attention_memory_gb > self.max_memory_gb:
            return False, f"estimated memory ({attention_memory_gb:.1f}GB) > limit ({self.max_memory_gb}GB)"
        
        return True, "valid"


@dataclass 
class GridSearchResult:
    """Results for a single grid search configuration"""
    
    # Configuration
    seq_len: int
    window_size: Optional[int]
    query_chunk_size: int
    dilation_rate: int
    
    # Results
    success: bool
    error_message: str = ""
    
    # Performance metrics
    standard_time: float = 0.0
    evolved_time: float = 0.0
    speedup: float = 0.0
    tokens_per_second: float = 0.0
    
    # Accuracy metrics
    cosine_similarity: float = 0.0
    mse: float = float('inf')
    max_diff: float = float('inf')
    weights_synced: bool = False
    perfect_accuracy: bool = False
    
    # Timing details
    benchmark_runs: int = 0
    std_time_std: float = 0.0
    evo_time_std: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'seq_len': self.seq_len,
            'window_size': self.window_size,
            'query_chunk_size': self.query_chunk_size,
            'dilation_rate': self.dilation_rate,
            'success': self.success,
            'error_message': self.error_message,
            'standard_time': self.standard_time,
            'evolved_time': self.evolved_time,
            'speedup': self.speedup,
            'tokens_per_second': self.tokens_per_second,
            'cosine_similarity': self.cosine_similarity,
            'mse': self.mse,
            'max_diff': self.max_diff,
            'weights_synced': self.weights_synced,
            'perfect_accuracy': self.perfect_accuracy,
            'benchmark_runs': self.benchmark_runs,
            'std_time_std': self.std_time_std,
            'evo_time_std': self.evo_time_std
        }


class AttentionGridSearch:
    """Grid search for optimal attention configurations"""
    
    def __init__(self, config: GridSearchConfig, evolved_program_path: str):
        self.config = config
        self.evolved_program_path = evolved_program_path
        self.results: List[GridSearchResult] = []
        self.current_progress = 0
        self.total_configs = 0
        
        # Load attention implementations
        self._load_implementations()
    
    def _load_implementations(self):
        """Load both standard and evolved attention implementations"""
        print("📥 Loading attention implementations...")
        
        # Load standard implementation
        current_dir = os.path.dirname(os.path.abspath(__file__))
        initial_program_path = os.path.join(current_dir, "initial_program.py")
        
        if not os.path.exists(initial_program_path):
            raise FileNotFoundError(f"Standard implementation not found: {initial_program_path}")
            
        spec = importlib.util.spec_from_file_location("standard_attention", initial_program_path)
        self.standard_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.standard_module)
        
        # Load evolved implementation
        if not os.path.exists(self.evolved_program_path):
            raise FileNotFoundError(f"Evolved implementation not found: {self.evolved_program_path}")
            
        spec = importlib.util.spec_from_file_location("evolved_attention", self.evolved_program_path)
        self.evolved_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.evolved_module)
        
        print("✅ Both implementations loaded successfully")
    
    def copy_module_weights(self, source_module, target_module) -> bool:
        """Copy weights from source to target module for fair comparison"""
        try:
            weight_attrs = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'q_norm', 'k_norm']
            copied_count = 0
            
            for attr_name in weight_attrs:
                if hasattr(source_module, attr_name) and hasattr(target_module, attr_name):
                    source_layer = getattr(source_module, attr_name)
                    target_layer = getattr(target_module, attr_name)
                    
                    if (hasattr(source_layer, 'weight') and hasattr(target_layer, 'weight')):
                        source_weight = source_layer.weight
                        target_weight = target_layer.weight
                        
                        if source_weight.shape == target_weight.shape:
                            target_layer.weight = mx.array(source_weight)
                            copied_count += 1
                    
                    if (hasattr(source_layer, 'bias') and hasattr(target_layer, 'bias') and
                        source_layer.bias is not None and target_layer.bias is not None):
                        if source_layer.bias.shape == target_layer.bias.shape:
                            target_layer.bias = mx.array(source_layer.bias)
            
            return copied_count > 0
            
        except Exception as e:
            print(f"      Weight sync failed: {str(e)}")
            return False
    
    def _create_attention_modules(self, seq_len: int, window_size: Optional[int], 
                                chunk_size: int, dilation: int) -> Tuple[Any, Any]:
        """Create both standard and evolved attention modules"""
        
        head_dim = self.config.hidden_size // self.config.num_heads
        
        # Create standard module
        standard_module = self.standard_module.create_test_attention_module(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_heads,
            num_kv_heads=self.config.num_kv_heads,
            head_dim=head_dim
        )
        
        # Create evolved module with specific parameters
        try:
            evolved_module = self.evolved_module.create_test_attention_module(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_heads,
                num_kv_heads=self.config.num_kv_heads,
                head_dim=head_dim,
                window_size=window_size,
                query_chunk_size=chunk_size,
                dilation_rate=dilation
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create evolved module: {str(e)}")
        
        return standard_module, evolved_module
    
    def _benchmark_module(self, module, x: mx.array, mask: mx.array, name: str) -> Dict[str, float]:
        """Benchmark a single attention module"""
        
        # Warmup
        for _ in range(self.config.warmup_runs):
            output = module(x, mask=mask)
            mx.eval(output)
        
        # Timed runs
        times = []
        for _ in range(self.config.benchmark_runs):
            start_time = time.time()
            output = module(x, mask=mask)
            mx.eval(output)
            end_time = time.time()
            
            run_time = end_time - start_time
            times.append(run_time)
            
            if run_time > self.config.timeout_seconds:
                raise TimeoutError(f"{name} run took too long: {run_time:.2f}s")
        
        return {
            "avg_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times)
        }
    
    def _check_accuracy(self, standard_module, evolved_module, x: mx.array, mask: mx.array) -> Dict[str, float]:
        """Check numerical accuracy between implementations"""
        
        try:
            # Sync weights for fair comparison
            weights_synced = self.copy_module_weights(standard_module, evolved_module)
            
            # Get outputs
            standard_output = standard_module(x, mask=mask)
            evolved_output = evolved_module(x, mask=mask)
            
            mx.eval(standard_output)
            mx.eval(evolved_output)
            
            # Calculate metrics
            mse = float(mx.mean((standard_output - evolved_output) ** 2))
            max_diff = float(mx.max(mx.abs(standard_output - evolved_output)))
            
            # Cosine similarity
            std_flat = standard_output.reshape(-1)
            evo_flat = evolved_output.reshape(-1)
            
            eps = 1e-8
            dot_product = float(mx.sum(std_flat * evo_flat))
            norm_std = float(mx.sqrt(mx.sum(std_flat ** 2) + eps))
            norm_evo = float(mx.sqrt(mx.sum(evo_flat ** 2) + eps))
            
            cosine_sim = dot_product / (norm_std * norm_evo)
            cosine_sim = max(-1.0, min(1.0, cosine_sim))  # Clamp to valid range
            
            return {
                "cosine_similarity": cosine_sim,
                "mse": mse,
                "max_diff": max_diff,
                "weights_synced": weights_synced
            }
            
        except Exception as e:
            return {
                "cosine_similarity": 0.0,
                "mse": float('inf'),
                "max_diff": float('inf'),
                "weights_synced": False,
                "error": str(e)
            }
    
    def test_configuration(self, seq_len: int, window_size: Optional[int], 
                          chunk_size: int, dilation: int) -> GridSearchResult:
        """Test a single configuration"""
        
        result = GridSearchResult(
            seq_len=seq_len,
            window_size=window_size,
            query_chunk_size=chunk_size,
            dilation_rate=dilation,
            success=False
        )
        
        try:
            # Create test data
            x = mx.random.normal((self.config.batch_size, seq_len, self.config.hidden_size))
            causal_mask = mx.triu(mx.full((seq_len, seq_len), -mx.inf), k=1)
            mask = mx.expand_dims(causal_mask, axis=0)
            
            # Create modules
            standard_module, evolved_module = self._create_attention_modules(
                seq_len, window_size, chunk_size, dilation
            )
            
            # Benchmark standard
            std_results = self._benchmark_module(standard_module, x, mask, "Standard")
            result.standard_time = std_results["avg_time"]
            result.std_time_std = std_results["std_time"]
            
            # Benchmark evolved
            evo_results = self._benchmark_module(evolved_module, x, mask, "Evolved")
            result.evolved_time = evo_results["avg_time"]
            result.evo_time_std = evo_results["std_time"]
            
            # Calculate performance metrics
            if result.standard_time > 0:
                result.speedup = result.standard_time / result.evolved_time
                total_tokens = self.config.batch_size * seq_len
                result.tokens_per_second = total_tokens / result.evolved_time
            
            # Check accuracy
            accuracy = self._check_accuracy(standard_module, evolved_module, x, mask)
            result.cosine_similarity = accuracy["cosine_similarity"]
            result.mse = accuracy["mse"]
            result.max_diff = accuracy["max_diff"]
            result.weights_synced = accuracy["weights_synced"]
            result.perfect_accuracy = (
                result.weights_synced and 
                result.cosine_similarity >= self.config.accuracy_threshold
            )
            
            result.benchmark_runs = self.config.benchmark_runs
            result.success = True
            
        except Exception as e:
            result.error_message = str(e)
            result.success = False
        
        return result
    
    def run_grid_search(self, checkpoint_file: Optional[str] = None) -> List[GridSearchResult]:
        """Run the complete grid search"""
        
        print("🔍 Starting MLX Attention Grid Search")
        print(f"   Sequence lengths: {self.config.sequence_lengths}")
        print(f"   Window sizes: {self.config.window_sizes}")
        print(f"   Query chunk sizes: {self.config.query_chunk_sizes}")
        print(f"   Dilation rates: {self.config.dilation_rates}")
        
        # Load existing results if checkpoint exists
        if checkpoint_file and os.path.exists(checkpoint_file):
            print(f"📂 Loading checkpoint: {checkpoint_file}")
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                self.results = [GridSearchResult(**r) for r in checkpoint_data.get('results', [])]
                self.current_progress = len(self.results)
                print(f"   Resumed from {self.current_progress} completed configurations")
        
        # Generate all configurations
        all_configs = list(product(
            self.config.sequence_lengths,
            self.config.window_sizes,
            self.config.query_chunk_sizes,
            self.config.dilation_rates
        ))
        
        self.total_configs = len(all_configs)
        print(f"   Total configurations: {self.total_configs}")
        
        # Skip already completed configurations
        completed_configs = set()
        for result in self.results:
            config_key = (result.seq_len, result.window_size, result.query_chunk_size, result.dilation_rate)
            completed_configs.add(config_key)
        
        # Process remaining configurations
        start_time = time.time()
        
        for config_idx, (seq_len, window_size, chunk_size, dilation) in enumerate(all_configs):
            config_key = (seq_len, window_size, chunk_size, dilation)
            
            # Skip if already completed
            if config_key in completed_configs:
                continue
            
            self.current_progress += 1
            progress_pct = (self.current_progress / self.total_configs) * 100
            
            print(f"\n🔄 [{self.current_progress}/{self.total_configs}] ({progress_pct:.1f}%) "
                  f"seq_len={seq_len}, window={window_size}, chunk={chunk_size}, dilation={dilation}")
            
            # Check if configuration is valid
            is_valid, reason = self.config.is_config_valid(seq_len, window_size, chunk_size, dilation)
            
            if not is_valid:
                print(f"   ⏭️  Skipping invalid config: {reason}")
                result = GridSearchResult(
                    seq_len=seq_len,
                    window_size=window_size,
                    query_chunk_size=chunk_size,
                    dilation_rate=dilation,
                    success=False,
                    error_message=f"Invalid config: {reason}"
                )
                self.results.append(result)
                continue
            
            # Test configuration
            try:
                result = self.test_configuration(seq_len, window_size, chunk_size, dilation)
                self.results.append(result)
                
                if result.success:
                    accuracy_symbol = "🎯" if result.perfect_accuracy else "📊"
                    print(f"   {accuracy_symbol} Speedup: {result.speedup:.2f}x, "
                          f"Accuracy: {result.cosine_similarity:.4f}, "
                          f"Synced: {result.weights_synced}")
                else:
                    print(f"   ❌ Failed: {result.error_message}")
            
            except Exception as e:
                print(f"   💥 Unexpected error: {str(e)}")
                result = GridSearchResult(
                    seq_len=seq_len,
                    window_size=window_size,
                    query_chunk_size=chunk_size,
                    dilation_rate=dilation,
                    success=False,
                    error_message=f"Unexpected error: {str(e)}"
                )
                self.results.append(result)
            
            # Save checkpoint periodically
            if checkpoint_file and self.current_progress % 10 == 0:
                self._save_checkpoint(checkpoint_file)
            
            # Progress estimate
            elapsed = time.time() - start_time
            if self.current_progress > 1:
                avg_time_per_config = elapsed / (self.current_progress - len(completed_configs))
                remaining_configs = self.total_configs - self.current_progress
                estimated_remaining = avg_time_per_config * remaining_configs
                print(f"   ⏱️  Est. remaining: {estimated_remaining/60:.1f} minutes")
        
        # Final checkpoint
        if checkpoint_file:
            self._save_checkpoint(checkpoint_file)
        
        elapsed_total = time.time() - start_time
        print(f"\n✅ Grid search complete! Total time: {elapsed_total/60:.1f} minutes")
        
        return self.results
    
    def _save_checkpoint(self, checkpoint_file: str):
        """Save current progress to checkpoint file"""
        checkpoint_data = {
            'config': {
                'sequence_lengths': self.config.sequence_lengths,
                'window_sizes': self.config.window_sizes,
                'query_chunk_sizes': self.config.query_chunk_sizes,
                'dilation_rates': self.config.dilation_rates,
                'hidden_size': self.config.hidden_size,
                'num_heads': self.config.num_heads,
                'accuracy_threshold': self.config.accuracy_threshold
            },
            'progress': {
                'current': self.current_progress,
                'total': self.total_configs
            },
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"   💾 Checkpoint saved: {checkpoint_file}")
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze grid search results and find optimal configurations"""
        
        successful_results = [r for r in self.results if r.success]
        perfect_accuracy_results = [r for r in successful_results if r.perfect_accuracy]
        
        print(f"\n📊 GRID SEARCH ANALYSIS")
        print(f"   Total configurations tested: {len(self.results)}")
        print(f"   Successful: {len(successful_results)}")
        print(f"   Perfect accuracy: {len(perfect_accuracy_results)}")
        
        if not perfect_accuracy_results:
            print("   ⚠️  No configurations achieved perfect accuracy!")
            return {}
        
        # Group by sequence length
        results_by_seq_len = {}
        for seq_len in self.config.sequence_lengths:
            seq_results = [r for r in perfect_accuracy_results if r.seq_len == seq_len]
            if seq_results:
                # Find best speedup for this sequence length
                best_result = max(seq_results, key=lambda x: x.speedup)
                results_by_seq_len[seq_len] = {
                    'best_config': best_result,
                    'all_perfect': seq_results,
                    'count': len(seq_results)
                }
        
        # Overall statistics
        all_speedups = [r.speedup for r in perfect_accuracy_results]
        all_accuracies = [r.cosine_similarity for r in perfect_accuracy_results]
        
        analysis = {
            'summary': {
                'total_tested': len(self.results),
                'successful': len(successful_results),
                'perfect_accuracy_count': len(perfect_accuracy_results),
                'perfect_accuracy_rate': len(perfect_accuracy_results) / len(self.results) if self.results else 0,
                'avg_speedup_perfect': np.mean(all_speedups) if all_speedups else 0,
                'max_speedup_perfect': np.max(all_speedups) if all_speedups else 0,
                'avg_accuracy_perfect': np.mean(all_accuracies) if all_accuracies else 0
            },
            'by_sequence_length': results_by_seq_len,
            'recommendations': self._generate_recommendations(results_by_seq_len)
        }
        
        return analysis
    
    def _generate_recommendations(self, results_by_seq_len: Dict[int, Dict]) -> Dict[str, Any]:
        """Generate configuration recommendations based on results"""
        
        recommendations = {
            'optimal_configs': {},
            'patterns': {},
            'general_advice': []
        }
        
        # Optimal config for each sequence length
        for seq_len, data in results_by_seq_len.items():
            best = data['best_config']
            recommendations['optimal_configs'][seq_len] = {
                'window_size': best.window_size,
                'query_chunk_size': best.query_chunk_size,
                'dilation_rate': best.dilation_rate,
                'speedup': best.speedup,
                'accuracy': best.cosine_similarity
            }
        
        # Pattern analysis
        if results_by_seq_len:
            # Window size patterns
            window_sizes = []
            chunk_sizes = []
            dilations = []
            
            for data in results_by_seq_len.values():
                best = data['best_config']
                window_sizes.append(best.window_size)
                chunk_sizes.append(best.query_chunk_size)
                dilations.append(best.dilation_rate)
            
            recommendations['patterns'] = {
                'common_window_size': max(set(window_sizes), key=window_sizes.count) if window_sizes else None,
                'common_chunk_size': max(set(chunk_sizes), key=chunk_sizes.count) if chunk_sizes else None,
                'common_dilation': max(set(dilations), key=dilations.count) if dilations else None
            }
        
        # General advice
        if len(results_by_seq_len) >= 2:
            short_configs = {k: v for k, v in results_by_seq_len.items() if k <= 512}
            long_configs = {k: v for k, v in results_by_seq_len.items() if k > 512}
            
            if short_configs and long_configs:
                recommendations['general_advice'].append(
                    "Different optimal configurations found for short vs long sequences"
                )
        
        return recommendations
    
    def create_visualizations(self, output_dir: str, analysis: Dict[str, Any]):
        """Create visualization plots for grid search results"""
        
        if not PLOTTING_AVAILABLE:
            print("📊 Plotting not available")
            return
        
        successful_results = [r for r in self.results if r.success]
        if not successful_results:
            print("📊 No successful results to plot")
            return
        
        print("📊 Creating grid search visualizations...")
        
        # Convert to structured data for plotting
        if PANDAS_AVAILABLE:
            df = pd.DataFrame([r.to_dict() for r in successful_results])
            
            # Check for perfect accuracy results
            perfect_df = df[df['perfect_accuracy'] == True]
            has_perfect_results = not perfect_df.empty
            
            # 1. Heatmap of speedup by sequence length and window size
            plt.figure(figsize=(12, 8))
            
            if has_perfect_results:
                try:
                    pivot_speedup = perfect_df.pivot_table(
                        values='speedup', 
                        index='window_size', 
                        columns='seq_len', 
                        aggfunc='max'
                    )
                    
                    # Check if pivot table has data
                    if not pivot_speedup.empty and pivot_speedup.notna().any().any():
                        sns.heatmap(pivot_speedup, annot=True, fmt='.2f', cmap='viridis')
                        plt.title('Maximum Speedup by Sequence Length and Window Size\n(Perfect Accuracy Only)')
                        plt.ylabel('Window Size')
                        plt.xlabel('Sequence Length')
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, 'speedup_heatmap_perfect.png'), dpi=150)
                        plt.close()
                        print(f"   ✅ Perfect accuracy heatmap saved")
                    else:
                        print(f"   ⚠️  Perfect accuracy heatmap: no valid data to plot")
                        plt.close()
                except Exception as e:
                    print(f"   ⚠️  Perfect accuracy heatmap failed: {str(e)}")
                    plt.close()
            
            # Alternative heatmap with all successful results if no perfect results
            if not has_perfect_results or len(perfect_df) < 4:  # Need minimum data for meaningful heatmap
                try:
                    plt.figure(figsize=(12, 8))
                    pivot_all = df.pivot_table(
                        values='speedup',
                        index='window_size',
                        columns='seq_len',
                        aggfunc='max'
                    )
                    
                    if not pivot_all.empty and pivot_all.notna().any().any():
                        sns.heatmap(pivot_all, annot=True, fmt='.2f', cmap='viridis')
                        plt.title('Maximum Speedup by Sequence Length and Window Size\n(All Successful Results)')
                        plt.ylabel('Window Size')
                        plt.xlabel('Sequence Length')
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, 'speedup_heatmap_all.png'), dpi=150)
                        plt.close()
                        print(f"   ✅ All results heatmap saved")
                    else:
                        print(f"   ⚠️  All results heatmap: no valid data to plot")
                        plt.close()
                except Exception as e:
                    print(f"   ⚠️  All results heatmap failed: {str(e)}")
                    plt.close()
            
            # 2. Scatter plot: Speedup vs Accuracy (always create this)
            try:
                plt.figure(figsize=(10, 6))
                
                colors = plt.cm.viridis(np.linspace(0, 1, len(self.config.sequence_lengths)))
                seq_len_colors = dict(zip(self.config.sequence_lengths, colors))
                
                plotted_any = False
                for seq_len in self.config.sequence_lengths:
                    seq_data = df[df['seq_len'] == seq_len]
                    if not seq_data.empty:
                        # Filter out invalid data
                        valid_data = seq_data[
                            (seq_data['cosine_similarity'].notna()) & 
                            (seq_data['speedup'].notna()) &
                            (seq_data['speedup'] > 0) &
                            (seq_data['cosine_similarity'] >= 0)
                        ]
                        
                        if not valid_data.empty:
                            plt.scatter(
                                valid_data['cosine_similarity'], 
                                valid_data['speedup'],
                                c=[seq_len_colors[seq_len]], 
                                label=f'seq_len={seq_len}',
                                alpha=0.7
                            )
                            plotted_any = True
                
                if plotted_any:
                    plt.axvline(x=self.config.accuracy_threshold, color='red', linestyle='--', 
                               label=f'Perfect accuracy threshold ({self.config.accuracy_threshold})')
                    plt.xlabel('Cosine Similarity (Accuracy)')
                    plt.ylabel('Speedup')
                    plt.title('Speedup vs Accuracy Trade-off')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'speedup_vs_accuracy.png'), dpi=150)
                    plt.close()
                    print(f"   ✅ Speedup vs accuracy plot saved")
                else:
                    print(f"   ⚠️  Speedup vs accuracy plot: no valid data to plot")
                    plt.close()
            except Exception as e:
                print(f"   ⚠️  Speedup vs accuracy plot failed: {str(e)}")
                plt.close()
            
            # 3. Configuration patterns (only if we have data)
            data_to_plot = perfect_df if has_perfect_results else df
            
            if len(data_to_plot) >= 2:  # Need at least some data for distributions
                try:
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    
                    # Window size distribution
                    window_data = data_to_plot['window_size'].fillna(-1)  # Replace None with -1 for plotting
                    if len(window_data) > 0:
                        axes[0,0].hist(window_data, bins=min(10, len(window_data.unique())), alpha=0.7)
                        axes[0,0].set_title('Distribution of Window Sizes')
                        axes[0,0].set_xlabel('Window Size (None = -1)')
                    
                    # Chunk size distribution  
                    chunk_data = data_to_plot['query_chunk_size']
                    if len(chunk_data) > 0:
                        axes[0,1].hist(chunk_data, bins=min(10, len(chunk_data.unique())), alpha=0.7)
                        axes[0,1].set_title('Distribution of Query Chunk Sizes')
                        axes[0,1].set_xlabel('Query Chunk Size')
                    
                    # Dilation distribution
                    dilation_data = data_to_plot['dilation_rate']
                    if len(dilation_data) > 0:
                        axes[1,0].hist(dilation_data, bins=min(8, len(dilation_data.unique())), alpha=0.7)
                        axes[1,0].set_title('Distribution of Dilation Rates')
                        axes[1,0].set_xlabel('Dilation Rate')
                    
                    # Speedup by sequence length
                    speedup_data = data_to_plot[['speedup', 'seq_len']]
                    speedup_data = speedup_data[speedup_data['speedup'].notna() & (speedup_data['speedup'] > 0)]
                    
                    if len(speedup_data) > 0 and len(speedup_data['seq_len'].unique()) > 1:
                        speedup_data.boxplot(column='speedup', by='seq_len', ax=axes[1,1])
                        axes[1,1].set_title('Speedup Distribution by Sequence Length')
                        axes[1,1].set_xlabel('Sequence Length')
                        axes[1,1].set_ylabel('Speedup')
                        plt.suptitle('')  # Remove automatic title from boxplot
                    else:
                        # Just show speedup histogram if not enough data for boxplot
                        axes[1,1].hist(speedup_data['speedup'], bins=min(10, len(speedup_data)), alpha=0.7)
                        axes[1,1].set_title('Speedup Distribution')
                        axes[1,1].set_xlabel('Speedup')
                    
                    plt.tight_layout()
                    
                    filename_suffix = 'perfect' if has_perfect_results else 'all'
                    plt.savefig(os.path.join(output_dir, f'configuration_patterns_{filename_suffix}.png'), dpi=150)
                    plt.close()
                    print(f"   ✅ Configuration patterns plot saved")
                    
                except Exception as e:
                    print(f"   ⚠️  Configuration patterns plot failed: {str(e)}")
                    plt.close()
            else:
                print(f"   ⚠️  Configuration patterns: insufficient data ({len(data_to_plot)} results)")
        
        else:
            # Fallback without pandas
            print("   ⚠️  Pandas not available, creating simple plots...")
            
            try:
                # Simple scatter plot without pandas
                plt.figure(figsize=(10, 6))
                
                accuracies = [r.cosine_similarity for r in successful_results if r.cosine_similarity > 0]
                speedups = [r.speedup for r in successful_results if r.speedup > 0]
                
                if len(accuracies) > 0 and len(speedups) > 0:
                    plt.scatter(accuracies[:len(speedups)], speedups[:len(accuracies)], alpha=0.7)
                    plt.axvline(x=self.config.accuracy_threshold, color='red', linestyle='--',
                               label=f'Perfect accuracy threshold ({self.config.accuracy_threshold})')
                    plt.xlabel('Cosine Similarity (Accuracy)')
                    plt.ylabel('Speedup')
                    plt.title('Speedup vs Accuracy Trade-off')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'speedup_vs_accuracy_simple.png'), dpi=150)
                    plt.close()
                    print(f"   ✅ Simple speedup vs accuracy plot saved")
                else:
                    print(f"   ⚠️  No valid data for simple plot")
                    plt.close()
            except Exception as e:
                print(f"   ⚠️  Simple plot failed: {str(e)}")
                plt.close()
        
        print(f"📊 Visualizations completed (saved to {output_dir})")


def generate_report(results: List[GridSearchResult], analysis: Dict[str, Any]) -> str:
    """Generate comprehensive report"""
    
    report = []
    report.append("=" * 80)
    report.append("🔍 MLX ATTENTION GRID SEARCH REPORT")
    report.append("=" * 80)
    
    # Summary
    summary = analysis.get('summary', {})
    report.append(f"\n📊 SUMMARY")
    report.append(f"   Total configurations tested: {summary.get('total_tested', 0)}")
    report.append(f"   Successful configurations: {summary.get('successful', 0)}")
    report.append(f"   Perfect accuracy configurations: {summary.get('perfect_accuracy_count', 0)}")
    report.append(f"   Perfect accuracy rate: {summary.get('perfect_accuracy_rate', 0):.1%}")
    
    if summary.get('perfect_accuracy_count', 0) > 0:
        report.append(f"   Average speedup (perfect accuracy): {summary.get('avg_speedup_perfect', 0):.2f}x")
        report.append(f"   Maximum speedup (perfect accuracy): {summary.get('max_speedup_perfect', 0):.2f}x")
    
    # Optimal configurations by sequence length
    by_seq_len = analysis.get('by_sequence_length', {})
    if by_seq_len:
        report.append(f"\n🎯 OPTIMAL CONFIGURATIONS BY SEQUENCE LENGTH")
        report.append("-" * 60)
        
        for seq_len in sorted(by_seq_len.keys()):
            data = by_seq_len[seq_len]
            best = data['best_config']
            report.append(f"\n   📏 Sequence Length: {seq_len}")
            report.append(f"      Window Size: {best.window_size}")
            report.append(f"      Query Chunk Size: {best.query_chunk_size}")
            report.append(f"      Dilation Rate: {best.dilation_rate}")
            report.append(f"      Speedup: {best.speedup:.2f}x")
            report.append(f"      Accuracy: {best.cosine_similarity:.4f}")
            report.append(f"      Perfect configs available: {data['count']}")
    
    # Patterns and recommendations
    recommendations = analysis.get('recommendations', {})
    patterns = recommendations.get('patterns', {})
    
    if patterns:
        report.append(f"\n🔍 CONFIGURATION PATTERNS")
        report.append("-" * 60)
        report.append(f"   Most common window size: {patterns.get('common_window_size')}")
        report.append(f"   Most common chunk size: {patterns.get('common_chunk_size')}")
        report.append(f"   Most common dilation rate: {patterns.get('common_dilation')}")
    
    # Implementation recommendations
    optimal_configs = recommendations.get('optimal_configs', {})
    if optimal_configs:
        report.append(f"\n💡 IMPLEMENTATION RECOMMENDATIONS")
        report.append("-" * 60)
        
        report.append("   Use sequence-length-adaptive configuration:")
        report.append("   ```python")
        report.append("   def get_optimal_config(seq_len):")
        
        for seq_len in sorted(optimal_configs.keys()):
            config = optimal_configs[seq_len]
            condition = f"seq_len <= {seq_len}" if seq_len == min(optimal_configs.keys()) else f"seq_len <= {seq_len}"
            report.append(f"       if {condition}:")
            report.append(f"           return {{")
            report.append(f"               'window_size': {config['window_size']},")
            report.append(f"               'query_chunk_size': {config['query_chunk_size']},")
            report.append(f"               'dilation_rate': {config['dilation_rate']}")
            report.append(f"           }}  # Expected speedup: {config['speedup']:.2f}x")
        
        report.append("   ```")
    
    # Failed configurations analysis
    failed_results = [r for r in results if not r.success]
    if failed_results:
        report.append(f"\n⚠️  FAILED CONFIGURATIONS ANALYSIS")
        report.append("-" * 60)
        
        error_counts = {}
        for result in failed_results:
            error_type = result.error_message.split(':')[0] if ':' in result.error_message else result.error_message
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            report.append(f"   {error_type}: {count} occurrences")
    
    report.append(f"\n" + "=" * 80)
    
    return "\n".join(report)


def main():
    """Main grid search execution"""
    
    parser = argparse.ArgumentParser(description="MLX Attention Configuration Grid Search")
    parser.add_argument("--evolved-program", required=True,
                       help="Path to evolved attention program")
    parser.add_argument("--output-dir", default="grid_search_results",
                       help="Output directory for results")
    parser.add_argument("--checkpoint", 
                       help="Checkpoint file for resuming search")
    parser.add_argument("--quick", action="store_true",
                       help="Run a quick search with reduced parameters")
    parser.add_argument("--seq-lengths", nargs='+', type=int,
                       help="Sequence lengths to test (overrides default)")
    parser.add_argument("--window-sizes", nargs='+', type=int,
                       help="Window sizes to test (use -1 for None)")
    parser.add_argument("--accuracy-threshold", type=float, default=0.999,
                       help="Threshold for perfect accuracy")
    parser.add_argument("--benchmark-runs", type=int, default=5,
                       help="Number of benchmark runs per configuration")
    parser.add_argument("--timeout", type=int, default=30,
                       help="Timeout per configuration in seconds")
    parser.add_argument("--plot", action="store_true",
                       help="Generate visualization plots")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.evolved_program):
        print(f"❌ Evolved program not found: {args.evolved_program}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup configuration
    if args.quick:
        # Quick search for testing
        config = GridSearchConfig(
            sequence_lengths=[128, 512],
            window_sizes=[None, 64, 128],
            query_chunk_sizes=[128, 256],
            dilation_rates=[1, 2],
            benchmark_runs=3,
            timeout_seconds=15
        )
    else:
        # Full search
        config = GridSearchConfig.default()
        config.benchmark_runs = args.benchmark_runs
        config.timeout_seconds = args.timeout
        config.accuracy_threshold = args.accuracy_threshold
        
        # Override with command line arguments
        if args.seq_lengths:
            config.sequence_lengths = args.seq_lengths
        if args.window_sizes:
            # Convert -1 to None for window_size
            config.window_sizes = [None if x == -1 else x for x in args.window_sizes]
    
    # Setup checkpoint
    checkpoint_file = args.checkpoint
    if not checkpoint_file:
        checkpoint_file = os.path.join(args.output_dir, "grid_search_checkpoint.json")
    
    print(f"🚀 Starting MLX Attention Grid Search")
    print(f"   Evolved program: {args.evolved_program}")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Checkpoint file: {checkpoint_file}")
    print(f"   Estimated configurations: {config.estimate_total_configs()}")
    
    try:
        # Run grid search
        grid_search = AttentionGridSearch(config, args.evolved_program)
        results = grid_search.run_grid_search(checkpoint_file)
        
        # Analyze results
        analysis = grid_search.analyze_results()
        
        # Generate report
        report = generate_report(results, analysis)
        print(f"\n{report}")
        
        # Save results
        results_file = os.path.join(args.output_dir, "grid_search_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'config': config.__dict__,
                'results': [r.to_dict() for r in results],
                'analysis': analysis
            }, f, indent=2, default=str)
        print(f"💾 Results saved: {results_file}")
        
        # Save report
        report_file = os.path.join(args.output_dir, "grid_search_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"📄 Report saved: {report_file}")
        
        # Create visualizations
        if args.plot:
            grid_search.create_visualizations(args.output_dir, analysis)
        
        print(f"\n✅ Grid search complete!")
        
        # Return appropriate exit code
        perfect_count = analysis.get('summary', {}).get('perfect_accuracy_count', 0)
        if perfect_count == 0:
            print("❌ No configurations achieved perfect accuracy")
            return 1
        else:
            print(f"✅ Found {perfect_count} configurations with perfect accuracy")
            return 0
        
    except Exception as e:
        print(f"❌ Grid search failed: {str(e)}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
