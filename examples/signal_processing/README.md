# Real-Time Adaptive Signal Processing Algorithm Evolution

This example demonstrates how to use OpenEvolve to automatically discover and optimize real-time signal processing algorithms for non-stationary time series data. The challenge involves developing algorithms that can filter volatile signals while preserving important dynamics and minimizing computational latency.

## Problem Overview

### The Challenge
We need to develop a real-time signal processing algorithm that can:

1. **Filter noise** from volatile, non-stationary time series data
2. **Preserve genuine signal dynamics** and trend changes
3. **Minimize spurious directional reversals** caused by noise
4. **Achieve near-zero phase delay** for real-time applications
5. **Operate efficiently** within computational constraints

### Input Signal Characteristics
- **Type**: Univariate time series (1D array of real-valued samples)
- **Properties**:
  - Non-linear dynamics
  - Non-stationary statistical properties  
  - Aperiodic (non-seasonal) behavior
  - High frequency variability and volatility
  - Rapidly changing spectral characteristics

### Technical Constraints
- **Causal Processing**: Must use finite-length sliding window
- **Fixed Latency**: Output length = Input length - Window size
- **Real-time Capability**: Process samples as they arrive
- **Memory Efficiency**: Bounded memory usage

## Multi-Objective Optimization Framework

The algorithm performance is evaluated using a composite metric based on the research specification:

### Optimization Function
```
J(θ) = α₁·S(θ) + α₂·L_recent(θ) + α₃·L_avg(θ) + α₄·R(θ)
```

Where:
- **S(θ)**: **Slope Change Penalty** - Counts directional reversals in the filtered signal
- **L_recent(θ)**: **Instantaneous Lag Error** - |y[n] - x[n]| at the most recent sample
- **L_avg(θ)**: **Average Tracking Error** - Mean absolute error over the processing window
- **R(θ)**: **False Reversal Penalty** - Trend changes that don't match the clean signal
- **Weighting coefficients**: α₁=0.3, α₂=α₃=0.2, α₄=0.3

### Additional Evaluation Metrics
- **Signal Fidelity**: Correlation with ground truth clean signal
- **Noise Reduction**: Improvement in signal-to-noise ratio
- **Computational Efficiency**: Processing time per sample
- **Robustness**: Consistent performance across diverse signal types

## Proposed Algorithmic Approaches

The initial implementation provides a foundation that evolution can improve upon:

### 1. Baseline Implementation
- Simple moving average filter
- Weighted exponential moving average

### 2. Potential Advanced Techniques (for evolution to discover)
- **Adaptive Filtering**: Kalman filters, particle filters, adaptive weights
- **Multi-Scale Processing**: Wavelet decomposition, empirical mode decomposition
- **Predictive Enhancement**: Local polynomial fitting, neural network prediction
- **Trend Detection**: Change point detection, momentum indicators
- **Hybrid Approaches**: Ensemble methods combining multiple techniques

## File Structure

```
signal_processing/
├── README.md              # This documentation
├── config.yaml           # OpenEvolve configuration
├── initial_program.py     # Initial signal processing implementation
├── evaluator.py          # Multi-objective evaluation system
├── requirements.txt       # Python dependencies
└── results/              # Generated results (after running)
```

## How to Run

### Prerequisites
1. Install OpenEvolve and its dependencies
2. Install example-specific requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your LLM API key (e.g., `OPENAI_API_KEY` environment variable)

### Testing the Setup (Recommended)
First, validate that everything is working correctly:
```bash
cd examples/signal_processing
python test_setup.py
```

This will test the initial implementation and evaluator to ensure everything is ready for evolution.

### Running the Evolution
From the OpenEvolve root directory:
```bash
python openevolve-run.py examples/signal_processing/config.yaml
```

Or from the signal_processing directory:
```bash
python ../../openevolve-run.py config.yaml
```

### Monitoring Progress
The evolution will create an `openevolve_output` directory containing:
- **Checkpoints**: Saved population states at regular intervals
- **Logs**: Detailed evolution progress and metrics
- **Best Programs**: Top-performing algorithm implementations

## Understanding the Results

### Key Metrics to Watch
1. **Overall Score**: Primary selection metric (higher is better)
2. **Composite Score**: The main J(θ) optimization function
3. **Correlation**: How well the filtered signal matches the clean ground truth
4. **Noise Reduction**: Improvement in signal quality
5. **Slope Changes**: Number of directional reversals (lower is better)
6. **Success Rate**: Fraction of test signals processed successfully

### Actual Evolution Patterns (Observed)
- **Early iterations (1-10)**: Discovered Savitzky-Golay filtering with adaptive polynomial order
- **Mid evolution (10-100)**: Parameter optimization and performance stabilization around 0.37 score
- **Advanced stages (100-130)**: Breakthrough to full Kalman Filter implementation with state-space modeling

## Test Signal Characteristics

The evaluator uses 5 different synthetic test signals to ensure robustness:

1. **Smooth Sinusoidal**: Basic sinusoid with linear trend
2. **Multi-Frequency**: Multiple frequency components combined
3. **Non-Stationary**: Frequency-modulated signal
4. **Step Changes**: Sudden level changes to test responsiveness
5. **Random Walk**: Stochastic process with trend

Each signal has different noise levels and lengths to test algorithm adaptability.

## Initial Algorithm Analysis

The starting point includes:
- **Basic moving average**: Simple but may over-smooth
- **Weighted moving average**: Emphasizes recent samples
- **Exponential weighting**: Exponentially decaying weights for trend preservation

This provides a baseline that evolution can significantly improve upon by discovering:
- Advanced filtering techniques
- Adaptive parameter adjustment
- Multi-scale processing
- Predictive elements
- Robust trend detection

## Interpreting Evolution Results

### Successful Evolution Indicators
- **Decreasing slope changes**: Algorithm learns to reduce noise-induced reversals
- **Improving correlation**: Better preservation of true signal structure
- **Balanced metrics**: Good performance across all test signals
- **Stable improvements**: Consistent gains over multiple iterations

### Common Evolution Discoveries
- **Adaptive window sizing**: Dynamic adjustment based on signal characteristics
- **Multi-pass filtering**: Combining multiple filtering stages
- **Outlier detection**: Identifying and handling anomalous samples
- **Frequency analysis**: Spectral-based filtering decisions
- **Predictive elements**: Using future sample prediction to reduce lag

## Configuration Options

Key parameters in `config.yaml`:
- **max_iterations**: Total evolution steps (200 recommended)
- **population_size**: Number of candidate algorithms (80)
- **cascade_thresholds**: Quality gates for evaluation stages
- **system_message**: Guides LLM toward signal processing expertise

## Extending the Example

### Adding New Test Signals
Modify `generate_test_signals()` in `evaluator.py` to include:
- Real-world datasets (financial, sensor, biomedical)
- Domain-specific signal characteristics
- Different noise models and intensities

### Customizing Evaluation Metrics
Adjust weights in the composite function or add new metrics:
- Phase delay measurement
- Spectral preservation
- Computational complexity analysis
- Memory usage optimization

### Advanced Algorithmic Constraints
Modify the evolution block to explore:
- Specific filtering architectures
- Hardware-optimized implementations
- Online learning capabilities
- Multi-channel processing

## Research Applications

This framework can be adapted for various domains:
- **Financial Markets**: High-frequency trading signal processing
- **Biomedical Engineering**: Real-time biosignal filtering
- **Sensor Networks**: Environmental monitoring and noise reduction
- **Control Systems**: Real-time feedback signal conditioning
- **Communications**: Adaptive signal processing for wireless systems

## Actual Evolution Results ✨

**After 130 iterations, OpenEvolve achieved a major algorithmic breakthrough!**

### Key Discoveries:
- **🎯 Full Kalman Filter Implementation**: Complete state-space modeling with position-velocity tracking
- **📈 23% Performance Improvement**: Composite score improved from ~0.30 to 0.3712
- **⚡ 2x Faster Execution**: Optimized from 20ms to 11ms processing time
- **🔧 Advanced Parameter Tuning**: Discovered optimal noise covariance matrices

### Evolution Timeline:
1. **Early Stage (1-10 iterations)**: Discovered Savitzky-Golay adaptive filtering
2. **Mid Evolution (10-100)**: Parameter optimization and technique refinement  
3. **Breakthrough (100-130)**: Full Kalman Filter with adaptive initialization

### Final Performance Metrics:
- **Composite Score**: 0.3712 (multi-objective optimization function)
- **Slope Changes**: 322.8 (19% reduction in spurious reversals)
- **Correlation**: 0.147 (22% improvement in signal fidelity)
- **Lag Error**: 0.914 (24% reduction in responsiveness delay)


### Algorithmic Sophistication Achieved:
```python
# Discovered Kalman Filter with optimized parameters:
class KalmanFilter:
    def __init__(self, sigma_a_sq=1.0, measurement_noise=0.04):
        # State transition for constant velocity model
        self.F = np.array([[1, dt], [0, 1]])
        # Optimized process noise (100x improvement)
        self.Q = G @ G.T * sigma_a_sq  
        # Tuned measurement trust (55% improvement)
        self.R = measurement_noise
```

The evolved solution demonstrates that **automated algorithm discovery can achieve expert-level signal processing implementations**, discovering sophisticated techniques like Kalman filtering and optimal parameter combinations that would typically require months of engineering effort.
