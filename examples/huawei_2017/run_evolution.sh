#!/bin/bash
# Script to run OpenEvolve on Huawei 2017 CDN optimization problem

set -e

echo "Starting OpenEvolve evolution for Huawei CodeCraft 2017 CDN Optimization"
echo "========================================================================"
echo ""

# Check if running from correct directory
if [ ! -f "initial_program.py" ]; then
    echo "Error: Must run from examples/huawei_2017 directory"
    exit 1
fi

# Check if case examples exist
if [ ! -d "case_example" ]; then
    echo "Error: case_example directory not found"
    exit 1
fi

# Default values
ITERATIONS=300
CONFIG="config.qwen.yaml"
CHECKPOINT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --iterations N      Number of evolution iterations (default: 300)"
            echo "  --config FILE       Config file to use (default: config.qwen.yaml)"
            echo "  --checkpoint DIR    Resume from checkpoint directory"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build command
CMD="python ../../openevolve-run.py initial_program.py evaluator.py --config $CONFIG --iterations $ITERATIONS"

if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

echo "Configuration:"
echo "  Iterations: $ITERATIONS"
echo "  Config: $CONFIG"
if [ -n "$CHECKPOINT" ]; then
    echo "  Checkpoint: $CHECKPOINT"
fi
echo ""
echo "Running: $CMD"
echo ""

# Run evolution
$CMD

echo ""
echo "Evolution complete! Check openevolve_output/ for results."
echo "Best solution: openevolve_output/best/best_program.py"
