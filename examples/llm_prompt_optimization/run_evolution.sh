#!/bin/bash
# Wrapper script to run OpenEvolve with the correct dataset

if [ $# -lt 1 ]; then
    echo "Usage: $0 <prompt_file> [additional_args...]"
    echo "Example: $0 emotion_prompt.txt --iterations 50"
    exit 1
fi

PROMPT_FILE=$1
shift  # Remove first argument

# Set the environment variable for the evaluator
export OPENEVOLVE_PROMPT=$PROMPT_FILE

# Run OpenEvolve
python ../../openevolve-run.py "$PROMPT_FILE" evaluator.py --config config.yaml "$@"