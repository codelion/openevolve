#!/bin/bash

# Define the number of problems for each split
declare -A split_counts=(
    ["bio_pop_growth"]=24
    ["chem_react"]=36
    ["matsci"]=25
    ["phys_osc"]=44
)

declare -A split_problem_dir_prefixes=(
    ["bio_pop_growth"]="BPG"  
    ["chem_react"]="CRK"       
    ["matsci"]="MatSci"           
    ["phys_osc"]="PO"         
)

base_problems_dir="./problems"

echo "Starting all experiments..."

for split_name in "${!split_counts[@]}"; do
    count=${split_counts[$split_name]}
    problem_dir_prefix=${split_problem_dir_prefixes[$split_name]}

    # Check if a prefix is defined (it can be an empty string if paths are like "split_name/0/")
    if [ -z "$problem_dir_prefix" ] && [ "${split_problem_dir_prefixes[$split_name]+_}" != "_" ]; then
        # This means the key exists but the value is an empty string, which is allowed.
        : # Do nothing, empty prefix is fine.
    elif [ -z "$problem_dir_prefix" ]; then
        echo ""
        echo "Warning: No problem directory prefix defined for split '$split_name' in 'split_problem_dir_prefixes'. Skipping this split."
        continue
    fi

    echo ""
    echo "----------------------------------------------------"
    echo "Processing Split: $split_name"
    echo "Number of problems: $count"
    echo "Problem directory prefix: '$problem_dir_prefix'" # Prefix like CRK, BPG, etc.
    echo "Expected problem path structure: $base_problems_dir/$split_name/${problem_dir_prefix}[ID]/"
    echo "----------------------------------------------------"

    # Loop from problem_id 0 to count-1
    for (( i=0; i<count; i++ )); do
        # Construct the path to the specific problem's directory
        # e.g., ./examples/symbolic_regression/problems/chem_react/CRK0
        problem_dir="$base_problems_dir/$split_name/$problem_dir_prefix$i"

        initial_program_path="$problem_dir/initial_program.py"
        evaluator_path="$problem_dir/evaluator.py"
        config_path="$problem_dir/config.yaml" # Assumes 'config.yaml' as in your original script

        # --- Sanity checks for file existence (optional but recommended) ---
        if [[ ! -f "$initial_program_path" ]]; then
            echo "  [Problem $i] SKIPPING: Initial program not found at $initial_program_path"
            continue
        fi
        if [[ ! -f "$evaluator_path" ]]; then
            echo "  [Problem $i] SKIPPING: Evaluator not found at $evaluator_path"
            continue
        fi
        if [[ ! -f "$config_path" ]]; then
            echo "  [Problem $i] SKIPPING: Config file not found at $config_path"
            continue
        fi
        # --- End Sanity checks ---

        echo "  Launching $split_name - Problem $i ($initial_program_path)"
        # Run the experiment in the background
        cmd="python ../../openevolve-run.py "$initial_program_path" "$evaluator_path" --config "$config_path" --iterations 200"
        eval $cmd &
    done
    wait    # let's do split by split
done

echo ""
echo "All experiment processes have been launched in the background."
echo "Waiting for all background processes to complete..."
wait
echo ""
echo "All experiments have completed."