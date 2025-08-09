#!/usr/bin/env python3
"""
Script to create OpenEvolve task files from AlgoTune tasks.

This script demonstrates how to use the AlgoTuneTaskAdapter to convert
AlgoTune tasks to OpenEvolve format using external AlgoTune repositories.
"""

import sys
import argparse
from pathlib import Path
from task_adapter import AlgoTuneTaskAdapter

def main():
    """Main function to create OpenEvolve task files."""
    
    parser = argparse.ArgumentParser(description="Create OpenEvolve task files from AlgoTune tasks")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name to create OpenEvolve files for"
    )
    
    parser.add_argument(
        "--algotune-path",
        type=str,
        required=True,
        help="Path to AlgoTune repository directory (e.g., /path/to/AlgoTune)"
    )

    args = parser.parse_args()
    
    try:
        # Initialize the adapter with AlgoTune path
        adapter = AlgoTuneTaskAdapter(algotune_path=args.algotune_path, task=args.task)
        
        # List available tasks
        available_tasks = adapter.list_available_tasks()
        print(f"Available tasks: {len(available_tasks)}")
        print(available_tasks)
        
        task_name = args.task
        
        # Check if the task exists
        if task_name not in available_tasks:
            print(f"Error: Task '{task_name}' not found")
            print(f"Available tasks: {available_tasks[:10]}...")
            return 1
        
        # Create the OpenEvolve files
        print(f"\nCreating OpenEvolve files for task: {task_name}")
        output_dir = adapter.create_task_files(task_name)
        print(f"✅ Created files in: {output_dir}")
        
        # List the created files
        output_path = Path(output_dir)
        created_files = list(output_path.glob("*"))
        print("Created files:")
        for file in created_files:
            print(f"  - {file.name}")
        
        print(f"\n✅ Successfully created OpenEvolve files for '{task_name}'")
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nExample usage:")
        print("  python create_task.py --algotune-path /path/to/AlgoTune --task svm")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 