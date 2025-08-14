#!/usr/bin/env python3
"""
Script to generate OpenEvolve task files for all AlgoTune tasks.

This script uses the AlgoTuneTaskAdapter to convert all available AlgoTune tasks
to OpenEvolve format using external AlgoTune repositories.
"""

import sys
import argparse
from pathlib import Path
from task_adapter import AlgoTuneTaskAdapter

def main():
    """Main function to generate OpenEvolve task files."""
    
    parser = argparse.ArgumentParser(description="Generate OpenEvolve task files from AlgoTune tasks")
    parser.add_argument(
        "--algotune-path",
        type=str,
        required=True,
        help="Path to AlgoTune repository directory (e.g., /path/to/AlgoTune)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available tasks"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for generated files (default: task name subdirectories)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize the adapter with AlgoTune path
        adapter = AlgoTuneTaskAdapter(algotune_path=args.algotune_path)
        
        # List available tasks
        available_tasks = adapter.list_available_tasks()
        print(f"Found {len(available_tasks)} available AlgoTune tasks")
        
        if args.list:
            print("\nAvailable tasks:")
            for i, task_name in enumerate(available_tasks, 1):
                print(f"  {i:3d}. {task_name}")
            return 0
        
        # Generate files for all tasks
        tasks_to_process = available_tasks
        print(f"\nGenerating OpenEvolve files for {len(tasks_to_process)} tasks...")
        
        successful = 0
        failed = 0
        
        for i, task_name in enumerate(tasks_to_process, 1):
            try:
                print(f"[{i:3d}/{len(tasks_to_process)}] Processing {task_name}...")
                output_dir = adapter.create_task_files(task_name, args.output_dir)
                print(f"  ✅ Success: {output_dir}")
                successful += 1
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                failed += 1
        
        print(f"\nSummary:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total: {len(tasks_to_process)}")
        
        if failed > 0:
            print(f"\nSome tasks failed to generate. Check the errors above.")
            return 1
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nExample usage:")
        print("  python generate_all_tasks.py --algotune-path /path/to/AlgoTune")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 