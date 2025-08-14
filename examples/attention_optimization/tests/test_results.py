#!/usr/bin/env python3
"""
Quick test to verify the setup is working correctly.
"""

import json
import subprocess
import sys
from pathlib import Path

def test_initial_program():
    """Test that initial_program.py works"""
    print("Testing initial_program.py...")
    
    try:
        result = subprocess.run([sys.executable, "initial_program.py"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            output = json.loads(result.stdout)
            print(f"✅ Initial program works. Params: {len(output['params'])} parameters")
            return True
        else:
            print(f"❌ Initial program failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Initial program error: {e}")
        return False

def test_evaluator():
    """Test that evaluator.py works"""
    print("Testing evaluator.py...")
    
    try:
        result = subprocess.run([sys.executable, "evaluator.py", "initial_program.py"], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            output = json.loads(result.stdout)
            if "score" in output:
                print(f"✅ Evaluator works. Score: {output['score']:.3f}")
                return True
            else:
                print(f"❌ Evaluator missing score: {output}")
                return False
        else:
            print(f"❌ Evaluator failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Evaluator error: {e}")
        return False

def test_mlir_file():
    """Test that MLIR file exists and is valid"""
    print("Testing MLIR baseline file...")
    
    mlir_file = Path("mlir/baseline_attention.mlir")
    if mlir_file.exists():
        content = mlir_file.read_text()
        if "func.func @baseline_attention" in content:
            print("✅ MLIR file exists and looks valid")
            return True
        else:
            print("❌ MLIR file missing expected content")
            return False
    else:
        print("❌ MLIR file not found")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing OpenEvolve attention optimization setup...")
    print("=" * 50)
    
    tests = [
        test_mlir_file,
        test_initial_program, 
        test_evaluator
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 Setup is ready! You can now run:")
        print("python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml --iterations 10")
    else:
        print("❌ Setup needs fixing before running evolution")
    
    return passed == len(tests)

if __name__ == "__main__":
    main()