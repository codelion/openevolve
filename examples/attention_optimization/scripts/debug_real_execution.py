#!/usr/bin/env python3
"""
Debug script to test MLIR real execution capabilities
"""

import subprocess
import tempfile
import shutil
from pathlib import Path

def check_mlir_tools():
    """Check what MLIR tools are available"""
    tools = [
        'mlir-opt',
        'mlir-translate', 
        'mlir-cpu-runner',
        'mlir-lsp-server',
        'clang',
        'gcc'
    ]
    
    print("🔍 Checking available tools:")
    available = {}
    for tool in tools:
        path = shutil.which(tool)
        available[tool] = path is not None
        status = "✅" if path else "❌"
        print(f"  {status} {tool}: {path or 'Not found'}")
    
    return available

def test_mlir_translate():
    """Test MLIR to LLVM translation"""
    print("\n🧪 Testing MLIR→LLVM translation:")
    
    # Simple test MLIR
    test_mlir = '''
module {
  func.func @simple_add(%arg0: f32, %arg1: f32) -> f32 {
    %0 = arith.addf %arg0, %arg1 : f32
    return %0 : f32
  }
}
    '''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        f.write(test_mlir)
        f.flush()
        
        try:
            # Test mlir-translate
            cmd = ['mlir-translate', '--mlir-to-llvmir', f.name]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ mlir-translate works!")
                print(f"   LLVM IR size: {len(result.stdout)} chars")
                return True
            else:
                print("❌ mlir-translate failed:")
                print(f"   Error: {result.stderr}")
                return False
                
        except FileNotFoundError:
            print("❌ mlir-translate not found")
            return False
        except Exception as e:
            print(f"❌ mlir-translate error: {e}")
            return False

def test_actual_mlir_file():
    """Test with your actual MLIR file"""
    print("\n🧪 Testing your actual MLIR file:")
    
    mlir_file = Path("mlir/self_attn_with_consts_linalg_dialect.mlir")
    if not mlir_file.exists():
        print("❌ MLIR file not found!")
        return False
    
    try:
        # Test basic parsing
        cmd = ['mlir-opt', str(mlir_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ MLIR file parses correctly")
            
            # Test optimization
            cmd = ['mlir-opt', str(mlir_file), '--canonicalize']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Basic optimization works")
                
                # Test LLVM translation
                if shutil.which('mlir-translate'):
                    cmd = ['mlir-translate', '--mlir-to-llvmir', str(mlir_file)]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print("✅ LLVM translation works!")
                        print(f"   LLVM IR size: {len(result.stdout)} chars")
                        return True
                    else:
                        print("❌ LLVM translation failed:")
                        print(f"   Error: {result.stderr[:500]}...")
                        return False
                else:
                    print("⚠️ mlir-translate not available")
                    return False
            else:
                print("❌ Basic optimization failed:")
                print(f"   Error: {result.stderr}")
                return False
        else:
            print("❌ MLIR file parsing failed:")
            print(f"   Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing MLIR file: {e}")
        return False

def suggest_fixes():
    """Suggest ways to enable real execution"""
    print("\n💡 Suggestions to enable real execution:")
    
    available = check_mlir_tools()
    
    if not available.get('mlir-translate'):
        print("1. Install mlir-translate:")
        print("   - Build LLVM/MLIR with: cmake -DLLVM_ENABLE_PROJECTS='mlir' ...")
        print("   - Or install via package manager if available")
    
    if not available.get('clang') and not available.get('gcc'):
        print("2. Install a C compiler (clang or gcc)")
    
    print("3. Alternative: Improve the simulation")
    print("   - Use more sophisticated IR analysis")
    print("   - Measure compilation time more accurately")
    print("   - Add pass-specific performance heuristics")

def main():
    print("🚀 MLIR Real Execution Debug Tool")
    print("=" * 50)
    
    available = check_mlir_tools()
    
    if available.get('mlir-translate'):
        if test_mlir_translate():
            test_actual_mlir_file()
    
    suggest_fixes()
    
    print("\n🎯 Quick fixes for better performance measurement:")
    print("1. Use compilation time as a proxy for optimization effectiveness")
    print("2. Analyze IR characteristics (instruction count, loop nesting)")  
    print("3. Implement pass-specific performance models")

if __name__ == "__main__":
    main()