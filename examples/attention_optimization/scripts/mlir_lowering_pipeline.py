#!/usr/bin/env python3
"""
MLIR Lowering Pipeline - Use mlir-opt to lower arith operations to LLVM
Uses proper MLIR lowering passes to convert all dialects to LLVM-compatible ones.
"""

import subprocess
import tempfile
import shutil
from pathlib import Path
import time

class MLIRLoweringPipeline:
    def __init__(self):
        self.verify_tools()
        
    def verify_tools(self):
        """Verify required MLIR tools"""
        required_tools = ['mlir-opt', 'mlir-translate']
        for tool in required_tools:
            if not shutil.which(tool):
                raise RuntimeError(f"Required tool not found: {tool}")
        print("✅ MLIR tools verified: mlir-opt, mlir-translate")

    def find_available_passes(self):
        """Find what lowering passes are available"""
        print("🔍 Finding available lowering passes...")
        
        try:
            result = subprocess.run(['mlir-opt', '--help'], capture_output=True, text=True)
            help_text = result.stdout
            
            # Look for conversion passes
            conversion_passes = []
            for line in help_text.splitlines():
                line = line.strip()
                if 'convert-' in line and '-to-' in line:
                    # Extract pass name
                    if line.startswith('--'):
                        pass_name = line.split()[0][2:]  # Remove --
                        conversion_passes.append(pass_name)
            
            print("📋 Available conversion passes:")
            relevant_passes = []
            for pass_name in sorted(conversion_passes):
                if any(keyword in pass_name for keyword in ['arith', 'func', 'llvm', 'std', 'scf']):
                    print(f"   ✅ {pass_name}")
                    relevant_passes.append(pass_name)
                else:
                    print(f"   ❓ {pass_name}")
            
            return relevant_passes
            
        except Exception as e:
            print(f"❌ Error finding passes: {e}")
            return []

    def test_lowering_passes(self, input_file):
        """Test different lowering pass combinations"""
        print(f"\n🧪 Testing lowering passes on {input_file}...")
        
        # Common lowering pass sequences
        pass_sequences = [
            # Basic arith lowering
            ["convert-arith-to-llvm"],
            
            # More comprehensive lowering
            ["convert-arith-to-llvm", "convert-func-to-llvm"],
            
            # Full lowering pipeline
            [
                "convert-arith-to-llvm",
                "convert-func-to-llvm", 
                "convert-scf-to-cf",
                "convert-cf-to-llvm"
            ],
            
            # Alternative approaches
            ["arith-bufferize", "convert-arith-to-llvm"],
            ["canonicalize", "convert-arith-to-llvm", "canonicalize"],
            
            # Try with reconcile-unrealized-casts
            [
                "convert-arith-to-llvm",
                "convert-func-to-llvm",
                "reconcile-unrealized-casts"
            ]
        ]
        
        successful_sequences = []
        
        for i, passes in enumerate(pass_sequences):
            print(f"\n📋 Testing sequence {i+1}: {' → '.join(passes)}")
            
            success = self.test_pass_sequence(input_file, passes)
            if success:
                successful_sequences.append(passes)
                print(f"   ✅ Sequence {i+1} works!")
            else:
                print(f"   ❌ Sequence {i+1} failed")
        
        return successful_sequences

    def test_pass_sequence(self, input_file, passes):
        """Test a specific sequence of passes"""
        try:
            # Build pipeline
            pipeline = f"builtin.module({','.join(passes)})"
            
            with tempfile.NamedTemporaryFile(suffix='.mlir', delete=False) as temp_file:
                # Apply passes
                cmd = ['mlir-opt', input_file, f'--pass-pipeline={pipeline}']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                
                if result.returncode != 0:
                    return False
                
                # Write result to temp file
                temp_file.write(result.stdout)
                temp_file.flush()
                
                # Test LLVM translation
                cmd = ['mlir-translate', '--mlir-to-llvmir', temp_file.name]
                translate_result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                success = translate_result.returncode == 0
                if success:
                    print(f"      💡 LLVM IR size: {len(translate_result.stdout)} chars")
                
                return success
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
            return False
        finally:
            try:
                Path(temp_file.name).unlink()
            except:
                pass

    def create_lowered_file(self, input_file, output_file, pass_sequence):
        """Create a fully lowered MLIR file"""
        print(f"\n🚀 Creating lowered file: {input_file} → {output_file}")
        print(f"📋 Using passes: {' → '.join(pass_sequence)}")
        
        try:
            # Build pipeline
            pipeline = f"builtin.module({','.join(pass_sequence)})"
            
            start_time = time.time()
            cmd = ['mlir-opt', input_file, f'--pass-pipeline={pipeline}', '-o', output_file]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            elapsed = time.time() - start_time
            
            if result.returncode != 0:
                print(f"❌ Lowering failed: {result.stderr}")
                return False
            
            print(f"✅ Lowering completed in {elapsed:.3f}s")
            
            # Verify the output
            output_path = Path(output_file)
            if output_path.exists():
                size = output_path.stat().st_size
                print(f"📄 Output file size: {size} bytes")
                
                # Test LLVM translation
                cmd = ['mlir-translate', '--mlir-to-llvmir', output_file]
                translate_result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                
                if translate_result.returncode == 0:
                    llvm_size = len(translate_result.stdout)
                    print(f"✅ LLVM translation successful! LLVM IR size: {llvm_size} chars")
                    
                    # Save LLVM IR too
                    llvm_file = output_file.replace('.mlir', '.ll')
                    with open(llvm_file, 'w') as f:
                        f.write(translate_result.stdout)
                    print(f"💾 LLVM IR saved to: {llvm_file}")
                    
                    return True
                else:
                    print(f"❌ LLVM translation failed: {translate_result.stderr[:200]}...")
                    return False
            
            return False
            
        except Exception as e:
            print(f"❌ Error creating lowered file: {e}")
            return False

    def process_file(self, input_file):
        """Complete pipeline to lower an MLIR file"""
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"❌ Input file not found: {input_file}")
            return None
        
        print(f"🎯 Processing {input_file}")
        print(f"📊 Input size: {input_path.stat().st_size} bytes")
        
        # Find available passes
        available_passes = self.find_available_passes()
        
        # Test lowering approaches
        successful_sequences = self.test_lowering_passes(str(input_path))
        
        if not successful_sequences:
            print("❌ No working lowering sequences found!")
            return None
        
        # Use the first successful sequence
        best_sequence = successful_sequences[0]
        print(f"\n🎯 Using best sequence: {' → '.join(best_sequence)}")
        
        # Create output filename
        output_file = str(input_path.parent / f"{input_path.stem}_lowered{input_path.suffix}")
        
        # Create the lowered file
        if self.create_lowered_file(str(input_path), output_file, best_sequence):
            print(f"🎉 Success! Lowered file created: {output_file}")
            return output_file
        else:
            print("❌ Failed to create lowered file")
            return None

def main():
    print("🚀 MLIR Lowering Pipeline")
    print("=" * 50)
    
    pipeline = MLIRLoweringPipeline()
    
    # Process your attention file
    input_file = "mlir/self_attn_with_consts_linalg_dialect.mlir"
    # input_file = "mlir/export_mlir.mlir"
    
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        print("Please specify the correct path to your MLIR file.")
        return
    
    lowered_file = pipeline.process_file(input_file)
    
    if lowered_file:
        print(f"\n🎯 Next steps:")
        print(f"1. Update your evaluator to use: {lowered_file}")
        print(f"2. The lowered file should work with mlir-translate")
        print(f"3. Run evolution with real LLVM execution!")
        print(f"\n📋 Quick test:")
        print(f"   mlir-translate --mlir-to-llvmir {lowered_file}")
    else:
        print("\n⚠️ Lowering failed. You may need to:")
        print("1. Check which conversion passes are available in your MLIR build")
        print("2. Manually inspect the MLIR file for unsupported constructs")
        print("3. Use alternative approaches like the dialect converter")

if __name__ == "__main__":
    main()