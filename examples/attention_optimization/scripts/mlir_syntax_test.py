# #!/usr/bin/env python3
# """
# Quick test script to verify MLIR syntax is correct.
# """

# import subprocess
# import tempfile
# from pathlib import Path

# def test_mlir_syntax():
#     """Test the corrected MLIR baseline syntax"""
    
#     baseline_mlir = '''
# #map_q = affine_map<(b, h, s1, s2, d) -> (b, h, s1, d)>
# #map_k = affine_map<(b, h, s1, s2, d) -> (b, h, s2, d)>
# #map_scores = affine_map<(b, h, s1, s2, d) -> (b, h, s1, s2)>
# #map_weights = affine_map<(b, h, s1, s2) -> (b, h, s1, s2)>
# #map_v = affine_map<(b, h, s1, s2, d) -> (b, h, s2, d)>
# #map_out = affine_map<(b, h, s1, s2, d) -> (b, h, s1, d)>

# module {
#   func.func @baseline_attention(
#       %query: tensor<1x8x128x64xf32>,
#       %key: tensor<1x8x128x64xf32>, 
#       %value: tensor<1x8x128x64xf32>
#   ) -> tensor<1x8x128x64xf32> {
    
#     %c0 = arith.constant 0.0 : f32
#     %cst_scale = arith.constant 0.125 : f32
    
#     // Initialize output tensors
#     %scores_init = tensor.empty() : tensor<1x8x128x128xf32>
#     %output_init = tensor.empty() : tensor<1x8x128x64xf32>
    
#     // Compute Q @ K^T (scaled dot-product attention)
#     %attention_scores = linalg.generic {
#       indexing_maps = [#map_q, #map_k, #map_scores],
#       iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
#     } ins(%query, %key : tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>) 
#       outs(%scores_init : tensor<1x8x128x128xf32>) {
#     ^bb0(%q: f32, %k: f32, %acc: f32):
#       %prod = arith.mulf %q, %k : f32
#       %scaled = arith.mulf %prod, %cst_scale : f32
#       %sum = arith.addf %acc, %scaled : f32
#       linalg.yield %sum : f32
#     } -> tensor<1x8x128x128xf32>
    
#     // Apply attention weights to values  
#     %attention_output = linalg.generic {
#       indexing_maps = [#map_weights, #map_v, #map_out],
#       iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]
#     } ins(%attention_scores, %value : tensor<1x8x128x128xf32>, tensor<1x8x128x64xf32>) 
#       outs(%output_init : tensor<1x8x128x64xf32>) {
#     ^bb0(%weight: f32, %v: f32, %acc: f32):
#       %weighted = arith.mulf %weight, %v : f32
#       %sum = arith.addf %acc, %weighted : f32
#       linalg.yield %sum : f32
#     } -> tensor<1x8x128x64xf32>
    
#     return %attention_output : tensor<1x8x128x64xf32>
#   }
# }
# '''
    
#     try:
#         # Write MLIR to temporary file
#         with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
#             f.write(baseline_mlir)
#             temp_file = f.name
        
#         print("🔧 Testing MLIR baseline syntax...")
        
#         # Test basic parsing
#         result = subprocess.run([
#             "mlir-opt", temp_file
#         ], capture_output=True, text=True, timeout=30)
        
#         Path(temp_file).unlink()  # Clean up
        
#         if result.returncode == 0:
#             print("✅ MLIR baseline syntax is correct!")
#             return True
#         else:
#             print(f"❌ MLIR syntax error: {result.stderr}")
#             return False
            
#     except Exception as e:
#         print(f"❌ Test error: {e}")
#         return False

# def test_tiling_pass():
#     """Test the linalg tiling pass syntax"""
    
#     simple_linalg = '''
# #map = affine_map<(d0, d1) -> (d0, d1)>
# module {
#   func.func @simple_add(%arg0: tensor<128x64xf32>, %arg1: tensor<128x64xf32>) -> tensor<128x64xf32> {
#     %0 = tensor.empty() : tensor<128x64xf32>
#     %1 = linalg.generic {
#       indexing_maps = [#map, #map, #map],
#       iterator_types = ["parallel", "parallel"]
#     } ins(%arg0, %arg1 : tensor<128x64xf32>, tensor<128x64xf32>) 
#       outs(%0 : tensor<128x64xf32>) {
#     ^bb0(%in: f32, %in_1: f32, %out: f32):
#       %2 = arith.addf %in, %in_1 : f32
#       linalg.yield %2 : f32
#     } -> tensor<128x64xf32>
#     return %1 : tensor<128x64xf32>
#   }
# }
# '''
    
#     try:
#         # Write MLIR to temporary file
#         with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
#             f.write(simple_linalg)
#             temp_file = f.name
        
#         print("\n🔧 Testing linalg tiling pass...")
        
#         # Test tiling with our syntax
#         pipeline = "builtin.module(linalg-tile{linalg-tile-sizes=32,32},canonicalize,cse)"
#         result = subprocess.run([
#             "mlir-opt", temp_file, f"--pass-pipeline={pipeline}"
#         ], capture_output=True, text=True, timeout=30)
        
#         Path(temp_file).unlink()  # Clean up
        
#         if result.returncode == 0:
#             print("✅ Linalg tiling pass works!")
#             print("Sample output:")
#             print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
#             return True
#         else:
#             print(f"❌ Tiling pass error: {result.stderr}")
#             return False
            
#     except Exception as e:
#         print(f"❌ Test error: {e}")
#         return False

# if __name__ == "__main__":
#     print("🚀 Testing MLIR Syntax Corrections\n")
    
#     success1 = test_mlir_syntax()
#     success2 = test_tiling_pass()
    
#     if success1 and success2:
#         print("\n🎉 All MLIR syntax tests passed!")
#         print("✅ Ready to run AlphaEvolve evolution")
#     else:
#         print("\n⚠️ Some tests failed. Check MLIR installation.")
        
#     print("\n📋 If tests passed, run:")
#     print("python openevolve-run.py fixed_initial_program.py fixed_evaluator.py --iterations 10")


#!/usr/bin/env python3
"""
Quick test script to verify MLIR syntax is correct.
"""

import subprocess
import tempfile
from pathlib import Path

def test_mlir_syntax():
    """Test the corrected MLIR baseline syntax"""
    
    baseline_mlir = '''
#map_q = affine_map<(b, h, s1, s2, d) -> (b, h, s1, d)>
#map_k = affine_map<(b, h, s1, s2, d) -> (b, h, s2, d)>
#map_scores = affine_map<(b, h, s1, s2, d) -> (b, h, s1, s2)>
#map_weights = affine_map<(b, h, s1, s2) -> (b, h, s1, s2)>
#map_v = affine_map<(b, h, s1, s2, d) -> (b, h, s2, d)>
#map_out = affine_map<(b, h, s1, s2, d) -> (b, h, s1, d)>

module {
  func.func @baseline_attention(
      %query: tensor<1x8x128x64xf32>,
      %key: tensor<1x8x128x64xf32>, 
      %value: tensor<1x8x128x64xf32>
  ) -> tensor<1x8x128x64xf32> {
    
    %c0 = arith.constant 0.0 : f32
    %cst_scale = arith.constant 0.125 : f32
    
    // Initialize output tensors
    %scores_init = tensor.empty() : tensor<1x8x128x128xf32>
    %output_init = tensor.empty() : tensor<1x8x128x64xf32>
    
    // Compute Q @ K^T (scaled dot-product attention)
    %attention_scores = linalg.generic {
      indexing_maps = [#map_q, #map_k, #map_scores],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
    } ins(%query, %key : tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>) 
      outs(%scores_init : tensor<1x8x128x128xf32>) {
    ^bb0(%q: f32, %k: f32, %acc: f32):
      %prod = arith.mulf %q, %k : f32
      %scaled = arith.mulf %prod, %cst_scale : f32
      %sum = arith.addf %acc, %scaled : f32
      linalg.yield %sum : f32
    } -> tensor<1x8x128x128xf32>
    
    // Apply attention weights to values  
    %attention_output = linalg.generic {
      indexing_maps = [#map_weights, #map_v, #map_out],
      iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]
    } ins(%attention_scores, %value : tensor<1x8x128x128xf32>, tensor<1x8x128x64xf32>) 
      outs(%output_init : tensor<1x8x128x64xf32>) {
    ^bb0(%weight: f32, %v: f32, %acc: f32):
      %weighted = arith.mulf %weight, %v : f32
      %sum = arith.addf %acc, %weighted : f32
      linalg.yield %sum : f32
    } -> tensor<1x8x128x64xf32>
    
    return %attention_output : tensor<1x8x128x64xf32>
  }
}
'''
    
    try:
        # Write MLIR to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
            f.write(baseline_mlir)
            temp_file = f.name
        
        print("🔧 Testing MLIR baseline syntax...")
        
        # Test basic parsing
        result = subprocess.run([
            "mlir-opt", temp_file
        ], capture_output=True, text=True, timeout=30)
        
        Path(temp_file).unlink()  # Clean up
        
        if result.returncode == 0:
            print("✅ MLIR baseline syntax is correct!")
            return True
        else:
            print(f"❌ MLIR syntax error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_tiling_pass():
    """Test the linalg tiling pass syntax"""
    
    simple_linalg = '''
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @simple_add(%arg0: tensor<128x64xf32>, %arg1: tensor<128x64xf32>) -> tensor<128x64xf32> {
    %0 = tensor.empty() : tensor<128x64xf32>
    %1 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0, %arg1 : tensor<128x64xf32>, tensor<128x64xf32>) 
      outs(%0 : tensor<128x64xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.addf %in, %in_1 : f32
      linalg.yield %2 : f32
    } -> tensor<128x64xf32>
    return %1 : tensor<128x64xf32>
  }
}
'''
    
    try:
        # Write MLIR to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
            f.write(simple_linalg)
            temp_file = f.name
        
        print("\n🔧 Testing linalg tiling pass...")
        
        # Test tiling with our syntax
        pipeline = "builtin.module(linalg-tile,canonicalize,cse)"
        result = subprocess.run([
            "mlir-opt", temp_file, f"--pass-pipeline={pipeline}"
        ], capture_output=True, text=True, timeout=30)
        
        Path(temp_file).unlink()  # Clean up
        
        if result.returncode == 0:
            print("✅ Linalg tiling pass works!")
            print("Sample output:")
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
            return True
        else:
            print(f"❌ Tiling pass error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing MLIR Syntax Corrections\n")
    
    success1 = test_mlir_syntax()
    success2 = test_tiling_pass()
    
    if success1 and success2:
        print("\n🎉 All MLIR syntax tests passed!")
        print("✅ Ready to run AlphaEvolve evolution")
    else:
        print("\n⚠️ Some tests failed. Check MLIR installation.")
        
    print("\n📋 If tests passed, run:")
    print("python openevolve-run.py fixed_initial_program.py fixed_evaluator.py --iterations 10")