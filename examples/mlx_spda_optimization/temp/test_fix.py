#!/usr/bin/env python3
"""
Quick test to verify the MLX array update fix is working correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_fix():
    """Test that the array update issue is fixed."""
    print("🔧 Testing MLX Array Update Fix...")
    print("=" * 40)
    
    try:
        import mlx.core as mx
        import initial_program
        
        # Test the specific function that was failing
        attention_fn = initial_program.evolved_scaled_dot_product_attention
        
        print("Testing long sequence (1024 tokens) that was failing...")
        
        # Create test inputs
        q = mx.random.normal((1, 8, 1024, 64))
        k = mx.random.normal((1, 8, 1024, 64)) 
        v = mx.random.normal((1, 8, 1024, 64))
        scale = 0.125
        
        # This should work now without the ArrayAt error
        output = attention_fn(q, k, v, scale=scale)
        
        print(f"✅ SUCCESS: Output shape = {output.shape}")
        
        # Check for valid output
        has_nan = bool(mx.any(mx.isnan(output)))
        has_inf = bool(mx.any(mx.isinf(output)))
        
        if not has_nan and not has_inf:
            print("✅ Valid output (no NaN/Inf)")
            return True
        else:
            print(f"❌ Invalid output: NaN={has_nan}, Inf={has_inf}")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fix()
    if success:
        print("\n🎉 Fix verified! The system should now work correctly.")
        print("You can run 'python test_system.py' for full verification.")
    else:
        print("\n❌ Fix not working. Please check the error above.")
    
    sys.exit(0 if success else 1)
