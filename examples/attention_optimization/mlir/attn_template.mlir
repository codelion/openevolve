// Template MLIR attention implementation
// Copy this to attn.mlir and customize as needed
// This will be used as the baseline for optimization

#map_q = affine_map<(b, h, s1, s2, d) -> (b, h, s1, d)>
#map_k = affine_map<(b, h, s1, s2, d) -> (b, h, s2, d)>
#map_scores = affine_map<(b, h, s1, s2, d) -> (b, h, s1, s2)>
#map_attn_in = affine_map<(b, h, s1, s2, d) -> (b, h, s1, s2)>
#map_value_in = affine_map<(b, h, s1, s2, d) -> (b, h, s2, d)>
#map_output = affine_map<(b, h, s1, s2, d) -> (b, h, s1, d)>

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
    
    // Apply attention weights to values (matmul: scores @ values)
    %attention_output = linalg.generic {
      indexing_maps = [#map_attn_in, #map_value_in, #map_output],
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