module {
  func.func @baseline_attention(
      %query: tensor<1x8x128x64xf32>,
      %key: tensor<1x8x128x64xf32>, 
      %value: tensor<1x8x128x64xf32>
  ) -> tensor<1x8x128x64xf32> {
    
    %c0 = arith.constant 0.0 : f32
    
    // Initialize output tensors
    %scores_init = tensor.empty() : tensor<1x8x128x128xf32>
    %output_init = tensor.empty() : tensor<1x8x128x64xf32>
    
    // Compute Q @ K^T (simplified for real compilation)
    %attention_scores = linalg.generic {
    // linalg.generic {
      indexing_maps = [
        affine_map<(b, h, s1, s2, d) -> (b, h, s1, d)>,
        affine_map<(b, h, s1, s2, d) -> (b, h, s2, d)>,
        affine_map<(b, h, s1, s2, d) -> (b, h, s1, s2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
    } ins(%query, %key : tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>) 
      outs(%scores_init : tensor<1x8x128x128xf32>) {
    ^bb0(%q: f32, %k: f32, %acc: f32):
      %prod = arith.mulf %q, %k : f32
      %sum = arith.addf %acc, %prod : f32
      linalg.yield %sum : f32
    }
    
    // Apply attention weights to values  
    %attention_output = linalg.generic {
    // linalg.generic {
      indexing_maps = [
        affine_map<(b, h, s1, s2, d) -> (b, h, s1, s2)>,
        affine_map<(b, h, s1, s2, d) -> (b, h, s2, d)>,
        affine_map<(b, h, s1, s2, d) -> (b, h, s1, d)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
    } ins(%attention_scores, %value : tensor<1x8x128x128xf32>, tensor<1x8x128x64xf32>) 
      outs(%output_init : tensor<1x8x128x64xf32>) {
    ^bb0(%weight: f32, %v: f32, %acc: f32):
      %weighted = arith.mulf %weight, %v : f32
      %sum = arith.addf %acc, %weighted : f32
      linalg.yield %sum : f32
    }
    
    return %attention_output : tensor<1x8x128x64xf32>
  }
}