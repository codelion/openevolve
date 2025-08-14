// Baseline self-attention implementation in MLIR
// This is the starting point for optimization

#map_q = affine_map<(b, h, s, d) -> (b, h, s, d)>
#map_k = affine_map<(b, h, s, d) -> (b, h, s, d)>
#map_v = affine_map<(b, h, s, d) -> (b, h, s, d)>
#map_out = affine_map<(b, h, s, d) -> (b, h, s, d)>

func.func @baseline_attention(
    %query: tensor<?x?x?x?xf32>,    // [batch, heads, seq_len, head_dim]
    %key: tensor<?x?x?x?xf32>,     // [batch, heads, seq_len, head_dim]  
    %value: tensor<?x?x?x?xf32>    // [batch, heads, seq_len, head_dim]
) -> tensor<?x?x?x?xf32> {
    
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    
    %batch_size = tensor.dim %query, %c0 : tensor<?x?x?x?xf32>
    %num_heads = tensor.dim %query, %c1 : tensor<?x?x?x?xf32>
    %seq_len = tensor.dim %query, %c2 : tensor<?x?x?x?xf32>
    %head_dim = tensor.dim %query, %c3 : tensor<?x?x?x?xf32>
    
    // Initialize output tensor
    %output_init = tensor.empty(%batch_size, %num_heads, %seq_len, %head_dim) : tensor<?x?x?x?xf32>
    
    // Step 1: Compute attention scores Q @ K^T
    %scores_init = tensor.empty(%batch_size, %num_heads, %seq_len, %seq_len) : tensor<?x?x?x?xf32>
    
    %attention_scores = linalg.generic {
        indexing_maps = [#map_q, #map_k, affine_map<(b, h, s1, s2) -> (b, h, s1, s2)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
    } ins(%query, %key : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) 
      outs(%scores_init : tensor<?x?x?x?xf32>) {
    ^bb0(%q: f32, %k: f32, %acc: f32):
        %prod = arith.mulf %q, %k : f32
        %sum = arith.addf %acc, %prod : f32
        linalg.yield %sum : f32
    }
    
    // Step 2: Apply scaling (1/sqrt(head_dim))
    %scale = arith.constant 0.125 : f32  // 1/sqrt(64) for head_dim=64
    %scaled_scores = linalg.generic {
        indexing_maps = [affine_map<(b, h, s1, s2) -> (b, h, s1, s2)>, 
                        affine_map<(b, h, s1, s2) -> (b, h, s1, s2)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%attention_scores : tensor<?x?x?x?xf32>) 
      outs(%scores_init : tensor<?x?x?x?xf32>) {
    ^bb0(%score: f32, %out: f32):
        %scaled = arith.mulf %score, %scale : f32
        linalg.yield %scaled : f32
    }
    
    // Step 3: Apply softmax
    %softmax_scores = linalg.softmax dimension(3) 
        ins(%scaled_scores : tensor<?x?x?x?xf32>) 
        outs(%scores_init : tensor<?x?x?x?xf32>)
    
    // Step 4: Apply attention weights to values
    %attention_output = linalg.generic {
        indexing_maps = [affine_map<(b, h, s1, s2) -> (b, h, s1, s2)>, 
                        #map_v, #map_out],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
    } ins(%softmax_scores, %value : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) 
      outs(%output_init : tensor<?x?x?x?xf32>) {
    ^bb0(%weight: f32, %v: f32, %acc: f32):
        %weighted = arith.mulf %weight, %v : f32
        %sum = arith.addf %acc, %weighted : f32
        linalg.yield %sum : f32
    }
    
    return %attention_output : tensor<?x?x?x?xf32>
}