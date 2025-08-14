#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map9 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map11 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, 0)>
module attributes {torch.debug_module_name = "SelfAttention"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1x16x64xf32>) -> tensor<1x16x64xf32> {
    %cst = arith.constant dense<[0.0643410683, -0.0784830302, 0.0300444961, -0.100014627, 0.0542372167, -0.112603575, 0.0649143904, -0.00424352288, 0.0275964737, -0.123659715, 0.0238945186, 0.0872097909, 0.0417913347, -0.0992809534, 0.0239272863, -0.0619421601, -0.0898991525, 0.117976949, 0.0412941277, 0.032616958, 1.497300e-02, 0.0757221431, -0.0286310613, 0.0943715274, 0.0392220169, 0.0572496355, 0.0999998599, -0.120468765, -0.0923999845, -0.087250173, -0.0972510725, 0.0798690766, 0.0461412817, -0.120583907, 0.0696480572, -0.0012768358, 0.0200815648, -0.00988648831, -0.0101515949, -0.0134925842, 0.016267851, -0.0443561971, -9.26017761E-4, -0.112554058, -0.0614943504, 0.0090611577, -0.0385854542, -0.114865005, -0.0852195174, -0.0580590814, 0.0980237424, -0.0287268609, -0.105674729, -0.00412739813, 0.0219341218, 0.0452054143, 0.0123965889, 0.117965624, -0.113564566, 0.00855109095, -0.0643291771, -0.0679123253, 0.0823878645, -0.114395827]> : tensor<64xf32>
    %cst_0 = arith.constant dense<"0x0001213131"> : tensor<64x64xf32>
    %cst_1 = arith.constant dense<[-0.0597988516, 0.09627074, 0.108430892, 0.0550045669, 0.0201129019, 0.101091653, -0.0823386163, -0.019345656, 0.00290776789, 0.0902089626, 0.0172834098, -0.122111529, -0.0422461927, -0.108984634, 0.0560320169, -0.0202036351, -0.0994065999, -0.00488929451, -0.0265434831, 0.0710891634, 0.0833828151, -0.102446303, 0.117722735, 0.0545018911, 0.0778864175, -0.0950038582, 0.121468887, 0.0699308366, 0.113065958, 0.111937523, -0.0588523895, 0.0996241569, 4.792750e-02, 0.0225001425, -0.0110603869, 0.0845735818, 0.107234657, -0.0964786857, -0.0775447785, 2.479370e-02, -0.0944011956, -0.040302515, -0.0275542885, -0.0330264419, -0.0882148444, -0.0467430651, 0.0800444185, 0.0419497192, 0.0497268587, -0.119412869, 0.0173888952, 7.641800e-02, -0.0243705213, 0.0384174734, 0.0856086909, 0.015830487, -0.10319148, -0.022280097, 0.107231244, 0.00780861079, 0.087155506, -0.0583211184, 0.0121517926, 0.113550022]> : tensor<64xf32>
    %cst_2 = arith.constant dense<"0x0001213131"> : tensor<64x64xf32>
    %cst_3 = arith.constant dense<[-0.011137113, 0.0111028105, 0.0723482221, -0.0816936046, 0.109250352, -0.111281827, 0.113956168, 0.0163055807, -0.108009681, 0.108792543, 0.0258730501, -0.0907550454, -0.0961481184, -7.081400e-02, -0.0936160833, 0.0726361871, -0.00128486753, 0.103041396, 0.037569344, -0.0361299068, -0.0788837671, -0.0612611622, 0.0283806622, -0.0683858246, 0.123593882, 0.0344175696, -6.505950e-02, 0.0427335054, 0.0473894179, 0.0805011243, -0.0020943433, 0.0463950336, -0.0804267525, 0.0194351673, 0.0864352583, -0.0472663045, 0.0992835611, -0.0638499707, 0.124598533, 0.0130473822, 0.0932537764, -0.0558549166, -0.0206701458, 0.0975215435, 0.111376673, -0.0363733321, -0.0887990147, 8.200960e-02, 0.0373901725, 0.118740261, 0.0936678051, 0.0237957984, 0.0488395542, 0.0999993532, 0.0898319184, -0.0989564508, 0.0152456015, -0.0344953835, 0.00453323126, 0.0778875052, -0.00154860318, 0.0484441817, -0.0571702123, 0.0476947576]> : tensor<64xf32>
    %cst_4 = arith.constant dense<"0x0001213131"> : tensor<64x64xf32>
    %cst_5 = arith.constant dense<[0.0854706168, -0.0383987129, -0.0988222956, 0.0727785826, 0.0460738093, -0.0380327255, -0.112702727, -0.122184947, -0.0294523239, 0.0928061455, -0.0813284516, 0.0318778157, 0.0559287816, -0.0202974379, 0.0983333289, 0.119929954, -0.0701448321, -0.0922226905, 0.0013795048, -0.0111889094, -0.0272324085, -0.0794680268, -0.0256328881, -0.0316309929, 0.0719788372, -0.0467860401, -0.0108575076, -0.00109305978, -5.079840e-02, -0.11722815, 0.084235087, 0.0849267244, 0.081811741, -0.0952921659, 0.0472761691, 0.0293507129, 0.0531315953, -0.0740950405, -0.0314445347, 0.0453533977, -0.0380002856, 0.0014564842, 0.0424681306, -0.00507420301, -0.00829535723, 0.0406988561, -0.0506670922, -0.112537771, -0.107068628, -0.0783562064, 0.048258543, -0.0740308911, -0.0737576932, 0.0261428505, 0.113005742, -0.110044226, -0.0436147302, -0.104245305, -0.0642879754, 0.00906430184, -0.103244737, 0.0595563352, -0.0580220819, 0.00220760703]> : tensor<64xf32>
    %cst_6 = arith.constant dense<"0x0001213131"> : tensor<64x64xf32>
    %c0_i64 = arith.constant 0 : i64
    %cst_7 = arith.constant 0.000000e+00 : f32
    %cst_8 = arith.constant 0xFF800000 : f32
    %cst_9 = arith.constant 2.8284271247461903 : f64
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_6 : tensor<64x64xf32>) outs(%0 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x64xf32>
    %2 = tensor.empty() : tensor<1x16x64xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x16x64xf32>) outs(%2 : tensor<1x16x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x16x64xf32>
    %4 = tensor.empty() : tensor<1x64x64xf32>
    %5 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<64x64xf32>) outs(%4 : tensor<1x64x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x64x64xf32>
    %6 = linalg.fill ins(%cst_7 : f32) outs(%2 : tensor<1x16x64xf32>) -> tensor<1x16x64xf32>
    %7 = linalg.batch_matmul ins(%3, %5 : tensor<1x16x64xf32>, tensor<1x64x64xf32>) outs(%6 : tensor<1x16x64xf32>) -> tensor<1x16x64xf32>
    %8 = linalg.generic {indexing_maps = [#map2, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7, %cst_5 : tensor<1x16x64xf32>, tensor<64xf32>) outs(%2 : tensor<1x16x64xf32>) {
    ^bb0(%in: f32, %in_18: f32, %out: f32):
      %52 = arith.addf %in, %in_18 : f32
      linalg.yield %52 : f32
    } -> tensor<1x16x64xf32>
    %expanded = tensor.expand_shape %8 [[0], [1], [2, 3]] : tensor<1x16x64xf32> into tensor<1x16x8x8xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_4 : tensor<64x64xf32>) outs(%0 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x64xf32>
    %10 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9 : tensor<64x64xf32>) outs(%4 : tensor<1x64x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x64x64xf32>
    %11 = linalg.batch_matmul ins(%3, %10 : tensor<1x16x64xf32>, tensor<1x64x64xf32>) outs(%6 : tensor<1x16x64xf32>) -> tensor<1x16x64xf32>
    %12 = linalg.generic {indexing_maps = [#map2, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11, %cst_3 : tensor<1x16x64xf32>, tensor<64xf32>) outs(%2 : tensor<1x16x64xf32>) {
    ^bb0(%in: f32, %in_18: f32, %out: f32):
      %52 = arith.addf %in, %in_18 : f32
      linalg.yield %52 : f32
    } -> tensor<1x16x64xf32>
    %expanded_10 = tensor.expand_shape %12 [[0], [1], [2, 3]] : tensor<1x16x64xf32> into tensor<1x16x8x8xf32>
    %13 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_2 : tensor<64x64xf32>) outs(%0 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x64xf32>
    %14 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%13 : tensor<64x64xf32>) outs(%4 : tensor<1x64x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x64x64xf32>
    %15 = linalg.batch_matmul ins(%3, %14 : tensor<1x16x64xf32>, tensor<1x64x64xf32>) outs(%6 : tensor<1x16x64xf32>) -> tensor<1x16x64xf32>
    %16 = linalg.generic {indexing_maps = [#map2, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %cst_1 : tensor<1x16x64xf32>, tensor<64xf32>) outs(%2 : tensor<1x16x64xf32>) {
    ^bb0(%in: f32, %in_18: f32, %out: f32):
      %52 = arith.addf %in, %in_18 : f32
      linalg.yield %52 : f32
    } -> tensor<1x16x64xf32>
    %expanded_11 = tensor.expand_shape %16 [[0], [1], [2, 3]] : tensor<1x16x64xf32> into tensor<1x16x8x8xf32>
    %17 = tensor.empty() : tensor<1x8x16x8xf32>
    %18 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x16x8x8xf32>) outs(%17 : tensor<1x8x16x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x8x16x8xf32>
    %19 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_10 : tensor<1x16x8x8xf32>) outs(%17 : tensor<1x8x16x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x8x16x8xf32>
    %20 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_11 : tensor<1x16x8x8xf32>) outs(%17 : tensor<1x8x16x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x8x16x8xf32>
    %21 = tensor.empty() : tensor<1x8x8x16xf32>
    %22 = linalg.generic {indexing_maps = [#map6, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%19 : tensor<1x8x16x8xf32>) outs(%21 : tensor<1x8x8x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x8x8x16xf32>
    %23 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%18 : tensor<1x8x16x8xf32>) outs(%17 : tensor<1x8x16x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x8x16x8xf32>
    %24 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22 : tensor<1x8x8x16xf32>) outs(%21 : tensor<1x8x8x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x8x8x16xf32>
    %collapsed = tensor.collapse_shape %23 [[0, 1], [2], [3]] : tensor<1x8x16x8xf32> into tensor<8x16x8xf32>
    %collapsed_12 = tensor.collapse_shape %24 [[0, 1], [2], [3]] : tensor<1x8x8x16xf32> into tensor<8x8x16xf32>
    %25 = tensor.empty() : tensor<8x16x16xf32>
    %26 = linalg.fill ins(%cst_7 : f32) outs(%25 : tensor<8x16x16xf32>) -> tensor<8x16x16xf32>
    %27 = linalg.batch_matmul ins(%collapsed, %collapsed_12 : tensor<8x16x8xf32>, tensor<8x8x16xf32>) outs(%26 : tensor<8x16x16xf32>) -> tensor<8x16x16xf32>
    %expanded_13 = tensor.expand_shape %27 [[0, 1], [2], [3]] : tensor<8x16x16xf32> into tensor<1x8x16x16xf32>
    %28 = tensor.empty() : tensor<1x8x16x16xf32>
    %29 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_13 : tensor<1x8x16x16xf32>) outs(%28 : tensor<1x8x16x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %52 = arith.truncf %cst_9 : f64 to f32
      %53 = arith.divf %in, %52 : f32
      linalg.yield %53 : f32
    } -> tensor<1x8x16x16xf32>
    %30 = tensor.empty() : tensor<1x8x16x1xi64>
    %31 = linalg.fill ins(%c0_i64 : i64) outs(%30 : tensor<1x8x16x1xi64>) -> tensor<1x8x16x1xi64>
    %32 = tensor.empty() : tensor<1x8x16x1xf32>
    %33 = linalg.fill ins(%cst_8 : f32) outs(%32 : tensor<1x8x16x1xf32>) -> tensor<1x8x16x1xf32>
    %34:2 = linalg.generic {indexing_maps = [#map6, #map10, #map10], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%29 : tensor<1x8x16x16xf32>) outs(%33, %31 : tensor<1x8x16x1xf32>, tensor<1x8x16x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_18: i64):
      %52 = linalg.index 3 : index
      %53 = arith.index_cast %52 : index to i64
      %54 = arith.maximumf %in, %out : f32
      %55 = arith.cmpf ogt, %in, %out : f32
      %56 = arith.select %55, %53, %out_18 : i64
      linalg.yield %54, %56 : f32, i64
    } -> (tensor<1x8x16x1xf32>, tensor<1x8x16x1xi64>)
    %35 = linalg.generic {indexing_maps = [#map9, #map11, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%29, %34#0 : tensor<1x8x16x16xf32>, tensor<1x8x16x1xf32>) outs(%28 : tensor<1x8x16x16xf32>) {
    ^bb0(%in: f32, %in_18: f32, %out: f32):
      %52 = arith.subf %in, %in_18 : f32
      linalg.yield %52 : f32
    } -> tensor<1x8x16x16xf32>
    %36 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%35 : tensor<1x8x16x16xf32>) outs(%28 : tensor<1x8x16x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %52 = math.exp %in : f32
      linalg.yield %52 : f32
    } -> tensor<1x8x16x16xf32>
    %37 = linalg.fill ins(%cst_7 : f32) outs(%32 : tensor<1x8x16x1xf32>) -> tensor<1x8x16x1xf32>
    %38 = linalg.generic {indexing_maps = [#map6, #map10], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%36 : tensor<1x8x16x16xf32>) outs(%37 : tensor<1x8x16x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %52 = arith.addf %in, %out : f32
      linalg.yield %52 : f32
    } -> tensor<1x8x16x1xf32>
    %39 = linalg.generic {indexing_maps = [#map9, #map11, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%36, %38 : tensor<1x8x16x16xf32>, tensor<1x8x16x1xf32>) outs(%28 : tensor<1x8x16x16xf32>) {
    ^bb0(%in: f32, %in_18: f32, %out: f32):
      %52 = arith.divf %in, %in_18 : f32
      linalg.yield %52 : f32
    } -> tensor<1x8x16x16xf32>
    %40 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%39 : tensor<1x8x16x16xf32>) outs(%28 : tensor<1x8x16x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x8x16x16xf32>
    %41 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%20 : tensor<1x8x16x8xf32>) outs(%17 : tensor<1x8x16x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x8x16x8xf32>
    %collapsed_14 = tensor.collapse_shape %40 [[0, 1], [2], [3]] : tensor<1x8x16x16xf32> into tensor<8x16x16xf32>
    %collapsed_15 = tensor.collapse_shape %41 [[0, 1], [2], [3]] : tensor<1x8x16x8xf32> into tensor<8x16x8xf32>
    %42 = tensor.empty() : tensor<8x16x8xf32>
    %43 = linalg.fill ins(%cst_7 : f32) outs(%42 : tensor<8x16x8xf32>) -> tensor<8x16x8xf32>
    %44 = linalg.batch_matmul ins(%collapsed_14, %collapsed_15 : tensor<8x16x16xf32>, tensor<8x16x8xf32>) outs(%43 : tensor<8x16x8xf32>) -> tensor<8x16x8xf32>
    %expanded_16 = tensor.expand_shape %44 [[0, 1], [2], [3]] : tensor<8x16x8xf32> into tensor<1x8x16x8xf32>
    %45 = tensor.empty() : tensor<1x16x8x8xf32>
    %46 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_16 : tensor<1x8x16x8xf32>) outs(%45 : tensor<1x16x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x16x8x8xf32>
    %collapsed_17 = tensor.collapse_shape %46 [[0], [1], [2, 3]] : tensor<1x16x8x8xf32> into tensor<1x16x64xf32>
    %47 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<64x64xf32>) outs(%0 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x64xf32>
    %48 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_17 : tensor<1x16x64xf32>) outs(%2 : tensor<1x16x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x16x64xf32>
    %49 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47 : tensor<64x64xf32>) outs(%4 : tensor<1x64x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x64x64xf32>
    %50 = linalg.batch_matmul ins(%48, %49 : tensor<1x16x64xf32>, tensor<1x64x64xf32>) outs(%6 : tensor<1x16x64xf32>) -> tensor<1x16x64xf32>
    %51 = linalg.generic {indexing_maps = [#map2, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%50, %cst : tensor<1x16x64xf32>, tensor<64xf32>) outs(%2 : tensor<1x16x64xf32>) {
    ^bb0(%in: f32, %in_18: f32, %out: f32):
      %52 = arith.addf %in, %in_18 : f32
      linalg.yield %52 : f32
    } -> tensor<1x16x64xf32>
    return %51 : tensor<1x16x64xf32>
  }
}

