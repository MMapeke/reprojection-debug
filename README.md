# reprojection-debug
This is temporary repo to debug reprojection.

eval_depth.py is hardcoded to reproject /lego/test/r_0_depth_0001.png into /lego/test/r_1_depth_0001.png and saves the results in test_results directory.

Possible source of error: NERF only lists the FOV in x dimension, which I assume
is the same in y dimension. The logic of the actual reprojection looks correct to me though
and I modified it to see if a slightly different approach produced the same result.
