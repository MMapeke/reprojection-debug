import torch
import torchvision
from torchvision.utils import save_image

#assume same cam intrinsics src and tgt
#src_poses_c2w puts cam src points into W space
def fwd_depth(tgt_pose_c2w, src_poses_c2w, src_depths, cam):

    #cam intrinsics 
    w, h, fx, fy =int(cam[0]), int(cam[1]), cam[2], cam[3]
    
    y, x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    y = y.cuda()
    x = x.cuda()

    # x from [0,w] to [-w/2, w/2]
    # Principal points are 0.5*w and 0.5*h
    # Reprojects 2d points to world space
    x_src = (x - w/2.0) * src_depths / fx
    y_src = (y - h/2.0) * src_depths / fy
    z_src = src_depths

    # Homogenous representation of our 3d world space points
    p_src = torch.stack((x_src.view((-1, h*w)),
                         y_src.view((-1, h*w)), 
                         z_src.view((-1, h*w)),
                         torch.ones((x_src.shape[0], h*w)).cuda()), 1)

    tgt_pose_w2c = torch.inverse(tgt_pose_c2w)
    K = torch.tensor([fx, 0, w*.5, 0,
                      0, fy, h*.5, 0,
                      0, 0, 1, 0]).reshape([3,4]).cuda()

    T = torch.matmul(tgt_pose_w2c, src_poses_c2w) # Camera 1 -> World -> Camera 2
    p_tgt = torch.matmul(T, p_src)
    p_z = p_tgt[:, 2, :] #distance along ray to each point is the tgt depth

    # Transforming back to 2d coordinates from 3d points
    p_x = torch.round((p_tgt[:, 0, :] / p_z * fx) + w/2.0).long()
    p_y = torch.round((p_tgt[:, 1, :] / p_z * fy) + h/2.0).long()

    # Resolves occlusion and discretize points by depth ranges
    # p_z_ord gives a plane number to every point
    num_planes = 128
    mx, mn = torch.max(p_z), torch.min(p_z)
    p_z_ord = torch.round((p_z - mn) / (mx-mn) *num_planes).int()

    in_bounds = torch.logical_and(torch.logical_and(p_x >= 0, p_x < w),
                                  torch.logical_and(p_y >= 0, p_y < h))

    o = torch.zeros(p_y.shape).cuda()
    for i in range(num_planes + 1, 0, -1):
        idx = torch.logical_and(p_z_ord == i, in_bounds)
        idx = torch.squeeze(idx)
        if not torch.any(idx):
            continue
    
        # Indexing into 1d array based off 2d array 
        #  y * width + x 
        o[0, p_y[0, idx] * w + p_x[0, idx]] = p_z[0, idx]
            
    o = o.reshape((-1, h, w))
     
    return o 

# Added by Marc, testing on NERF synthetic lego dataset
# NOTE: Hardcoded for depth frame0 and depth frame1
# Saves warped image, and blended image (warped image blended with reference)
def depth_warp_lego():
    w = h = 800
    fx = fy = 800 * 0.6911112070083618
    
    cam = [w, h, fx, fy]

    #frame0
    extrinsics_frame1 = torch.tensor(
        [
                [
                    -0.9999999403953552,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    -0.7341099977493286,
                    0.6790305972099304,
                    2.737260103225708
                ],
                [
                    0.0,
                    0.6790306568145752,
                    0.7341098785400391,
                    2.959291696548462
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ]
    , dtype=torch.float32).cuda()
    extrinsics_frame1 = torch.transpose(extrinsics_frame1, 0, 1)

    #frame1
    extrinsics_frame2 = torch.tensor(
        [
                [
                    -0.9980266690254211,
                    0.04609514772891998,
                    -0.042636688798666,
                    -0.17187398672103882
                ],
                [
                    -0.06279051303863525,
                    -0.7326614260673523,
                    0.6776907444000244,
                    2.731858730316162
                ],
                [
                    -3.7252896323280993e-09,
                    0.6790306568145752,
                    0.7341099381446838,
                    2.959291696548462
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        , dtype=torch.float32).cuda()
    extrinsics_frame2 = torch.transpose(extrinsics_frame2, 0, 1)

    src_depth = torchvision.io.read_image("./lego/test/r_0_depth_0001.png", torchvision.io.ImageReadMode.GRAY).cuda()
    tgt_depth = torchvision.io.read_image("./lego/test/r_1_depth_0001.png", torchvision.io.ImageReadMode.GRAY).cuda()
    zero_channel = torch.zeros_like(src_depth)

    output = fwd_depth(extrinsics_frame2, extrinsics_frame1, src_depth, cam)
    save_image(output/255, './test_results/r_0_to_r_1.png')
    
    # Red is warped, Blue is original, Purple is Overlap
    blend = torch.cat((output, zero_channel, tgt_depth), dim = 0)
    save_image(blend/255, './test_results/blended.png')


depth_warp_lego()