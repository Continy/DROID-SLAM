import sys
sys.path.append('droid_slam')
sys.path.append('thirdparty/tartanair_tools')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob
import time
import yaml
import argparse

from droid import Droid

def image_stream(datapath, image_size=[384, 512], intrinsics_vec=[320.0, 320.0, 320.0, 240.0], stereo=False):
    """ image generator """

    # read all png images in folder
    ht0, wd0 = [480, 640]
    images_left = sorted(glob.glob(os.path.join(datapath, 'image_lcam_front/*.png')))
    images_right = sorted(glob.glob(os.path.join(datapath, 'image_rcam_front/*.png')))

    data = []
    for t in range(len(images_left)):
        images = [ cv2.resize(cv2.imread(images_left[t]), (image_size[1], image_size[0])) ]
        if stereo:
            images += [ cv2.resize(cv2.imread(images_right[t]), (image_size[1], image_size[0])) ]

        images = torch.from_numpy(np.stack(images, 0)).permute(0,3,1,2)
        intrinsics = .8 * torch.as_tensor(intrinsics_vec)

        data.append((t, images, intrinsics))

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default="datasets/TartanAir")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=1000)
    parser.add_argument("--image_size", default=[512,512])
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--plot_curve", action="store_true")
    parser.add_argument("--id", type=int, default=-1)

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--keyframe_thresh", type=float, default=3.5)
    parser.add_argument("--frontend_thresh", type=float, default=15)
    parser.add_argument("--frontend_window", type=int, default=20)
    parser.add_argument("--frontend_radius", type=int, default=1)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--backend_thresh", type=float, default=20.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)

    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')
    args.upsample = False
    from data_readers.tartan_v2 import test_split
    from evaluation.tartanair_evaluator import TartanAirEvaluator

    if not os.path.isdir("figures"):
        os.mkdir("figures")

    if args.id >= 0:
        test_split = [ test_split[args.id] ]

    ate_list = []
    for scene in test_split:
        print("Performing evaluation on {}".format(scene))
        torch.cuda.empty_cache()
        droid = Droid(args)

        scenedir = os.path.join(args.datapath, scene)
        
        for (tstamp, image, intrinsics) in tqdm(image_stream(scenedir, 
                                                             image_size=args.image_size,
                                                             stereo=args.stereo,
                                                             intrinsics_vec=[320.0, 320.0, 320.0, 320.0])):
            droid.track(tstamp, image, intrinsics=intrinsics)

        # fill in non-keyframe poses + global BA
        traj_est = droid.terminate(image_stream(scenedir, image_size=args.image_size,
                                                intrinsics_vec=[320.0, 320.0, 320.0, 320.0]))

        ### do evaluation ###
        evaluator = TartanAirEvaluator()
        gt_file = os.path.join(scenedir, "pose_lcam_front.txt")
        traj_ref = np.loadtxt(gt_file, delimiter=' ')[:, [1, 2, 0, 4, 5, 3, 6]] # ned -> xyz

        # usually stereo should not be scale corrected, but we are comparing monocular and stereo here
        results = evaluator.evaluate_one_trajectory(
            traj_ref, traj_est, scale=True, title=scenedir[-20:].replace('/', '_'))
        
        # save traj_est
        traj_est_path = scene.replace('/', '_') + "_traj_est.txt"
        traj_est_ned = traj_est[:, [2, 0, 1, 5, 3, 4, 6]]
        np.savetxt(traj_est_path, traj_est_ned, delimiter=' ')
        
        print("Saved estimated trajectory to {}".format(traj_est_path))
        
        print(results)
        ate_list.append(results["ate_score"])

    print("Results")
    print(ate_list)

    if args.plot_curve:
        import matplotlib.pyplot as plt
        ate = np.array(ate_list)
        xs = np.linspace(0.0, 1.0, 512)
        ys = [np.count_nonzero(ate < t) / ate.shape[0] for t in xs]

        plt.plot(xs, ys)
        plt.xlabel("ATE [m]")
        plt.ylabel("% runs")
        plt.savefig("tartan_v2_ate_curve.png")

