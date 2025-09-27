import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *

def send_stereo_pair_to_server(address, result_queue, left_rgb, right_rgb):
    """Child process that sends image data to the inference server."""
    with Client(('localhost', address), authkey=b'secret') as conn:
        message = {"left_rgb": left_rgb, "right_rgb": right_rgb}
        conn.send(message)
        result = conn.recv()
        result_queue.put(result)

class FS_model():
    def __init__(self, args, debug=False):
        ckpt_dir = args.ckpt_dir
        cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
        for k in args.__dict__:
            cfg[k] = args.__dict__[k]
        args = OmegaConf.create(cfg)
        self.model = FoundationStereo(args)
        ckpt = torch.load(ckpt_dir)
        self.model.load_state_dict(ckpt['model'])
        self.model.cuda()
        self.model.eval()

        self.scale = args.scale
        assert self.scale<=1, "scale must be <=1"

        self.debug = debug

    def predict(self, left_rgb, right_rgb):

        img0 = cv2.resize(left_rgb, fx=scale, fy=scale, dsize=None)
        img1 = cv2.resize(right_rgb, fx=scale, fy=scale, dsize=None)
        H,W = img0.shape[:2]
        img0_ori = img0.copy()
        logging.info(f"img0: {img0.shape}")


        img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
        img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        with torch.cuda.amp.autocast(True):
            if not args.hiera:
            disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
            else:
            disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)

        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H,W)
        logging.info(f"Output saved to {args.out_dir}")

        if self.debug:
            vis = vis_disparity(disp)
            vis = np.concatenate([img0_ori, vis], axis=1)
            imageio.imwrite(f'{args.out_dir}/vis_left_depth.png', vis)
            logging.info(f"Output saved to {args.out_dir}")


        if args.get_pc:
            with open(args.intrinsic_file, 'r') as f:
            lines = f.readlines()
            K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
            baseline = float(lines[1])

        K[:2] *= scale
        depth = K[0,0]*baseline/disp
        if self.debug:
            xyz_map = depth2xyzmap(depth, K)
            pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0_ori.reshape(-1,3))
            keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.z_far)
            keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
            pcd = pcd.select_by_index(keep_ids)
            o3d.io.write_point_cloud(f'{args.out_dir}/pcd_foundation_stereo.ply', pcd)
            logging.info(f"PCL saved to {args.out_dir}")

            if args.denoise_cloud:
                logging.info("[Optional step] denoise point cloud...")
                cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
                inlier_cloud = pcd.select_by_index(ind)
                o3d.io.write_point_cloud(f'{args.out_dir}/pcd_foundation_stereo_denoise.ply', inlier_cloud)
                pcd = inlier_cloud

            logging.info("Visualizing point cloud. Press ESC to exit.")
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pcd)
            vis.get_render_option().point_size = 1.0
            vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
            vis.run()
            vis.destroy_window()

        return depth
