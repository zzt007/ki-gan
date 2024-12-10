import argparse
import os
import random
import torch
import matplotlib.lines as mlines
import sys
from attrdict import AttrDict
sys.path.append("/root/trajectory_prediction/KI_GAN")
from kigan.data.loader import data_loader
from kigan.models import TrajectoryGenerator
from kigan.utils import relative_to_abs, get_dset_path

import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator

# conversion_factor = 111000 单位是m，因为经度差1°约为111千米，纬度差1°约为111千米。
def convert_to_latlon(traj, conversion_factor=111000):
    lat = traj[:, :, 0] / conversion_factor
    lon = traj[:, :, 1] / conversion_factor
    return np.stack([lat, lon], axis=2)

def plot_trajectories_on_map(real_traj, pred_traj, seq_start_end, plot_number):
    G = ox.graph_from_xml("/root/trajectory_prediction/KI_GAN/scripts/map.osm")
    bgcolor = 'white'
    edge_color = 'black'
    fig, ax = ox.plot_graph(G, bgcolor=bgcolor, edge_color=edge_color, show=False, close=False)

    # 定义颜色映射
    '''
    Reds 并不是特指某个具体颜色，而是一个颜色映射（colormap）的名称。
    在 Matplotlib 中，Reds 是一个预定义的颜色映射，它从白色渐变到红色。
    这个颜色映射通常用于热图或其他需要颜色渐变的可视化场景。
    当 point 值为 0 时，颜色接近白色。
    当 point 值为最大值时，颜色为深红色。
    '''
    # cmap = plt.get_cmap('Reds')
    # cmap2 = plt.get_cmap('Greens')
    
    color_real = 'red'
    color_pred = 'blue'

    for (start, end) in seq_start_end:
        end = min(end, real_traj.shape[0])
        for i in range(start, end):
            num_points = real_traj.shape[0]

            plt.plot(real_traj[:num_points, i, 1], real_traj[:num_points, i, 0], color=color_real, linestyle='--')

            # 计算真值轨迹的方向箭头
            last_point = real_traj[num_points - 2, i]
            next_point = real_traj[num_points - 1, i]
            dx = next_point[1] - last_point[1]
            dy = next_point[0] - last_point[0]
            plt.quiver(last_point[1],last_point[0],
                       dx,dy,angles='xy',scale_units='xy',
                       scale=1,color=color_real,
                       width=0.01,headwidth=5,headlength=20,
                       pivot='tail',headaxislength=5)
                        
                        
            num_points_pred = pred_traj[13:31].shape[0] # 此处修改是12-12 还是 12-18，当为输出18时，则pred_traj[13:31]；当为12时，则pred_traj[13:25]
            plt.plot(pred_traj[13:31, i, 1], pred_traj[13:31, i, 0], color=color_pred, linestyle='--')

            # 计算预测轨迹的方向箭头
            last_point_pred = pred_traj[13:31][num_points_pred - 2, i]
            next_point_pred = pred_traj[13:31][num_points_pred - 1, i]
            dx_pred = next_point_pred[1] - last_point_pred[1]
            dy_pred = next_point_pred[0] - last_point_pred[0]
            plt.quiver(last_point_pred[1],last_point_pred[0],
                       dx_pred,dy_pred,
                       angles='xy',scale_units='xy',
                       scale=1,color=color_pred,
                       width=0.01,headwidth=5,headlength=20,
                       pivot='tail',headaxislength=5)
            
    # 添加图例
    red_line = mlines.Line2D([], [], color='red', linestyle='--', label='Ground Truth')
    blue_line = mlines.Line2D([], [], color='blue', linestyle='--', label='Predicted')

    ax.legend(handles=[red_line, blue_line])


    if not os.path.exists('Results1212'):
        os.makedirs('Results1212')

    plt.savefig(f'Results1212/trajectory_{plot_number}.png',dpi=300)
    plt.close(fig)

def detect(args, loader, generator, num_samples):

    num_batches = len(loader)
    if num_samples > num_batches:
        num_samples = num_batches
    selected_batches = np.random.choice(num_batches, num_samples, replace=False)

    counter = 0
    plot_number = 1
    for batch in loader:
        if counter in selected_batches:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
             loss_mask, seq_start_end, vx, vy, ax, ay, agent_type, size, traffic_state) = batch

            pred_traj_fake_rel = generator(
                obs_traj, obs_traj_rel, seq_start_end, vx, vy, ax, ay, agent_type, size, traffic_state
            )
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            real_traj = torch.cat([obs_traj, pred_traj_gt], dim=0).detach().cpu().numpy()

            pred_traj = torch.cat([obs_traj, pred_traj_fake], dim=0).detach().cpu().numpy()

            real_traj_latlon = convert_to_latlon(real_traj)
            pred_traj_latlon = convert_to_latlon(pred_traj)

            plot_trajectories_on_map(real_traj_latlon, pred_traj_latlon, seq_start_end, plot_number)
            plot_number += 1

        counter += 1
        if counter >= num_batches:
            break

def main(args):
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    checkpoint = torch.load(args.model_path)
    generator = get_generator(checkpoint)
    _args = AttrDict(checkpoint['args'])
    path = get_dset_path(_args.dataset_name, 'train')
    _, loader = data_loader(_args, path)
    detect(_args, loader, generator, args.num_samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoint_with_model_.pt')
    parser.add_argument('--num_samples', default=10, type=int)
    args = parser.parse_args()
    main(args)
