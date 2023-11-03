import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
from io import BytesIO
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm

def get_img_from_fig(fig, dpi=180):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def make_gif(input_dir, output_dir):
    #maps = np.load(os.path.join(input_dir, 'maps.npz'))['arr_0'][:, 150:450, 150:450]
    semantic_maps = np.load(os.path.join(input_dir, 'semantic_maps.npz'))['arr_0']
    obs_maps = np.load(os.path.join(input_dir, 'obs_maps.npz'))['arr_0']
    #maps = maps.astype(np.uint8)
    semantic_maps = semantic_maps.astype(np.uint8)
    robot_poses = np.loadtxt(os.path.join(input_dir, 'poses.txt'))
    rgbs = np.load(os.path.join(input_dir, 'rgbs.npz'))['arr_0'].astype(np.uint8)
    depths = np.load(os.path.join(input_dir, 'depths.npz'))['arr_0']
    actions = np.load(os.path.join(input_dir, 'actions.txt.npz'))['arr_0']
    goal_coords = np.loadtxt(os.path.join(input_dir, 'goal_coords.txt'))
    fin = open(os.path.join(input_dir, 'path_to_goal.txt'), 'r')
    paths = []
    for line in fin.readlines():
        path = [float(x) for x in line.split()]
        paths.append(np.array([[path[i], path[i + 1]] for i in range(0, len(path), 2)]))
    
    frames = []
    for i in tqdm(list(range(0, len(obs_maps), 3)) + [len(obs_maps) - 1] * 5):
        #if i % 5 != 0:
        #    continue
        fig = plt.figure(figsize=(10, 4), dpi=80)
        plt.subplot(1, 3, 1)
        plt.imshow(rgbs[i])
        plt.subplot(1, 3, 2)
        plt.title('Skill: {}'.format(actions[i][1]), fontsize=16)
        plt.imshow((depths[i, :, :, 0] - 0.5) / 4.5, cmap='gray')
        plt.subplot(1, 3, 3)
        map_rgb = np.zeros((obs_maps[i].shape[0], obs_maps[i].shape[1], 3))
        map_rgb[:, :, 0] = 1 - obs_maps[i, ::-1, :]
        map_rgb[:, :, 1] = 1 - obs_maps[i, ::-1, :]
        map_rgb[:, :, 2] = 1 - obs_maps[i, ::-1, :]
        map_rgb[semantic_maps[i, ::-1, :] > 0] = [255, 0, 0]
        plt.imshow(map_rgb)
        if actions[i][1] == 'PONI' and goal_coords[i][0] > -1000:
            plt.scatter([goal_coords[i][0] / 0.05 + 240], [-goal_coords[i][1] / 0.05 + 240], color='r')
        plt.plot(robot_poses[:i + 1, 0] / 0.05 + 240, robot_poses[:i + 1, 1] / 0.05 + 240, color='orange', alpha=0.5)
        plt.arrow(robot_poses[i, 0] / 0.05 + 240, robot_poses[i, 1] / 0.05 + 240, 
                  np.cos(robot_poses[i, 2]) * 10, -np.sin(robot_poses[i, 2]) * 10,
                  width=2, head_width=5, color='b')
        if len(paths) > i and len(paths[i]) > 0:
            plt.plot(paths[i][:, 0] / 0.05 + 240, -paths[i][:, 1] / 0.05 + 240, color='g', alpha=0.5)
        plot_img_np = get_img_from_fig(fig)
        frames.append(plot_img_np)
        plt.clf()
        
    frame = ImageSequenceClip(frames, fps=2)
    split = input_dir.split('/')
    if len(split[-1]) == 0:
        output_filename = split[-2] + '.gif'
    else:
        output_filename = split[-1] + '.gif'
    frame.write_gif(os.path.join(output_dir, output_filename)) 
    
if __name__ == '__main__':
    assert(len(sys.argv) >= 3)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    make_gif(input_dir, output_dir)