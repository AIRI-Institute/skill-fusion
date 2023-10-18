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
    obs_maps = np.load(os.path.join(input_dir, 'maps.npz'))['arr_0']
    agent_positions = np.loadtxt(os.path.join(input_dir, 'poses.txt'))
    goal_coords = np.loadtxt(os.path.join(input_dir, 'goal_coords.txt'))

    frames = []
    for i in tqdm(range(len(obs_maps))):
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(obs_maps[i])
        plt.plot(agent_positions[:i + 1, 1], agent_positions[:i + 1, 0], color='orange', lw=2)
        plt.scatter(goal_coords[i:i + 1, 1], goal_coords[i:i + 1, 0], color='r')
        plot_img_np = get_img_from_fig(fig)
        frames.append(plot_img_np)

    frame = ImageSequenceClip(frames, fps=5)
    split = input_dir.split('/')
    if len(split[-1]) == 0:
        output_filename = split[-2] + '.gif'
    else:
        output_filename = split[-1] + '.gif'
    frame.write_gif(os.path.join(output_dir, output_filename))
    #frame.write_videofile(os.path.join(output_dir, output_filename)) 
    
if __name__ == '__main__':
    assert(len(sys.argv) >= 3)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    make_gif(input_dir, output_dir)