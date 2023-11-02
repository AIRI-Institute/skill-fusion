import argparse
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

matplotlib.use("Agg")


def parse_args() -> Path:
    """Parse input CLI arguments.

    Raises:
        ValueError: If the given '--data_root' does not exist.

    Returns:
        Path: Data root directory path.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        required=True,
        type=Path,
        help="The data root directory.",
    )

    args = parser.parse_args()

    data_root: Path = args.data_root
    if not data_root.exists():
        raise ValueError("Given data_root directory does not exist.")

    return data_root


def center_crop(img, dim):
    """Returns center cropped image
    Args:
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped
    """
    width, height = img.shape[1], img.shape[0]

    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y - ch2 : mid_y + ch2, mid_x - cw2 : mid_x + cw2]
    return crop_img


def draw_pointcloud(pc: np.ndarray, size: Tuple[int, int] = (540, 540)) -> np.ndarray:
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    # Set limits for x, y, and z axes
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])
    ax.set_zlim([-10, 30])
    # Calculate distance of each point from origin
    dist = np.sqrt(np.sum(np.square(pc), axis=1))
    # Normalize distance values to [0, 1] range
    norm = plt.Normalize(dist.min(), dist.max())
    colors = plt.cm.jet(norm(dist))
    # Plot point cloud with color based on distance from origin
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=0.1, c=colors)
    # Fix the viewpoint
    ax.view_init(elev=45, azim=-180)
    # Hide axes
    ax.set_axis_off()
    # Add tight layout
    plt.tight_layout()
    # Convert plot to cv2 image
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    img = center_crop(img, (960, 960))
    img = cv2.resize(img, size)
    plt.close(fig)
    return img


def load_pc(filepath: Path) -> np.ndarray:
    pc = np.fromfile(filepath, dtype=np.float32).reshape((-1, 4))[:, :-1]
    pc = pc[~np.isnan(pc).any(axis=1)]
    return pc


def load_img(filepath: Path, size: Tuple[int, int] = (854, 480)) -> np.ndarray:
    img = cv2.imread(str(filepath))
    img = cv2.resize(img, size)
    return img


def draw_track_map(
    current_pose: np.ndarray, all_utms: np.ndarray, size: Tuple[int, int] = (540, 540)
) -> np.ndarray:
    x, y = all_utms[:, 0], all_utms[:, 1]
    cur_x, cur_y = current_pose[0], current_pose[1]
    fig, ax = plt.subplots(dpi=200)
    ax.scatter(y, x, s=0.5, c="blue")
    ax.scatter(cur_y, cur_x, s=10, c="red")
    ax.set_xlabel("x")
    ax.set_xlim(-300, 150)
    ax.set_xticks([])
    ax.set_ylabel("y")
    ax.set_ylim(-300, 200)
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    fig.canvas.draw()
    # convert canvas to image
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    # convert from RGB to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = center_crop(img, (960, 960))
    img = cv2.resize(img, size)
    return img


def make_frame(
    img_front: np.ndarray, img_back: np.ndarray, img_lidar: np.ndarray, img_map: np.ndarray
) -> np.ndarray:
    frame = np.full((1080, 1920, 3), fill_value=255, dtype=np.uint8)
    frame[40:520, 40:894] = img_front
    frame[560:1040, 40:894] = img_back
    frame[0:540, 1200:1740] = img_lidar
    frame[540:1080, 1200:1740] = img_map
    return frame


if __name__ == "__main__":
    data_root = parse_args()

    front_files = sorted([f for f in (data_root / "front_cam").iterdir() if f.suffix == ".png"])
    back_files = sorted([f for f in (data_root / "back_cam").iterdir() if f.suffix == ".png"])
    lidar_files = sorted([f for f in (data_root / "lidar").iterdir() if f.suffix == ".bin"])

    poses_df = pd.read_csv(data_root / "track.csv", index_col=0)
    all_utms = poses_df[["tx", "ty"]].to_numpy()

    video_name = str(data_root / "demo.mp4")
    fps = 3
    frame_size = (1920, 1080)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_name, fourcc, fps, frame_size)

    for front_f, back_f, lidar_f, utm in tqdm(
        zip(front_files, back_files, lidar_files, all_utms), total=len(front_files)
    ):
        front_img = load_img(front_f)
        back_img = load_img(back_f)
        lidar_img = draw_pointcloud(load_pc(lidar_f))
        map_img = draw_track_map(utm, all_utms)
        frame = make_frame(front_img, back_img, lidar_img, map_img)
        out.write(frame)

    out.release()
