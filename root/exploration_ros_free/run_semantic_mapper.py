import numpy as np
from semantic_mapping_from_dataset import SemanticMapper
import subprocess
import os
from skimage.io import imsave

def main():
    mapper = SemanticMapper()
    dataset_dir = '/root/exploration_ros_free/observations_dataset'
    path_to_save = './test_sem_map2/results_decay0.9_thr2_1step_delayed'
    save_gifs = False
    path_to_save_gifs = './sem_map_test_results'
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    if not os.path.exists(path_to_save_gifs):
        os.mkdir(path_to_save_gifs)
    closenesses = []
    ious = []
    episode_names = [x for x in os.listdir(dataset_dir) if x.startswith('episode')]
    episode_names.sort(key=lambda s: int(s.split('_')[1]))
    for ep_name in episode_names:
        if not ep_name.startswith('episode'):
            continue
        data_path = os.path.join(dataset_dir, ep_name)
        mapper.reset()
        trajectory = np.loadtxt(os.path.join(dataset_dir, 'poses', ep_name + '.txt'))
        tilt_angles = np.loadtxt(os.path.join(dataset_dir, 'view_angles', ep_name + '.txt'))
        mapper.run(data_path, trajectory, tilt_angles)
        closeness_sigmoid, iou = mapper.calculate_metrics()
        print('Closeness:', closeness_sigmoid)
        print('IoU:', iou)
        closenesses.append(closeness_sigmoid)
        ious.append(iou)
        save_dir = os.path.join(path_to_save, ep_name)
        #save_dir = './test_exp'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        np.savetxt(os.path.join(save_dir, 'results.txt'), np.array([closeness_sigmoid, iou]))
        obs_map = mapper.obs_maps[-1]
        sem_map = mapper.sem_maps[-1]
        obs_map_gt = mapper.gt_obs_maps[-1]
        sem_map_gt = mapper.gt_sem_maps[-1]
        map_rgb = np.zeros((obs_map.shape[0], obs_map.shape[1], 3))
        map_rgb[:, :, 0] = 1 - obs_map[::-1, :]
        map_rgb[:, :, 1] = 1 - obs_map[::-1, :]
        map_rgb[:, :, 2] = 1 - obs_map[::-1, :]
        map_rgb[sem_map > 0] = [255, 0, 0]
        imsave(os.path.join(save_dir, 'predicted_map.png'), map_rgb)
        map_rgb = np.zeros((obs_map.shape[0], obs_map.shape[1], 3))
        map_rgb[:, :, 0] = 1 - obs_map_gt[::-1, :]
        map_rgb[:, :, 1] = 1 - obs_map_gt[::-1, :]
        map_rgb[:, :, 2] = 1 - obs_map_gt[::-1, :]
        map_rgb[sem_map_gt > 0] = [255, 0, 0]
        imsave(os.path.join(save_dir, 'gt_map.png'), map_rgb)
        if save_gifs:
            np.savez(os.path.join(save_dir, 'obs_maps.npz'), mapper.obs_maps)
            np.savez(os.path.join(save_dir, 'semantic_maps.npz'), mapper.sem_maps)
            np.savez(os.path.join(save_dir, 'obs_maps_gt.npz'), mapper.gt_obs_maps)
            np.savez(os.path.join(save_dir, 'semantic_maps_gt.npz'), mapper.gt_sem_maps)
            np.savez(os.path.join(save_dir, 'semantic_masks.npz'), mapper.sem_masks)
            np.savez(os.path.join(save_dir, 'semantic_masks_gt.npz'), mapper.gt_sem_masks)
            np.savetxt(os.path.join(save_dir, 'agent_positions.txt'), mapper.robot_poses)
            subprocess.run(['python', 'create_gif_sem_map_test.py', save_dir, path_to_save_gifs])
            for file in os.listdir(save_dir):
                if file != 'results.txt' and not file.endswith('.png'):
                    try:
                        os.remove(os.path.join(save_dir, file))
                    except IsADirectoryError:
                        pass
    print('Average closeness and IoU over all:', np.mean(closenesses), np.mean(ious))

if __name__ == '__main__':
    main()