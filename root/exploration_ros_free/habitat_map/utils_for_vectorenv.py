import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2
import io

from utils import FrameSkip, FrameStack, draw_top_down_map, plot_colortable, rand_cmap

def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask

def get_img_from_fig(fig, dpi=100):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_colors(one_env):
    obs = one_env.reset(14515)
    semantic_image = one_env.sem_to_model
    semantic_image[semantic_image==255] = 41
    one_env.index_to_title_map[41] = one_env.index_to_title_map[-1]
    new_colors = list(mcolors.CSS4_COLORS.values())
    cmap = LinearSegmentedColormap.from_list('test', new_colors, N=len(new_colors))            
    ccmap = matplotlib.cm.get_cmap('jet')   
    mapping = np.array([ one_env.instance_id_to_label_id[i] for i in range(len(one_env.instance_id_to_label_id)) ]+[41])
    new_cmap = rand_cmap(42, type='bright', first_color_black=True, last_color_black=False, verbose=True)
    colorss = {}
    elements_in_image = np.unique(mapping)
    for i,entry in enumerate(elements_in_image):
        label = one_env.index_to_title_map[entry] + ' (' + str(entry)+ ') '
        norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=42)
        colorss[label] = new_cmap(norm(entry))
        
    return colorss, new_cmap

def draw_image_maps(one_env_dict,one_env_mapp,RADIUS,mask,rgb_image,semantic_image,new_cmap,top_down_map,im, show_image):
    
    ind_elements = [3,4,13,6,7,9,10,11,38,31,28,5]
    kernel = np.ones((3, 3), 'uint8')
    vector = []
    if show_image:
        f = plt.figure(figsize=(20,25))
        ax00 = f.add_subplot(616)
        ax00.imshow(im)
        ax01 = f.add_subplot(5,4,13)
        ax01.imshow(one_env_mapp[40])
        ax01.add_patch(plt.Circle([128,128],5,color='red'))
        ax01.add_patch(plt.Circle((128, 128), RADIUS, color='w', fill=False))
        ax02 = f.add_subplot(5,4,14)
        ax02.imshow(semantic_image, cmap=new_cmap, vmin=-1, vmax=42)
        ax03 = f.add_subplot(5,4,16)
        ax03.imshow(top_down_map)   
        ax04 = f.add_subplot(5,4,15)
        ax04.imshow(rgb_image)  
        
        
    for ii,i in enumerate(ind_elements):
        map_item = cv2.morphologyEx(one_env_mapp[i], cv2.MORPH_OPEN, kernel)
        vector.append(int((map_item*mask>0.1).any()))
        if show_image:
            ax = f.add_subplot(5,4,ii+1)
            ax.set_title(one_env_dict[i])
            ax.imshow(map_item)
            ax.add_patch(plt.Circle((128, 128), RADIUS, color='w', fill=False))
     
    
    if show_image:  
        ax00.set_title(str(vector))
        return vector, f
    else:
        return vector, None
    
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__    
    