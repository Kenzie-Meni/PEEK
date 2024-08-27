import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from pathlib import Path
from scipy.special import entr

def compute_PEEK(feature_maps, h, w):
    # make feature map positive
    positivized_maps = feature_maps + np.abs(np.min(feature_maps))

    # compute entropy maps
    entropy_map = -np.sum(entr(positivized_maps), axis=-1)

    # reshape to size of real image (width x height)
    peek_map = cv2.resize(entropy_map, (w, h))
    
    return peek_map

def compute_global_min_max(modules, frame_paths, feature_folder):
    global_min = float('inf')
    global_max = float('-inf')
    
    for frame_path in frame_paths:
        frame_filename = os.path.split(frame_path)[-1]
        feature_map_path = f'{feature_folder}/{frame_filename[:-4]}.pkl'
        
        with open(feature_map_path, 'rb') as f:
            loaded_feature_maps = pickle.load(f)
        
        for layer in modules:
            feature_maps = loaded_feature_maps[layer][0].cpu().numpy()
            feature_maps = np.moveaxis(feature_maps, 0, -1)
            peek_map = compute_PEEK(feature_maps, h, w)
            
            global_min = min(global_min, np.min(peek_map))
            global_max = max(global_max, np.max(peek_map))
    
    return global_min, global_max

def standardize_peek_map_global(peek_map, global_min, global_max):
    standardized_peek_map = (peek_map - global_min) / (global_max - global_min)
    return standardized_peek_map

def plot_PEEK(modules, frame_paths, feature_folder, save_path=False, run_path=False, verbose=False):
    
    global_min, global_max = compute_global_min_max(modules, frame_paths, feature_folder)
    
    for frame_path in frame_paths:
        frame_filename = os.path.split(frame_path)[-1]
        feature_map_path = f'{feature_folder}/{frame_filename[:-4]}.pkl'
        
        if run_path:
            cols=4  # Add one extra column for the unstandardized PEEK
            fig, axes = plt.subplots(len(modules), cols)
        else:
            cols=3  # Add one extra column for the unstandardized PEEK
            fig, axes = plt.subplots(len(modules), cols)
        
        image = plt.imread(frame_path)
        h, w, _ = image.shape
         
        with open(feature_map_path, 'rb') as f:
            loaded_feature_maps = pickle.load(f)
            
        for i, layer in enumerate(modules):
            axes[i,0].imshow(image)
            
            feature_maps = loaded_feature_maps[layer][0].cpu().numpy()
            feature_maps = np.moveaxis(feature_maps, 0, -1)
            peek_map = compute_PEEK(feature_maps, h, w)
            
            # Plot unstandardized PEEK map
            axes[i,1].imshow(image)
            axes[i,1].imshow(peek_map, alpha=0.7, cmap='jet')
            
            # Standardize the PEEK map using global min and max
            standardized_peek_map = standardize_peek_map_global(peek_map, global_min, global_max)
            
            # Plot standardized PEEK map
            axes[i,2].imshow(image)
            axes[i,2].imshow(standardized_peek_map, alpha=0.7, cmap='jet')
            
            if i==0:
                axes[i,0].set_title('Input Image')
                axes[i,1].set_title('Unstandardized PEEK')
                axes[i,2].set_title('Global Standardized PEEK')
                if run_path: axes[i,3].set_title('Predictions')
               
            axes[i,0].set_ylabel(f'Module {layer}')
            
            if run_path:
                inferred_image = plt.imread(f'{run_path}/{frame_filename}')
                axes[i,3].imshow(inferred_image)
            
        if verbose:
            print(f'Finished with frame {frame_path}.')
            if save_path: print(f'Saving figure to {save_path}/{frame_filename}')
            
        for i in range(len(modules)):                
            for j in range(cols):
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
            
        fig.tight_layout()
        
        if save_path:
            save_fig_path = f'{save_path}/{frame_filename}'
            Path(save_path).mkdir(parents=True, exist_ok=True)
            fig.savefig(save_fig_path)
            fig.clear()



#original peek plottingdsw

# def plot_PEEK(modules, frame_paths, feature_folder, save_path=False, run_path=False, verbose=False):
        
#     # loop over the frames for which we plot PEEK maps
#     for frame_path in frame_paths:
#         # find filename and path to feature maps
#         frame_filename = os.path.split(frame_path)[-1]
#         feature_map_path = f'{feature_folder}/{frame_filename[:-4]}.pkl'
        
#         # make subplots -- 3 columns if we include inferred images
#         if run_path:
#             cols=3
#             fig, axes = plt.subplots(len(modules), cols)

#         else:
#             cols=2
#             fig, axes = plt.subplots(len(modules), cols)
        
#         # read image and get dimensions
#         image = plt.imread(frame_path)
#         h, w, _ = image.shape
         
#         # load all feature maps for the image
#         with open(feature_map_path, 'rb') as f:
#             loaded_feature_maps = pickle.load(f)
            
#         # plot for each module
#         for i, layer in enumerate(modules):
#             # plot the original image
#             axes[i,0].imshow(image)
            
#             # compute PEEK map
#             feature_maps = loaded_feature_maps[layer][0].cpu().numpy()
#             feature_maps = np.moveaxis(feature_maps, 0, -1)
#             peek_map = compute_PEEK(feature_maps, h, w)
            
#             # plot the original image with semi-transparent PEEK map on top
#             axes[i,1].imshow(image)
#             axes[i,1].imshow(peek_map, alpha=0.7, cmap='jet')
            
#             # add column titles
#             if i==0:
#                 axes[i,0].set_title('Input Image')
#                 axes[i,1].set_title('PEEK')
#                 if run_path: axes[i,2].set_title('Predictions')
               
#             # add row title
#             axes[i,0].set_ylabel(f'Module {layer}')
            
#             # plot the image with its bounding boxes
#             if run_path:
#                 inferred_image = plt.imread(f'{run_path}/{frame_filename}')
#                 axes[i,2].imshow(inferred_image)
            
#         # print status updates
#         if verbose:
#             print(f'Finished with frame {frame_path}.')
#             if save_path: print(f'Saving figure to {save_path}/{frame_filename}')
            
#         # Add row labels and hide all axes
#         for i in range(len(modules)):                
#             for j in range(cols):  # Iterate through columns
#                 axes[i, j].set_xticks([])  # Hide the x-axis ticks and labels
#                 axes[i, j].set_yticks([])  # Hide the y-axis ticks and labels
            
#         # tighten the layout
#         fig.tight_layout()
        
#         # save figures
#         if save_path:
#             # create folder if it does not exist
#             save_fig_path = f'{save_path}/{frame_filename}'
#             Path(save_path).mkdir(parents=True, exist_ok=True)
            
#             # save figure
#             fig.savefig(save_fig_path)
#             fig.clear()