#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
###########################################################################
## ROBUST TRIANGULATION  OF 2D COORDINATES                               ##
###########################################################################

This module triangulates 2D json coordinates and builds a .trc file readable 
by OpenSim.

The triangulation is weighted by the likelihood of each detected 2D keypoint 
(if they meet the likelihood threshold). If the reprojection error is above a
threshold, right and left sides are swapped; if it is still above, a camera 
is removed for this point and this frame, until the threshold is met. If more 
cameras are removed than a predefined minimum, triangulation is skipped for 
the point and this frame. In the end, missing values are interpolated.

In case of multiple subjects detection, make sure you first run the 
personAssociation module. It will then associate people across frames by 
measuring the frame-by-frame distance between them.

INPUTS: 
- a calibration file (.toml extension)
- json files for each camera with only one person of interest
- a Config.toml file
- a skeleton model

OUTPUTS: 
- a .trc file with 3D coordinates in Y-up system coordinates
'''


## INIT
import os
import glob
import fnmatch
import re
import numpy as np
import json
import itertools as it
import pandas as pd
import cv2
import toml
from tqdm import tqdm
from collections import Counter
from anytree import RenderTree
from anytree.importer import DictImporter
import logging
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RangeSlider
from scipy.spatial import ConvexHull
from matplotlib.animation import FuncAnimation

from Pose2Sim.common import retrieve_calib_params, computeP, weighted_triangulation, \
    reprojection, euclidean_distance, sort_people_sports2d, interpolate_zeros_nans, \
    sort_stringlist_by_last_number, zup2yup, convert_to_c3d
from Pose2Sim.skeletons import *


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
from importlib.metadata import version
__version__ = version('pose2sim')
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def count_persons_in_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return len(data.get('people', []))
    

def make_trc(config_dict, Q, keypoints_names, f_range, id_person=-1):
    '''
    Make Opensim compatible trc file from a dataframe with 3D coordinates

    INPUT:
    - config_dict: dictionary of configuration parameters
    - Q: pandas dataframe with 3D coordinates as columns, frame number as rows
    - keypoints_names: list of strings
    - f_range: list of two numbers. Range of frames

    OUTPUT:
    - trc file
    '''

    # Read config_dict
    project_dir = config_dict.get('project').get('project_dir')
    multi_person = config_dict.get('project').get('multi_person')
    if multi_person:
        seq_name = f'{os.path.basename(os.path.realpath(project_dir))}_P{id_person+1}'
    else:
        seq_name = f'{os.path.basename(os.path.realpath(project_dir))}'
    pose3d_dir = os.path.join(project_dir, 'pose-3d')

    # Get frame_rate
    video_dir = os.path.join(project_dir, 'videos')
    vid_img_extension = config_dict['pose']['vid_img_extension']
    video_files = glob.glob(os.path.join(video_dir, '*'+vid_img_extension))
    frame_rate = config_dict.get('project').get('frame_rate')
    if frame_rate == 'auto': 
        try:
            cap = cv2.VideoCapture(video_files[0])
            cap.read()
            if cap.read()[0] == False:
                raise
            frame_rate = round(cap.get(cv2.CAP_PROP_FPS))
        except:
            frame_rate = 60

    trc_f = f'{seq_name}_{f_range[0]}-{f_range[1]}.trc'

    #Header
    DataRate = CameraRate = OrigDataRate = frame_rate
    NumFrames = len(Q)
    NumMarkers = len(keypoints_names)
    header_trc = ['PathFileType\t4\t(X/Y/Z)\t' + trc_f, 
            'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames', 
            '\t'.join(map(str,[DataRate, CameraRate, NumFrames, NumMarkers, 'm', OrigDataRate, f_range[0], f_range[1]])),
            'Frame#\tTime\t' + '\t\t\t'.join(keypoints_names) + '\t\t\t',
            '\t\t'+'\t'.join([f'X{i+1}\tY{i+1}\tZ{i+1}' for i in range(len(keypoints_names))]) + '\t']
    
    # Zup to Yup coordinate system
    Q = zup2yup(Q)
    
    #Add Frame# and Time columns
    Q.index = np.array(range(f_range[0], f_range[1]))
    Q.insert(0, 't', Q.index/ frame_rate)
    # Q = Q.fillna(' ')

    #Write file
    if not os.path.exists(pose3d_dir): os.mkdir(pose3d_dir)
    trc_path = os.path.realpath(os.path.join(pose3d_dir, trc_f))
    with open(trc_path, 'w') as trc_o:
        [trc_o.write(line+'\n') for line in header_trc]
        Q.to_csv(trc_o, sep='\t', index=True, header=None, lineterminator='\n')

    return trc_path


def retrieve_right_trc_order(trc_paths):
    '''
    Lets the user input which static file correspond to each generated trc file.
    
    INPUT:
    - trc_paths: list of strings
    
    OUTPUT:
    - trc_id: list of integers
    '''
    
    logging.info('\n\nReordering trc file IDs:')
    logging.info(f'\nPlease visualize the generated trc files in Blender or OpenSim.\nTrc files are stored in {os.path.dirname(trc_paths[0])}.\n')
    retry = True
    while retry:
        retry = False
        logging.info('List of trc files:')
        [logging.info(f'#{t_list}: {os.path.basename(trc_list)}') for t_list, trc_list in enumerate(trc_paths)]
        trc_id = []
        for t, trc_p in enumerate(trc_paths):
            logging.info(f'\nStatic trial #{t} corresponds to trc number:')
            trc_id += [input('Enter ID:')]
        
        # Check non int and duplicates
        try:
            trc_id = [int(t) for t in trc_id]
            duplicates_in_input = (len(trc_id) != len(set(trc_id)))
            if duplicates_in_input:
                retry = True
                print('\n\nWARNING: Same ID entered twice: please check IDs again.\n')
        except:
            print('\n\nWARNING: The ID must be an integer: please check IDs again.\n')
            retry = True
    
    return trc_id


def recap_triangulate(config_dict, error, nb_cams_excluded, keypoints_names, cam_excluded_count, interp_frames, non_interp_frames, trc_path):
    '''
    Print a message giving statistics on reprojection errors (in pixel and in m)
    as well as the number of cameras that had to be excluded to reach threshold 
    conditions. Also stored in User/logs.txt.

    INPUT:
    - a Config.toml file
    - error: dataframe 
    - nb_cams_excluded: dataframe
    - keypoints_names: list of strings

    OUTPUT:
    - Message in console
    '''

    # Read config_dict
    project_dir = config_dict.get('project').get('project_dir')
    # if batch
    session_dir = os.path.realpath(os.path.join(project_dir, '..'))
    # if single trial
    session_dir = session_dir if 'Config.toml' in os.listdir(session_dir) else os.getcwd()
    calib_dir = [os.path.join(session_dir, c) for c in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, c)) and  'calib' in c.lower()][0]
    calib_file = glob.glob(os.path.join(calib_dir, '*.toml'))[0] # lastly created calibration file
    calib = toml.load(calib_file)
    cal_keys = [c for c in calib.keys() 
            if c not in ['metadata', 'capture_volume', 'charuco', 'checkerboard'] 
            and isinstance(calib[c],dict)]
    cam_names = np.array([calib[c].get('name') if calib[c].get('name') else c for c in cal_keys])
    cam_names = cam_names[list(cam_excluded_count[0].keys())]
    error_threshold_triangulation = config_dict.get('triangulation').get('reproj_error_threshold_triangulation')
    likelihood_threshold = config_dict.get('triangulation').get('likelihood_threshold_triangulation')
    show_interp_indices = config_dict.get('triangulation').get('show_interp_indices')
    interpolation_kind = config_dict.get('triangulation').get('interpolation')
    interp_gap_smaller_than = config_dict.get('triangulation').get('interp_if_gap_smaller_than')
    fill_large_gaps_with = config_dict.get('triangulation').get('fill_large_gaps_with')
    make_c3d = config_dict.get('triangulation').get('make_c3d')
    handle_LR_swap = config_dict.get('triangulation').get('handle_LR_swap')
    undistort_points = config_dict.get('triangulation').get('undistort_points')
    
    # Recap
    calib_cam1 = calib[cal_keys[0]]
    fm = calib_cam1['matrix'][0][0]
    Dm = euclidean_distance(calib_cam1['translation'], [0,0,0])

    logging.info('')
    nb_persons_to_detect = len(error)
    for n in range(nb_persons_to_detect):
        if nb_persons_to_detect > 1:
            logging.info(f'\n\nPARTICIPANT {n+1}\n')
        
        for idx, name in enumerate(keypoints_names):
            mean_error_keypoint_px = np.around(error[n].iloc[:,idx].mean(), decimals=1) # RMS à la place?
            mean_error_keypoint_m = np.around(mean_error_keypoint_px * Dm / fm, decimals=3)
            mean_cam_excluded_keypoint = np.around(nb_cams_excluded[n].iloc[:,idx].mean(), decimals=2)
            logging.info(f'Mean reprojection error for {name} is {mean_error_keypoint_px} px (~ {mean_error_keypoint_m} m), reached with {mean_cam_excluded_keypoint} excluded cameras. ')
            if show_interp_indices:
                if interpolation_kind != 'none':
                    if len(list(interp_frames[n][idx])) == 0 and len(list(non_interp_frames[n][idx])) == 0:
                        logging.info(f'  No frames needed to be interpolated.')
                    if len(list(interp_frames[n][idx]))>0: 
                        interp_str = str(interp_frames[n][idx]).replace(":", " to ").replace("'", "").replace("]", "").replace("[", "")
                        logging.info(f'  Frames {interp_str} were interpolated.')
                    if len(list(non_interp_frames[n][idx]))>0:
                        noninterp_str = str(non_interp_frames[n][idx]).replace(":", " to ").replace("'", "").replace("]", "").replace("[", "")
                        logging.info(f'  Frames {noninterp_str} were not interpolated.')
                else:
                    logging.info(f'  No frames were interpolated because \'interpolation_kind\' was set to none. ')
        
        mean_error_px = np.around(error[n]['mean'].mean(), decimals=1)
        mean_error_mm = np.around(mean_error_px * Dm / fm *1000, decimals=1)
        mean_cam_excluded = np.around(nb_cams_excluded[n]['mean'].mean(), decimals=2)

        logging.info(f'\n--> Mean reprojection error for all points on all frames is {mean_error_px} px, which roughly corresponds to {mean_error_mm} mm. ')
        logging.info(f'Cameras were excluded if likelihood was below {likelihood_threshold} and if the reprojection error was above {error_threshold_triangulation} px.') 
        if interpolation_kind != 'none':
            logging.info(f'Gaps were interpolated with {interpolation_kind} method if smaller than {interp_gap_smaller_than} frames. Larger gaps were filled with {["the last valid value" if fill_large_gaps_with == "last_value" else "zeros" if fill_large_gaps_with == "zeros" else "NaNs"][0]}.') 
        logging.info(f'In average, {mean_cam_excluded} cameras had to be excluded to reach these thresholds.')
        
        cam_excluded_count[n] = {i: v for i, v in zip(cam_names, cam_excluded_count[n].values())}
        cam_excluded_count[n] = {k: v for k, v in sorted(cam_excluded_count[n].items(), key=lambda item: item[1])[::-1]}
        str_cam_excluded_count = ''
        for i, (k, v) in enumerate(cam_excluded_count[n].items()):
            if i ==0:
                 str_cam_excluded_count += f'Camera {k} was excluded {int(np.round(v*100))}% of the time, '
            elif i == len(cam_excluded_count[n])-1:
                str_cam_excluded_count += f'and Camera {k}: {int(np.round(v*100))}%.'
            else:
                str_cam_excluded_count += f'Camera {k}: {int(np.round(v*100))}%, '
        logging.info(str_cam_excluded_count)
        logging.info(f'\n3D coordinates are stored at {trc_path[n]}.')
        
    logging.info('\n\n')
    if make_c3d:
        logging.info('All trc files have been converted to c3d.')
    logging.info(f'Limb swapping was {"handled" if handle_LR_swap else "not handled"}.')
    logging.info(f'Lens distortions were {"taken into account" if undistort_points else "not taken into account"}.')


def triangulation_from_best_cameras(config_dict, coords_2D_kpt, coords_2D_kpt_swapped, projection_matrices, calib_params):
    '''
    Triangulates 2D keypoint coordinates. If reprojection error is above threshold,
    tries swapping left and right sides. If still above, removes a camera until error
    is below threshold unless the number of remaining cameras is below a predefined number.

    1. Creates subset with N cameras excluded 
    2. Tries all possible triangulations
    3. Chooses the one with smallest reprojection error
    If error too big, take off one more camera.
        If then below threshold, retain result.
        If better but still too big, take off one more camera.
    
    INPUTS:
    - a Config.toml file
    - coords_2D_kpt: (x,y,likelihood) * ncams array
    - coords_2D_kpt_swapped: (x,y,likelihood) * ncams array  with left/right swap
    - projection_matrices: list of arrays

    OUTPUTS:
    - Q: array of triangulated point (x,y,z,1.)
    - error_min: float
    - nb_cams_excluded: int
    '''
    
    # Read config_dict
    error_threshold_triangulation = config_dict.get('triangulation').get('reproj_error_threshold_triangulation')
    min_cameras_for_triangulation = config_dict.get('triangulation').get('min_cameras_for_triangulation')
    handle_LR_swap = config_dict.get('triangulation').get('handle_LR_swap')

    undistort_points = config_dict.get('triangulation').get('undistort_points')
    if undistort_points:
        calib_params_K = calib_params['K']
        calib_params_dist = calib_params['dist']
        calib_params_R = calib_params['R']
        calib_params_T = calib_params['T']

    # Initialize
    x_files, y_files, likelihood_files = coords_2D_kpt
    x_files_swapped, y_files_swapped, likelihood_files_swapped = coords_2D_kpt_swapped
    n_cams = len(x_files)
    error_min = np.inf 
    
    nb_cams_off = 0 # cameras will be taken-off until reprojection error is under threshold
    # print('\n')
    while error_min > error_threshold_triangulation and n_cams - nb_cams_off >= min_cameras_for_triangulation:
        # print("error min ", error_min, "thresh ", error_threshold_triangulation, 'nb_cams_off ', nb_cams_off)
        # Create subsets with "nb_cams_off" cameras excluded
        id_cams_off = np.array(list(it.combinations(range(n_cams), nb_cams_off)))
        
        if undistort_points:
            calib_params_K_filt = [calib_params_K]*len(id_cams_off)
            calib_params_dist_filt = [calib_params_dist]*len(id_cams_off)
            calib_params_R_filt = [calib_params_R]*len(id_cams_off)
            calib_params_T_filt = [calib_params_T]*len(id_cams_off)
        projection_matrices_filt = [projection_matrices]*len(id_cams_off)

        x_files_filt = np.vstack([x_files.copy()]*len(id_cams_off))
        y_files_filt = np.vstack([y_files.copy()]*len(id_cams_off))
        x_files_swapped_filt = np.vstack([x_files_swapped.copy()]*len(id_cams_off))
        y_files_swapped_filt = np.vstack([y_files_swapped.copy()]*len(id_cams_off))
        likelihood_files_filt = np.vstack([likelihood_files.copy()]*len(id_cams_off))
        
        if nb_cams_off > 0:
            for i in range(len(id_cams_off)):
                x_files_filt[i][id_cams_off[i]] = np.nan
                y_files_filt[i][id_cams_off[i]] = np.nan
                x_files_swapped_filt[i][id_cams_off[i]] = np.nan
                y_files_swapped_filt[i][id_cams_off[i]] = np.nan
                likelihood_files_filt[i][id_cams_off[i]] = np.nan
        
        # Excluded cameras index and count
        id_cams_off_tot_new = [np.argwhere(np.isnan(x)).ravel() for x in likelihood_files_filt]
        nb_cams_excluded_filt = [np.count_nonzero(np.nan_to_num(x)==0) for x in likelihood_files_filt] # count nans and zeros
        nb_cams_off_tot = max(nb_cams_excluded_filt)
        # print('likelihood_files_filt ',likelihood_files_filt)
        # print('nb_cams_excluded_filt ', nb_cams_excluded_filt, 'nb_cams_off_tot ', nb_cams_off_tot)
        if nb_cams_off_tot > n_cams - min_cameras_for_triangulation:
            break
        id_cams_off_tot = id_cams_off_tot_new
        
        # print('still in loop')
        if undistort_points:
            calib_params_K_filt = [ [ c[i] for i in range(n_cams) if not np.isnan(likelihood_files_filt[j][i]) and not likelihood_files_filt[j][i]==0. ] for j, c in enumerate(calib_params_K_filt) ]
            calib_params_dist_filt = [ [ c[i] for i in range(n_cams) if not np.isnan(likelihood_files_filt[j][i]) and not likelihood_files_filt[j][i]==0. ] for j, c in enumerate(calib_params_dist_filt) ]
            calib_params_R_filt = [ [ c[i] for i in range(n_cams) if not np.isnan(likelihood_files_filt[j][i]) and not likelihood_files_filt[j][i]==0. ] for j, c in enumerate(calib_params_R_filt) ]
            calib_params_T_filt = [ [ c[i] for i in range(n_cams) if not np.isnan(likelihood_files_filt[j][i]) and not likelihood_files_filt[j][i]==0. ] for j, c in enumerate(calib_params_T_filt) ]
        projection_matrices_filt = [ [ p[i] for i in range(n_cams) if not np.isnan(likelihood_files_filt[j][i]) and not likelihood_files_filt[j][i]==0. ] for j, p in enumerate(projection_matrices_filt) ]
        
        # print('\nnb_cams_off', repr(nb_cams_off), 'nb_cams_excluded', repr(nb_cams_excluded_filt))
        # print('likelihood_files ', repr(likelihood_files))
        # print('y_files ', repr(y_files))
        # print('x_files ', repr(x_files))
        # print('x_files_swapped ', repr(x_files_swapped))
        # print('likelihood_files_filt ', repr(likelihood_files_filt))
        # print('x_files_filt ', repr(x_files_filt))
        # print('id_cams_off_tot ', id_cams_off_tot)
        
        x_files_filt = [ np.array([ xx for ii, xx in enumerate(x) if not np.isnan(likelihood_files_filt[i][ii]) and not likelihood_files_filt[i][ii]==0. ]) for i,x in enumerate(x_files_filt) ]
        y_files_filt = [ np.array([ xx for ii, xx in enumerate(x) if not np.isnan(likelihood_files_filt[i][ii]) and not likelihood_files_filt[i][ii]==0. ]) for i,x in enumerate(y_files_filt) ]
        x_files_swapped_filt = [ np.array([ xx for ii, xx in enumerate(x) if not np.isnan(likelihood_files_filt[i][ii]) and not likelihood_files_filt[i][ii]==0. ]) for i,x in enumerate(x_files_swapped_filt) ]
        y_files_swapped_filt = [ np.array([ xx for ii, xx in enumerate(x) if not np.isnan(likelihood_files_filt[i][ii]) and not likelihood_files_filt[i][ii]==0. ]) for i,x in enumerate(y_files_swapped_filt) ]
        likelihood_files_filt = [ np.array([ xx for ii, xx in enumerate(x) if not np.isnan(xx) and not xx==0. ]) for x in likelihood_files_filt ]
        # print('y_files_filt ', repr(y_files_filt))
        # print('x_files_filt ', repr(x_files_filt))
        # Triangulate 2D points
        Q_filt = [weighted_triangulation(projection_matrices_filt[i], x_files_filt[i], y_files_filt[i], likelihood_files_filt[i]) for i in range(len(id_cams_off))]
        
        # Reprojection
        if undistort_points:
            coords_2D_kpt_calc_filt = [np.array([cv2.projectPoints(np.array(Q_filt[i][:-1]), calib_params_R_filt[i][j], calib_params_T_filt[i][j], calib_params_K_filt[i][j], calib_params_dist_filt[i][j])[0].ravel() 
                                        for j in range(n_cams-nb_cams_excluded_filt[i])]) 
                                        for i in range(len(id_cams_off))]
            coords_2D_kpt_calc_filt = [[coords_2D_kpt_calc_filt[i][:,0], coords_2D_kpt_calc_filt[i][:,1]] for i in range(len(id_cams_off))]
        else:
            coords_2D_kpt_calc_filt = [reprojection(projection_matrices_filt[i], Q_filt[i]) for i in range(len(id_cams_off))]
        coords_2D_kpt_calc_filt = np.array(coords_2D_kpt_calc_filt, dtype=object)
        x_calc_filt = coords_2D_kpt_calc_filt[:,0]
        # print('x_calc_filt ', x_calc_filt)
        y_calc_filt = coords_2D_kpt_calc_filt[:,1]
        
        # Reprojection error
        error = []
        for config_off_id in range(len(x_calc_filt)):
            q_file = [(x_files_filt[config_off_id][i], y_files_filt[config_off_id][i]) for i in range(len(x_files_filt[config_off_id]))]
            q_calc = [(x_calc_filt[config_off_id][i], y_calc_filt[config_off_id][i]) for i in range(len(x_calc_filt[config_off_id]))]
            error.append( np.mean( [euclidean_distance(q_file[i], q_calc[i]) for i in range(len(q_file))] ) )
        # print('error ', error)
            
        # Choosing best triangulation (with min reprojection error)
        # print('\n', error)
        # print('len(error) ', len(error))
        # print('len(x_calc_filt) ', len(x_calc_filt))
        # print('len(likelihood_files_filt) ', len(likelihood_files_filt))
        # print('len(id_cams_off_tot) ', len(id_cams_off_tot))
        # print('min error ', np.nanmin(error))
        # print('argmin error ', np.nanargmin(error))
        error_min = np.nanmin(error)
        # print(error_min)
        best_cams = np.nanargmin(error)
        nb_cams_excluded = nb_cams_excluded_filt[best_cams]
        
        Q = Q_filt[best_cams][:-1]


        # Swap left and right sides if reprojection error still too high
        if handle_LR_swap and error_min > error_threshold_triangulation:
            # print('handle')
            n_cams_swapped = 1
            error_off_swap_min = error_min
            while error_off_swap_min > error_threshold_triangulation and n_cams_swapped < (n_cams - nb_cams_off_tot) / 2: # more than half of the cameras switched: may triangulate twice the same side
                # print('SWAP: nb_cams_off ', nb_cams_off, 'n_cams_swapped ', n_cams_swapped, 'nb_cams_off_tot ', nb_cams_off_tot)
                # Create subsets 
                id_cams_swapped = np.array(list(it.combinations(range(n_cams-nb_cams_off_tot), n_cams_swapped)))
                # print('id_cams_swapped ', id_cams_swapped)
                x_files_filt_off_swap = [[x] * len(id_cams_swapped) for x in x_files_filt]
                y_files_filt_off_swap = [[y] * len(id_cams_swapped) for y in y_files_filt]
                # print('x_files_filt_off_swap ', x_files_filt_off_swap)
                # print('y_files_filt_off_swap ', y_files_filt_off_swap)
                for id_off in range(len(id_cams_off)): # for each configuration with nb_cams_off_tot removed 
                    for id_swapped, config_swapped in enumerate(id_cams_swapped): # for each of these configurations, test all subconfigurations with with n_cams_swapped swapped
                        # print('id_off ', id_off, 'id_swapped ', id_swapped, 'config_swapped ',  config_swapped)
                        x_files_filt_off_swap[id_off][id_swapped][config_swapped] = x_files_swapped_filt[id_off][config_swapped] 
                        y_files_filt_off_swap[id_off][id_swapped][config_swapped] = y_files_swapped_filt[id_off][config_swapped]
                                
                # Triangulate 2D points
                Q_filt_off_swap = np.array([[weighted_triangulation(projection_matrices_filt[id_off], x_files_filt_off_swap[id_off][id_swapped], y_files_filt_off_swap[id_off][id_swapped], likelihood_files_filt[id_off]) 
                                                for id_swapped in range(len(id_cams_swapped))]
                                                for id_off in range(len(id_cams_off))] )
                
                # Reprojection
                if undistort_points:
                    coords_2D_kpt_calc_off_swap = [np.array([[cv2.projectPoints(np.array(Q_filt_off_swap[id_off][id_swapped][:-1]), calib_params_R_filt[id_off][j], calib_params_T_filt[id_off][j], calib_params_K_filt[id_off][j], calib_params_dist_filt[id_off][j])[0].ravel() 
                                                    for j in range(n_cams-nb_cams_off_tot)] 
                                                    for id_swapped in range(len(id_cams_swapped))])
                                                    for id_off in range(len(id_cams_off))]
                    coords_2D_kpt_calc_off_swap = np.array([[[coords_2D_kpt_calc_off_swap[id_off][id_swapped,:,0], coords_2D_kpt_calc_off_swap[id_off][id_swapped,:,1]] 
                                                    for id_swapped in range(len(id_cams_swapped))] 
                                                    for id_off in range(len(id_cams_off))])
                else:
                    coords_2D_kpt_calc_off_swap = [np.array([reprojection(projection_matrices_filt[id_off], Q_filt_off_swap[id_off][id_swapped]) 
                                                    for id_swapped in range(len(id_cams_swapped))])
                                                    for id_off in range(len(id_cams_off))]
                # print(repr(coords_2D_kpt_calc_off_swap))
                x_calc_off_swap = [c[:,0] for c in coords_2D_kpt_calc_off_swap]
                y_calc_off_swap = [c[:,1] for c in coords_2D_kpt_calc_off_swap]
                
                # Reprojection error
                # print('x_files_filt_off_swap ', x_files_filt_off_swap)
                # print('x_calc_off_swap ', x_calc_off_swap)
                error_off_swap = []
                for id_off in range(len(id_cams_off)):
                    error_percam = []
                    for id_swapped, config_swapped in enumerate(id_cams_swapped):
                        # print(id_off,id_swapped,n_cams,nb_cams_off)
                        # print(repr(x_files_filt_off_swap))
                        q_file_off_swap = [(x_files_filt_off_swap[id_off][id_swapped][i], y_files_filt_off_swap[id_off][id_swapped][i]) for i in range(n_cams - nb_cams_off_tot)]
                        q_calc_off_swap = [(x_calc_off_swap[id_off][id_swapped][i], y_calc_off_swap[id_off][id_swapped][i]) for i in range(n_cams - nb_cams_off_tot)]
                        error_percam.append( np.mean( [euclidean_distance(q_file_off_swap[i], q_calc_off_swap[i]) for i in range(len(q_file_off_swap))] ) )
                    error_off_swap.append(error_percam)
                error_off_swap = np.array(error_off_swap)
                # print('error_off_swap ', error_off_swap)
                
                # Choosing best triangulation (with min reprojection error)
                error_off_swap_min = np.min(error_off_swap)
                best_off_swap_config = np.unravel_index(error_off_swap.argmin(), error_off_swap.shape)
                
                id_off_cams = best_off_swap_config[0]
                id_swapped_cams = id_cams_swapped[best_off_swap_config[1]]
                Q_best = Q_filt_off_swap[best_off_swap_config][:-1]

                n_cams_swapped += 1

            if error_off_swap_min < error_min:
                error_min = error_off_swap_min
                best_cams = id_off_cams
                Q = Q_best
        
        # print(error_min)
        
        nb_cams_off += 1
    
    # Index of excluded cams for this keypoint
    # print('Loop ended')
    
    if 'best_cams' in locals():
        # print(id_cams_off_tot)
        # print('len(id_cams_off_tot) ', len(id_cams_off_tot))
        # print('id_cams_off_tot ', id_cams_off_tot)
        id_excluded_cams = id_cams_off_tot[best_cams]
        # print('id_excluded_cams ', id_excluded_cams)
    else:
        id_excluded_cams = list(range(n_cams))
        nb_cams_excluded = n_cams
    # print('id_excluded_cams ', id_excluded_cams)
    
    # If triangulation not successful, error = nan,  and 3D coordinates as missing values
    if error_min > error_threshold_triangulation:
        error_min = np.nan
        Q = np.array([np.nan, np.nan, np.nan])
        
    return Q, error_min, nb_cams_excluded, id_excluded_cams


def extract_files_frame_f(json_tracked_files_f, keypoints_ids, nb_persons_to_detect):
    '''
    Extract data from json files for frame f, 
    in the order of the body model hierarchy.

    INPUTS:
    - json_tracked_files_f: list of str. Paths of json_files for frame f.
    - keypoints_ids: list of int. Keypoints IDs in the order of the hierarchy.
    - nb_persons_to_detect: int

    OUTPUTS:
    - x_files, y_files, likelihood_files: [[[list of coordinates] * n_cams ] * nb_persons_to_detect]
    '''

    n_cams = len(json_tracked_files_f)
    
    x_files = [[] for n in range(nb_persons_to_detect)]
    y_files = [[] for n in range(nb_persons_to_detect)]
    likelihood_files = [[] for n in range(nb_persons_to_detect)]
    for n in range(nb_persons_to_detect):
        for cam_nb in range(n_cams):
            x_files_cam, y_files_cam, likelihood_files_cam = [], [], []
            try:
                with open(json_tracked_files_f[cam_nb], 'r') as json_f:
                    js = json.load(json_f)
                    for keypoint_id in keypoints_ids:
                        try:
                            x_files_cam.append( js['people'][n]['pose_keypoints_2d'][keypoint_id*3] )
                            y_files_cam.append( js['people'][n]['pose_keypoints_2d'][keypoint_id*3+1] )
                            likelihood_files_cam.append( js['people'][n]['pose_keypoints_2d'][keypoint_id*3+2] )
                        except:
                            x_files_cam.append( np.nan )
                            y_files_cam.append( np.nan )
                            likelihood_files_cam.append( np.nan )
            except:
                x_files_cam = [np.nan] * len(keypoints_ids)
                y_files_cam = [np.nan] * len(keypoints_ids)
                likelihood_files_cam = [np.nan] * len(keypoints_ids)
            x_files[n].append(x_files_cam)
            y_files[n].append(y_files_cam)
            likelihood_files[n].append(likelihood_files_cam)
        
    x_files = np.array(x_files)
    y_files = np.array(y_files)
    likelihood_files = np.array(likelihood_files)

    return x_files, y_files, likelihood_files


def animate_pre_post_tracking(pre_tracking_data, post_tracking_data, folder_name=None, frame_step=10, interval=100):
    """
    Create an animation comparing pre-tracking and post-tracking data with convex hulls.
    
    Parameters:
    -----------
    pre_tracking_data : list
        List of dictionaries containing pre-tracking detection data
    post_tracking_data : numpy.ndarray
        Array containing post-tracking keypoints
    folder_name : str, optional
        Name to display in the plot title
    frame_step : int, optional
        Number of frames to skip between animation steps
    interval : int, optional
        Animation speed in milliseconds
    """
    fig, (pre_ax, post_ax) = plt.subplots(1, 2, figsize=(15, 5))
    
    def update(frame):
        pre_ax.clear()
        post_ax.clear()
        pre_ax.set_title(f'Pre-Tracking: {folder_name}')
        pre_ax.set_xlim([0, 4000])
        pre_ax.set_ylim([-3000, 0])
        post_ax.set_title(f'Post-Tracking: {folder_name}')
        post_ax.set_xlim([0, 4000])
        post_ax.set_ylim([-3000, 0])

        # Pre-tracking data
        if frame < len(pre_tracking_data):
            people = pre_tracking_data[frame]
            hulls = []
            for person in people:
                x_data = person['pose_keypoints_2d'][0::3]
                y_data = person['pose_keypoints_2d'][1::3]
                valid_points = [(x, y) for x, y in zip(x_data, y_data) if x != 0 and y != 0 and not np.isnan(x) and not np.isnan(y)]
                if len(valid_points) >= 3:
                    try: # Add try-except block for robustness
                        hull = ConvexHull(valid_points)
                        hulls.append(hull)
                        pre_ax.plot([p[0] for p in valid_points], [-p[1] for p in valid_points], 'o')
                    except Exception as e:
                        print(f"Warning: Could not compute Convex Hull for frame {frame}: {e}") # Add warning if hull fails
            for hull in hulls:
                for simplex in hull.simplices:
                    pre_ax.plot(hull.points[simplex, 0], -hull.points[simplex, 1], 'k-')

        # Post-tracking data
        if frame < post_tracking_data.shape[0]:
            x_data = post_tracking_data[frame, 0::3]
            y_data = post_tracking_data[frame, 1::3]
            # Filter out both (0,0) points AND NaNs
            valid_points = [(x, y) for x, y in zip(x_data, y_data)
                            if x != 0 and y != 0 and not np.isnan(x) and not np.isnan(y)]
            if len(valid_points) >= 3:
                try: # Add try-except block for robustness
                    hull = ConvexHull(valid_points)
                    post_ax.plot([p[0] for p in valid_points], [-p[1] for p in valid_points], 'bo')
                    for simplex in hull.simplices:
                        post_ax.plot(hull.points[simplex, 0], -hull.points[simplex, 1], 'k-')
                except Exception as e:
                    print(f"Warning: Could not compute Convex Hull for frame {frame}: {e}") # Add warning if hull fails

    max_frames = max(len(pre_tracking_data), post_tracking_data.shape[0])
    frames = list(range(0, max_frames, frame_step))
    
    ani = FuncAnimation(fig, update, frames=frames, interval=interval, repeat=False)
    
    plt.tight_layout()
    plt.show(block=False)
    
    # Calculate total duration of the animation
    total_duration = len(frames) * interval / 1000  # in seconds
    
    # Close the animation window after it's done
    plt.pause(total_duration + 1)  # Animation time + 1 second
    plt.close(fig)


def triangulate_all(config_dict):
    '''
    For each frame
    For each keypoint
    - Triangulate keypoint
    - Reproject it on all cameras
    - Take off cameras until requirements are met
    Interpolate missing values
    Create trc file
    Print recap message
    
     INPUTS: 
    - a calibration file (.toml extension)
    - json files for each camera with indices matching the detected persons
    - a Config.toml file
    - a skeleton model
    
    OUTPUTS: 
    - a .trc file with 3D coordinates in Y-up system coordinates 
    '''
    
    # Read config_dict
    project_dir = config_dict.get('project').get('project_dir')
    # if batch
    session_dir = os.path.realpath(os.path.join(project_dir, '..'))
    # if single trial
    session_dir = session_dir if 'Config.toml' in os.listdir(session_dir) else os.getcwd()
    multi_person = config_dict.get('project').get('multi_person')
    pose_model = config_dict.get('pose').get('pose_model')
    frame_range = config_dict.get('project').get('frame_range')
    likelihood_threshold = config_dict.get('triangulation').get('likelihood_threshold_triangulation')
    interpolation_kind = config_dict.get('triangulation').get('interpolation')
    interp_gap_smaller_than = config_dict.get('triangulation').get('interp_if_gap_smaller_than')
    fill_large_gaps_with = config_dict.get('triangulation').get('fill_large_gaps_with')
    show_interp_indices = config_dict.get('triangulation').get('show_interp_indices')
    undistort_points = config_dict.get('triangulation').get('undistort_points')
    make_c3d = config_dict.get('triangulation').get('make_c3d')
    handle_LR_swap = config_dict.get('triangulation').get('handle_LR_swap')
    undistort_points = config_dict.get('triangulation').get('undistort_points')
    
    # Custom setting for MAE tracking frame selection
    MAE_tracking = config_dict.get('triangulation', {}).get('MAE_tracking', False)
    # Add MAE tracking threshold from config, default to 100 if not specified
    mae_threshold = config_dict.get('triangulation', {}).get('mae_tracking_threshold', 100) 
    # Add Manual frame selection setting
    Manual_frame = config_dict.get('triangulation', {}).get('Manual_frame', False) 
    
    try:
        calib_dir = [os.path.join(session_dir, c) for c in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, c)) and  'calib' in c.lower()][0]
    except:
        raise Exception(f'No .toml calibration direcctory found.')
    try:
        calib_file = glob.glob(os.path.join(calib_dir, '*.toml'))[0] # lastly created calibration file
        calib_data = toml.load(calib_file) # Load calibration data
    except Exception as e:
         raise Exception(f'Error loading calibration file {calib_file}: {e}')
    pose_dir = os.path.join(project_dir, 'pose')
    poseSync_dir = os.path.join(project_dir, 'pose-sync')
    poseTracked_dir = os.path.join(project_dir, 'pose-associated')
    
    # Projection matrix from toml calibration file
    P = computeP(calib_file, undistort=undistort_points)
    calib_params = retrieve_calib_params(calib_file)
        
    # Retrieve keypoints from model
    try: # from skeletons.py
        if pose_model.upper() == 'BODY_WITH_FEET': pose_model = 'HALPE_26'
        elif pose_model.upper() == 'WHOLE_BODY_WRIST': pose_model = 'COCO_133_WRIST'
        elif pose_model.upper() == 'WHOLE_BODY': pose_model = 'COCO_133'
        elif pose_model.upper() == 'BODY': pose_model = 'COCO_17'
        elif pose_model.upper() == 'HAND': pose_model = 'HAND_21'
        elif pose_model.upper() == 'FACE': pose_model = 'FACE_106'
        elif pose_model.upper() == 'ANIMAL': pose_model = 'ANIMAL2D_17'
        else: pass
        model = eval(pose_model)
    except:
        try: # from Config.toml
            model = DictImporter().import_(config_dict.get('pose').get(pose_model))
            if model.id == 'None':
                model.id = None
        except:
            raise NameError('{pose_model} not found in skeletons.py nor in Config.toml')
            
    keypoints_ids = [node.id for _, _, node in RenderTree(model) if node.id!=None]
    keypoints_names = [node.name for _, _, node in RenderTree(model) if node.id!=None]
    keypoints_idx = list(range(len(keypoints_ids)))
    keypoints_nb = len(keypoints_ids)
    # for pre, _, node in RenderTree(model): 
    #     print(f'{pre}{node.name} id={node.id}')
    
    # left/right swapped keypoints
    keypoints_names_swapped = ['L'+keypoint_name[1:] if keypoint_name.startswith('R') else 'R'+keypoint_name[1:] if keypoint_name.startswith('L') else keypoint_name for keypoint_name in keypoints_names]
    keypoints_names_swapped = [keypoint_name_swapped.replace('right', 'left') if keypoint_name_swapped.startswith('right') else keypoint_name_swapped.replace('left', 'right') if keypoint_name_swapped.startswith('left') else keypoint_name_swapped for keypoint_name_swapped in keypoints_names_swapped]
    keypoints_idx_swapped = [keypoints_names.index(keypoint_name_swapped) for keypoint_name_swapped in keypoints_names_swapped] # find index of new keypoint_name
    
    # Define pose directory before potentially changing it based on tracking results
    if not os.path.exists(poseSync_dir):
        pose_dir_source = pose_dir 
    else:
        pose_dir_source = poseSync_dir

    calib_cam_keys = [k for k in calib_data.keys() if isinstance(calib_data[k], dict) and k not in ['metadata', 'capture_volume', 'charuco', 'checkerboard']] 
    # Use cam_id_to_plot = 0 assuming first camera is representative
    if 0 < len(calib_cam_keys):
        camera_key = calib_cam_keys[0]
        if 'size' in calib_data[camera_key] and len(calib_data[camera_key]['size']) == 2:
            size = calib_data[camera_key]['size']
            img_width = int(size[0])
            img_height = int(size[1])
            logging.info(f"Using resolution {img_width}x{img_height} from calibration file for camera '{camera_key}'.")

    # 2d-pose files selection
    try:
        pose_listdirs_names = next(os.walk(pose_dir))[1]
        os.listdir(os.path.join(pose_dir, pose_listdirs_names[0]))[0]
    except:
        raise ValueError(f'No json files found in {pose_dir} subdirectories. Make sure you run Pose2Sim.poseEstimation() first.')
    pose_listdirs_names = sort_stringlist_by_last_number(pose_listdirs_names)
    json_dirs_names = [k for k in pose_listdirs_names if 'json' in k]
    n_cams = len(json_dirs_names)
    try: 
        json_files_names = [fnmatch.filter(os.listdir(os.path.join(poseTracked_dir, js_dir)), '*.json') for js_dir in json_dirs_names]
        pose_dir = poseTracked_dir
    except:
        try: 
            json_files_names = [fnmatch.filter(os.listdir(os.path.join(poseSync_dir, js_dir)), '*.json') for js_dir in json_dirs_names]
            pose_dir = poseSync_dir
        except:
            try:
                json_files_names = [fnmatch.filter(os.listdir(os.path.join(pose_dir, js_dir)), '*.json') for js_dir in json_dirs_names]
            except:
                raise Exception(f'No json files found in {pose_dir}, {poseSync_dir}, nor {poseTracked_dir} subdirectories. Make sure you run Pose2Sim.poseEstimation() first.')
    json_files_names = [sort_stringlist_by_last_number(js) for js in json_files_names]    

    # frame range selection
    default_f_range = [0, min([len(j) for j in json_files_names])]
    if frame_range == []:
        f_range = default_f_range
    else:
        f_range = frame_range
        
    # Interactive frame range selection if Manual_frame is True
    if Manual_frame:
        logging.info(f"\nManual_frame enabled. Preparing interactive plot for frame range selection...")
        
        # --- Plotting Logic Start ---
        # Load data for the first camera and first person for all frames
        all_frames_data_x = []
        all_frames_data_y = []
        logging.info(f"Loading 2D data for plotting (Cam 0, Person 0)..."  )
        person_id_to_plot = 0 # Assuming we plot the first person
        cam_id_to_plot = 0 # Assuming we plot the first camera view
        
        # Load 2D keypoint data for plotting
        for f_idx in tqdm(range(default_f_range[0], default_f_range[1]), desc="Loading frames for plot"): 
            json_file_name_f = None
            frame_x_data_all_people = [] # Store x data for all people in this frame
            frame_y_data_all_people = [] # Store y data for all people in this frame
            try:
                # Find the json file for the current frame index f_idx for the specific camera
                matching_files = [j for j in json_files_names[cam_id_to_plot] if int(re.split(r'(\d+)', j)[-2]) == f_idx]
                if matching_files:
                    json_file_name_f = matching_files[0]
            except IndexError: # Handles cases where json_files_names[cam_id_to_plot] might be empty or index out of range
                pass
                
            if json_file_name_f:
                json_f_path = os.path.join(pose_dir, json_dirs_names[cam_id_to_plot], json_file_name_f)
                try:
                    with open(json_f_path, 'r') as f_json:
                        js_data = json.load(f_json)
                        num_people_in_frame = len(js_data.get('people', []))
                    
                    if num_people_in_frame > 0:
                        # Extract data for all detected people in this frame for cam 0
                        x_frame_all_people, y_frame_all_people, _ = extract_files_frame_f([json_f_path], keypoints_ids, num_people_in_frame)
                        
                        # Store data for each person
                        for p_idx in range(num_people_in_frame):
                            if x_frame_all_people.shape[0] > p_idx and x_frame_all_people.shape[1] > 0:
                                frame_x_data_all_people.append(x_frame_all_people[p_idx][0])
                                frame_y_data_all_people.append(y_frame_all_people[p_idx][0])
                            else: # Handle potential inconsistencies
                                frame_x_data_all_people.append([np.nan] * keypoints_nb)
                                frame_y_data_all_people.append([np.nan] * keypoints_nb)
                except FileNotFoundError:
                     logging.debug(f"JSON file not found for frame {f_idx}: {json_f_path}")
                     # Append empty lists if file not found to maintain frame count
                except Exception as e:
                    logging.warning(f"Error processing frame {f_idx} ({json_f_path}): {e}")
                    # Append empty lists in case of other errors
            # else: File name not found, append empty lists
                
            all_frames_data_x.append(frame_x_data_all_people)
            all_frames_data_y.append(frame_y_data_all_people)

        if not all_frames_data_x: # Check if any data was loaded
             logging.warning("No 2D data loaded for plotting. Skipping interactive plot.")
        else:
            fig, ax = plt.subplots()
            plt.subplots_adjust(left=0.1, bottom=0.25)
            ax.set_title(f'Camera {cam_id_to_plot} - Frame 0')
            ax.set_xlim(0, img_width)
            ax.set_ylim(img_height, 0) # Invert Y axis for image coordinates
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True)
            
            # Define color cycle based on matplotlib defaults
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            
            # Shared variable to store the selected range
            # Initialize with the f_range determined earlier (either default or from config)
            selected_f_range = list(f_range) 
            
            # Initial plot for the start of the initial range
            initial_frame_idx = selected_f_range[0]
            plot_lines = [] # Keep track of plot objects for people
            initial_frame_x = all_frames_data_x[initial_frame_idx] if initial_frame_idx < len(all_frames_data_x) else []
            initial_frame_y = all_frames_data_y[initial_frame_idx] if initial_frame_idx < len(all_frames_data_y) else []
            for p_idx in range(len(initial_frame_x)):
                 color = color_cycle[p_idx % len(color_cycle)]
                 line, = ax.plot(initial_frame_x[p_idx], initial_frame_y[p_idx], 'o', markersize=5, color=color)
                 plot_lines.append(line)
            ax.set_title(f'Camera {cam_id_to_plot} - Frame {initial_frame_idx} ({len(initial_frame_x)} people)')
            
            # RangeSlider
            plt.subplots_adjust(left=0.1, bottom=0.30) # Adjust bottom to make space for widgets
            ax_rangeslider = plt.axes([0.1, 0.15, 0.8, 0.03]) # Position for RangeSlider
            range_slider = RangeSlider(
                ax=ax_rangeslider,
                label='Frame Range',
                valmin=default_f_range[0],
                valmax=default_f_range[1] - 1,
                valinit=(selected_f_range[0], selected_f_range[1]-1), # Use initial f_range, adjust end for inclusive display
                valstep=1
            )

            # Button
            ax_button = plt.axes([0.8, 0.05, 0.15, 0.04]) # Position for Button
            confirm_button = Button(ax_button, 'Confirm Range')

            def update_plot(frame_index):
                # Get data for the specified frame index
                current_frame_x = all_frames_data_x[frame_index] if frame_index < len(all_frames_data_x) else []
                current_frame_y = all_frames_data_y[frame_index] if frame_index < len(all_frames_data_y) else []
                
                # Remove previous people plots
                for line in plot_lines:
                    try:
                        line.remove()
                    except ValueError: # Handle cases where line might already be removed
                        pass 
                plot_lines.clear()
                
                # Plot all people for the current frame
                for p_idx in range(len(current_frame_x)):
                    color = color_cycle[p_idx % len(color_cycle)]
                    line, = ax.plot(current_frame_x[p_idx], current_frame_y[p_idx], 'o', markersize=5, color=color)
                    plot_lines.append(line)
                
                ax.set_title(f'Camera {cam_id_to_plot} - Frame {frame_index} ({len(current_frame_x)} people detected)')
                fig.canvas.draw_idle()

            def range_slider_update(val):
                # Update the plot to show the start frame of the selected range
                start_frame_display = int(val[0])
                update_plot(start_frame_display)
                # Update the shared variable in real-time (optional, button confirms final)
                # selected_f_range[0] = int(val[0])
                # selected_f_range[1] = int(val[1]) + 1 # Adjust end frame back to exclusive

            def confirm_action(event):
                # Get the final range from the slider when button is clicked
                final_range = range_slider.val
                selected_f_range[0] = int(final_range[0])
                selected_f_range[1] = int(final_range[1]) + 1 # Adjust end frame to be exclusive for range function
                logging.info(f"Range selected: [{selected_f_range[0]} - {selected_f_range[1]-1}]")
                plt.close(fig) # Close the plot window

            range_slider.on_changed(range_slider_update)
            confirm_button.on_clicked(confirm_action)

            logging.info("Displaying interactive plot. Adjust the range slider and click 'Confirm Range'...")
            plt.show() # Blocks execution until plt.close(fig) is called by the button
            
            # Use the range selected via the plot
            f_range = selected_f_range
            logging.info(f"Using selected frame range: [{f_range[0]} - {f_range[1]-1}]")
            
        # --- Plotting Logic End ---
        
    # --- MAE Based Person Tracking Start (only if MAE_tracking is True) ---
    if MAE_tracking:
        logging.info(f"\nStarting MAE based person tracking for each camera within range [{f_range[0]} - {f_range[1]-1}]...")
        
        # Define path for temporary tracked files
        pose_tracked_selected_dir = os.path.join(project_dir, 'pose-tracked-selected')
        if not os.path.exists(pose_tracked_selected_dir):
            os.makedirs(pose_tracked_selected_dir)
            
        tracked_data_available = True # Flag to track if tracking was successful for all cams
        
        # Store pre-tracking data for visualization
        pre_tracking_data_by_cam = {}
        post_tracking_data_by_cam = {}

        for cam_id, cam_dir_name in enumerate(json_dirs_names):
            logging.info(f"\nProcessing camera: {cam_dir_name}")
            cam_folder_path = os.path.join(pose_dir_source, cam_dir_name) # Use original pose_dir
            
            # Create subdirectory in the tracked directory
            save_folder_cam = os.path.join(pose_tracked_selected_dir, cam_dir_name)
            if not os.path.exists(save_folder_cam):
                os.makedirs(save_folder_cam)
                
            # Collect pre-tracking data for this camera (for visualization)
            pre_tracking_data_cam = []
            for f_idx in range(f_range[0], f_range[1]):
                frame_files = [j for j in json_files_names[cam_id] if int(re.split(r'(\d+)', j)[-2]) == f_idx]
                people_in_frame = []
                if frame_files:
                    frame_json_path = os.path.join(cam_folder_path, frame_files[0])
                    try:
                        with open(frame_json_path, 'r') as f_json:
                            js_data = json.load(f_json)
                            people_in_frame = js_data.get('people', [])
                    except Exception as e:
                        logging.debug(f"Error reading frame {f_idx} for pre-tracking data: {e}")
                pre_tracking_data_cam.append(people_in_frame)
            
            pre_tracking_data_by_cam[cam_dir_name] = pre_tracking_data_cam

            # 1. Load data for the first frame in the range for this camera
            first_frame_idx = f_range[0]
            first_frame_files = [j for j in json_files_names[cam_id] if int(re.split(r'(\d+)', j)[-2]) == first_frame_idx]
            
            people_in_first_frame = []
            first_frame_json_path = None
            if first_frame_files:
                first_frame_json_path = os.path.join(cam_folder_path, first_frame_files[0])
                try:
                    with open(first_frame_json_path, 'r') as f_json:
                        js_data = json.load(f_json)
                        people_in_first_frame = js_data.get('people', [])
                except FileNotFoundError:
                    logging.warning(f"JSON file not found for first frame {first_frame_idx} in camera {cam_dir_name}. Skipping tracking for this camera.")
                    tracked_data_available = False
                    continue
                except Exception as e:
                    logging.warning(f"Error reading first frame {first_frame_idx} JSON for camera {cam_dir_name}: {e}. Skipping tracking for this camera.")
                    tracked_data_available = False
                    continue

            if not people_in_first_frame:
                logging.warning(f"No people detected in the first frame {first_frame_idx} for camera {cam_dir_name}. Skipping tracking for this camera.")
                tracked_data_available = False
                continue

            # 2. Manual Person Selection using adapted select_person_manually logic
            logging.info(f"Please select the person to track for camera {cam_dir_name} (Frame {first_frame_idx}).")
            fig_select, ax_select = plt.subplots()
            person_patches_select = []
            keypoint_count_cam = len(people_in_first_frame[0]['pose_keypoints_2d']) # Get keypoint count from first person

            for i, person in enumerate(people_in_first_frame):
                keypoints = np.array(person['pose_keypoints_2d']).reshape(-1, 3)
                x_data = keypoints[:, 0]
                y_data = keypoints[:, 1]
                valid = (x_data != 0) & (y_data != 0) & ~np.isnan(x_data) & ~np.isnan(y_data) # Added isnan check
                x_data_valid = x_data[valid]
                y_data_valid = y_data[valid]
                valid_points = np.column_stack((x_data_valid, y_data_valid)) # Combine valid points

                # Plot points
                scat = ax_select.scatter(x_data_valid, -y_data_valid, label=f'Person {i+1}')
                # Add label near the center of the points
                if len(x_data_valid) > 0:
                     ax_select.annotate(f'{i+1}', xy=(np.mean(x_data_valid), -np.mean(y_data_valid)), color='red', fontsize=12, ha='center', va='center')

                # Calculate and plot Convex Hull if enough valid points
                if len(valid_points) >= 3:
                    try:
                        hull = ConvexHull(valid_points)
                        # Plot Convex Hull
                        for simplex in hull.simplices:
                            ax_select.plot(hull.points[simplex, 0], -hull.points[simplex, 1], 'k-')
                    except Exception as e:
                        logging.warning(f"Could not compute Convex Hull for person {i+1} in camera {cam_dir_name}, frame {first_frame_idx}: {e}")

                # Store for click detection (using original valid points before hull)
                person_patches_select.append({'scatter': scat, 'index': i, 'points': valid_points})

            # Use image dimensions from calibration if available, else default
            img_width_select, img_height_select = img_width, img_height # Use dimensions fetched earlier
            ax_select.set_title(f'Click on the person to track in Camera {cam_dir_name} (Frame {first_frame_idx})')
            ax_select.set_xlim([0, img_width_select])
            ax_select.set_ylim([-img_height_select, 0]) # Invert Y
            ax_select.grid(True)
            ax_select.legend()

            selected_person_idx_cam = [-1] # Use list to allow modification within onclick

            def onclick_select(event):
                if event.inaxes == ax_select:
                    x_click = event.xdata
                    y_click = -event.ydata # De-invert Y for calculation
                    min_dist = float('inf')
                    clicked_idx = -1
                    for patch_info in person_patches_select:
                        points = patch_info['points']
                        if len(points) > 0:
                           distances = np.sqrt((points[:, 0] - x_click)**2 + (points[:, 1] - y_click)**2)
                           dist = np.min(distances)
                           if dist < min_dist:
                                min_dist = dist
                                clicked_idx = patch_info['index']
                    if clicked_idx != -1:
                        selected_person_idx_cam[0] = clicked_idx
                        logging.info(f"Selected Person {clicked_idx + 1} for camera {cam_dir_name}.")
                        plt.close(fig_select)
                    else:
                         logging.info("No person close enough to click detected. Try clicking closer.")

            cid_select = fig_select.canvas.mpl_connect('button_press_event', onclick_select)
            plt.show() # Wait for user click

            if selected_person_idx_cam[0] == -1:
                logging.warning(f"No person selected for camera {cam_dir_name}. Skipping tracking for this camera.")
                # Decide how to handle this - skip camera, use default, raise error? Skipping for now.
                tracked_data_available = False
                continue 

            # 3. Track the selected person using MAE
            tracked_person_data_cam = {} # Store tracked data {frame_idx: keypoints_list}
            
            # Initial keypoints of the selected person
            data_to_track_cam = np.array(people_in_first_frame[selected_person_idx_cam[0]]['pose_keypoints_2d'])
            tracked_person_data_cam[first_frame_idx] = data_to_track_cam.tolist()

            logging.info(f"Tracking Person {selected_person_idx_cam[0]+1} from frame {f_range[0]+1} to {f_range[1]-1} for camera {cam_dir_name}...")
            total_min_avg_cam = 0
            count_min_avg_cam = 0

            for f_idx in tqdm(range(f_range[0] + 1, f_range[1]), desc=f"Tracking Cam {cam_id}", ncols=100, leave=False):
                current_frame_files = [j for j in json_files_names[cam_id] if int(re.split(r'(\d+)', j)[-2]) == f_idx]
                current_pos_cam = np.zeros(keypoint_count_cam) # Default to zeros if lost or no file

                tracking_lost = True
                best_match = -1

                if current_frame_files:
                    current_json_path = os.path.join(cam_folder_path, current_frame_files[0])
                    try:
                        with open(current_json_path, 'r') as f_json:
                           js_data_f = json.load(f_json)
                           people_in_frame_f = js_data_f.get('people', [])
                           
                           if people_in_frame_f:
                               mae_cam = []
                               for k, person_f in enumerate(people_in_frame_f):
                                   p1 = np.array(person_f['pose_keypoints_2d'])
                                   if len(p1) < keypoint_count_cam:
                                       p1 = np.pad(p1, (0, keypoint_count_cam - len(p1)), 'constant')
                                   elif len(p1) > keypoint_count_cam:
                                        p1 = p1[:keypoint_count_cam] 
                                        
                                   x0, y0 = np.array(data_to_track_cam[::3]), np.array(data_to_track_cam[1::3])
                                   x1, y1 = p1[::3], p1[1::3]
                                   
                                   min_len = min(len(x0), len(x1))
                                   x0, y0 = x0[:min_len], y0[:min_len]
                                   x1, y1 = x1[:min_len], y1[:min_len]

                                   valid = np.where((x0 != 0) & (y0 != 0) & (x1 != 0) & (y1 != 0))[0]
                                   
                                   if valid.size == 0:
                                       mae_val = float('inf')
                                   else:
                                       x_mae = np.mean(np.abs(x0[valid] - x1[valid]))
                                       y_mae = np.mean(np.abs(y0[valid] - y1[valid]))
                                       mae_val = np.mean([x_mae, y_mae])
                                   
                                   mae_cam.append(mae_val) # List of MAEs for this frame
                                   
                               # Filter out NaNs and Infs before finding the minimum
                               valid_mae_candidates = [(val, idx) for idx, val in enumerate(mae_cam) if not np.isnan(val) and val != float('inf')]

                               if valid_mae_candidates: # Check if there are any valid candidates
                                   min_avg_cam, I1_cam = min(valid_mae_candidates)
                                   best_match = I1_cam # Index of the best valid candidate

                                   # Compare minimum valid MAE to threshold
                                   if min_avg_cam <= mae_threshold:
                                       # Track using the best valid candidate (I1_cam)
                                       current_pos_cam = np.array(people_in_frame_f[I1_cam]['pose_keypoints_2d'])
                                       if len(current_pos_cam) < keypoint_count_cam:
                                            current_pos_cam = np.pad(current_pos_cam, (0, keypoint_count_cam - len(current_pos_cam)), 'constant')
                                       elif len(current_pos_cam) > keypoint_count_cam:
                                            current_pos_cam = current_pos_cam[:keypoint_count_cam]
                                       data_to_track_cam = current_pos_cam # Update tracker
                                       total_min_avg_cam += min_avg_cam
                                       count_min_avg_cam += 1
                                       tracking_lost = False
                                   else: # Best valid MAE is still too high
                                       logging.debug(f"Frame {f_idx}: Tracking lost in camera {cam_dir_name}. Best valid MAE {min_avg_cam:.2f} > threshold {mae_threshold}.")
                                       tracking_lost = True # Ensure tracking_lost is set
                                       
                               else: # No valid candidates found (all were NaN/Inf)
                                   min_avg_cam = float('inf') # Set MAE to infinity if no valid candidates
                                   best_match = -1
                                   logging.debug(f"Frame {f_idx}: Tracking lost in camera {cam_dir_name}. No valid MAE candidates found.")
                                   tracking_lost = True # Ensure tracking_lost is set
                                   
                           else:
                               # No people detected in frame
                               logging.debug(f"Frame {f_idx}: No people detected in camera {cam_dir_name}.")
                               tracking_lost = True # Set lost if no people
                    except FileNotFoundError:
                         logging.debug(f"JSON file not found for frame {f_idx} in camera {cam_dir_name}.")
                         tracking_lost = True # Set lost if no file
                    except Exception as e:
                         logging.warning(f"Error processing frame {f_idx} for camera {cam_dir_name}: {e}")
                         tracking_lost = True # Set lost on error
                else:
                    # No file found for this frame
                    logging.debug(f"Frame {f_idx}: No file found for camera {cam_dir_name}.")
                    tracking_lost = True # Set lost if no file

                tracked_person_data_cam[f_idx] = current_pos_cam.tolist()

            avg_min_avg_cam = total_min_avg_cam / count_min_avg_cam if count_min_avg_cam > 0 else float('inf')
            logging.info(f"Tracking complete for camera {cam_dir_name}. Average MAE (when tracked): {avg_min_avg_cam:.2f}")
            
            # 4. Save tracked data for this camera
            logging.info(f"Saving tracked data for camera {cam_dir_name} to {save_folder_cam}...")
            for f_idx in range(f_range[0], f_range[1]):
                 frame_json_file_name = None
                 # Find the original filename for this frame index to maintain naming convention
                 matching_files = [j for j in json_files_names[cam_id] if int(re.split(r'(\d+)', j)[-2]) == f_idx]
                 if matching_files:
                     frame_json_file_name = matching_files[0]
                 else:
                     # If original file didn't exist, create a name (e.g., using a template)
                     # This assumes a consistent naming like 'frame_xxxxx.json'
                     # Find example filename to deduce pattern
                     example_name = json_files_names[cam_id][0] if json_files_names[cam_id] else "output_000000000000.json"
                     name_parts = re.split(r'(\d+)', example_name)
                     frame_num_str = str(f_idx).zfill(len(name_parts[-2])) # Pad with zeros
                     frame_json_file_name = f"{name_parts[0]}{frame_num_str}{name_parts[-1]}"
                     logging.debug(f"Original file for frame {f_idx} not found, creating filename: {frame_json_file_name}")

                 frame_data_to_save = tracked_person_data_cam.get(f_idx, np.zeros(keypoint_count_cam).tolist()) # Use zeros if frame wasn't tracked

                 frame_output_data = {
                    "version": 1.3,
                    "people": [{
                        "person_id": [selected_person_idx_cam[0]], # Save selected index
                        "pose_keypoints_2d": frame_data_to_save,
                        "face_keypoints_2d": [],
                        "hand_left_keypoints_2d": [],
                        "hand_right_keypoints_2d": [],
                        "pose_keypoints_3d": [],
                        "face_keypoints_3d": [],
                        "hand_left_keypoints_3d": [],
                        "hand_right_keypoints_3d": []
                    }]
                 }
                 save_path = os.path.join(save_folder_cam, frame_json_file_name)
                 try:
                     with open(save_path, 'w') as f_out:
                         json.dump(frame_output_data, f_out)
                 except Exception as e:
                     logging.error(f"Failed to save tracked data for frame {f_idx}, camera {cam_dir_name}: {e}")
                     tracked_data_available = False # Mark as failure if saving fails
            
            # Store the tracked data for post-tracking animation
            post_tracking_data_cam = np.zeros((f_range[1] - f_range[0], keypoint_count_cam))
            for f_rel_idx, f_idx in enumerate(range(f_range[0], f_range[1])):
                if f_idx in tracked_person_data_cam:
                    post_tracking_data_cam[f_rel_idx] = tracked_person_data_cam[f_idx]
            
            post_tracking_data_by_cam[cam_dir_name] = post_tracking_data_cam

        # Visualize tracking results with animation for each camera
        if tracked_data_available:
            visualize_tracking = config_dict.get('triangulation', {}).get('visualize_tracking', True)
            if visualize_tracking:
                logging.info(f"\nVisualizing tracking results with animation...")
                for cam_dir_name in json_dirs_names:
                    logging.info(f"Animating tracking results for camera {cam_dir_name}...")
                    try:
                        animate_pre_post_tracking(
                            pre_tracking_data_by_cam[cam_dir_name],
                            post_tracking_data_by_cam[cam_dir_name],
                            folder_name=cam_dir_name,
                            frame_step=config_dict.get('triangulation', {}).get('animation_frame_step', 10),
                            interval=config_dict.get('triangulation', {}).get('animation_interval', 100)
                        )
                    except Exception as e:
                        logging.error(f"Failed to create animation for {cam_dir_name}: {e}")
                        
        # After looping through all cameras:
        if tracked_data_available:
             logging.info(f"\nMAE Based Person Tracking completed for all cameras. Using tracked data from '{pose_tracked_selected_dir}' for triangulation.")
             pose_dir = pose_tracked_selected_dir # IMPORTANT: Update pose_dir to use the tracked files
             # Update json_files_names to reflect the content of the new directory
             json_files_names = [fnmatch.filter(os.listdir(os.path.join(pose_dir, js_dir)), '*.json') for js_dir in json_dirs_names]
             json_files_names = [sort_stringlist_by_last_number(js) for js in json_files_names]
        else:
             logging.warning("\nMAE Based Person Tracking failed for one or more cameras. Proceeding with original data for triangulation.")
             # pose_dir remains the original source directory

        # --- MAE Based Person Tracking End ---

    frame_nb = f_range[1] - f_range[0]
    
    # Check that camera number is consistent between calibration file and pose folders
    if n_cams != len(P):
        raise Exception(f'Error: The number of cameras is not consistent: Found {len(P)} cameras in the calibration file, and {n_cams} cameras based on the number of pose folders.')
    
    # Triangulation
    if multi_person:
        # If tracking was successful, there should only be one person per file
        if tracked_data_available and MAE_tracking:
             nb_persons_to_detect = 1
             logging.info("Using tracked data: processing as single person.")
        else: # Fallback or original multi-person logic
            nb_persons_to_detect = max(max(count_persons_in_json(os.path.join(pose_dir, json_dirs_names[c], json_fname)) for json_fname in json_files_names[c]) for c in range(n_cams))
    else: # single_person specified in config
        nb_persons_to_detect = 1

    Q = [[[np.nan]*3]*keypoints_nb for n in range(nb_persons_to_detect)]
    Q_old = [[[np.nan]*3]*keypoints_nb for n in range(nb_persons_to_detect)]
    error = [[] for n in range(nb_persons_to_detect)]
    nb_cams_excluded = [[] for n in range(nb_persons_to_detect)]
    id_excluded_cams = [[] for n in range(nb_persons_to_detect)]
    Q_tot, error_tot, nb_cams_excluded_tot,id_excluded_cams_tot = [], [], [], []
    for f in tqdm(range(*f_range)):
        # print(f'\nFrame {f}:')        
        # Get x,y,likelihood values from files
        json_files_names_f = [[j for j in json_files_names[c] if int(re.split(r'(\d+)',j)[-2])==f] for c in range(n_cams)]
        json_files_names_f = [j for j_list in json_files_names_f for j in (j_list or ['none'])]
        json_files_f = [os.path.join(pose_dir, json_dirs_names[c], json_files_names_f[c]) for c in range(n_cams)]

        x_files, y_files, likelihood_files = extract_files_frame_f(json_files_f, keypoints_ids, nb_persons_to_detect)
        # [[[list of coordinates] * n_cams ] * nb_persons_to_detect]
        # vs. [[list of coordinates] * n_cams ] 
        
        # undistort points
        if undistort_points:
            for n in range(nb_persons_to_detect):
                points = [np.array(tuple(zip(x_files[n][i],y_files[n][i]))).reshape(-1, 1, 2).astype('float32') for i in range(n_cams)]
                undistorted_points = [cv2.undistortPoints(points[i], calib_params['K'][i], calib_params['dist'][i], None, calib_params['optim_K'][i]) for i in range(n_cams)]
                x_files[n] =  np.array([[u[i][0][0] for i in range(len(u))] for u in undistorted_points])
                y_files[n] =  np.array([[u[i][0][1] for i in range(len(u))] for u in undistorted_points])
                # This is good for slight distortion. For fisheye camera, the model does not work anymore. See there for an example https://github.com/lambdaloop/aniposelib/blob/d03b485c4e178d7cff076e9fe1ac36837db49158/aniposelib/cameras.py#L301

        # Replace likelihood by 0 if under likelihood_threshold
        with np.errstate(invalid='ignore'):
            for n in range(nb_persons_to_detect):
                x_files[n][likelihood_files[n] < likelihood_threshold] = np.nan
                y_files[n][likelihood_files[n] < likelihood_threshold] = np.nan
                likelihood_files[n][likelihood_files[n] < likelihood_threshold] = np.nan
        
        # Q_old = Q except when it has nan, otherwise it takes the Q_old value
        nan_mask = np.isnan(Q)
        Q_old = np.where(nan_mask, Q_old, Q)
        Q = [[] for n in range(nb_persons_to_detect)]
        error = [[] for n in range(nb_persons_to_detect)]
        nb_cams_excluded = [[] for n in range(nb_persons_to_detect)]
        id_excluded_cams = [[] for n in range(nb_persons_to_detect)]
        
        for n in range(nb_persons_to_detect):
            for keypoint_idx in keypoints_idx:
            # keypoints_nb = 2
            # for keypoint_idx in range(2):
            # Triangulate cameras with min reprojection error
                # print('\n', keypoints_names[keypoint_idx])
                coords_2D_kpt = np.array( (x_files[n][:, keypoint_idx], y_files[n][:, keypoint_idx], likelihood_files[n][:, keypoint_idx]) )
                coords_2D_kpt_swapped = np.array(( x_files[n][:, keypoints_idx_swapped[keypoint_idx]], y_files[n][:, keypoints_idx_swapped[keypoint_idx]], likelihood_files[n][:, keypoints_idx_swapped[keypoint_idx]] ))

                Q_kpt, error_kpt, nb_cams_excluded_kpt, id_excluded_cams_kpt = triangulation_from_best_cameras(config_dict, coords_2D_kpt, coords_2D_kpt_swapped, P, calib_params) # P has been modified if undistort_points=True

                Q[n].append(Q_kpt)
                error[n].append(error_kpt)
                nb_cams_excluded[n].append(nb_cams_excluded_kpt)
                id_excluded_cams[n].append(id_excluded_cams_kpt)
        
        # Re-identification only needed if not using tracked data or if tracking failed
        if multi_person and (not MAE_tracking or not tracked_data_available): 
            # reID persons across frames by checking the distance from one frame to another
            # print('Q before ordering ', np.array(Q)[:,:2])
            if f != f_range[0]: # Start check from the second frame in the range
                Q, associated_tuples = sort_people_sports2d(Q_old, Q)
                # Q, personsIDs_sorted, associated_tuples = sort_people(Q_old, Q)
                # print('Q after ordering ', personsIDs_sorted, associated_tuples, np.array(Q)[:,:2])
                
                error_sorted, nb_cams_excluded_sorted, id_excluded_cams_sorted = [], [], []
                for i in range(len(Q)):
                    id_in_old =  associated_tuples[:,1][associated_tuples[:,0] == i].tolist()
                    if len(id_in_old) > 0:
                        # personsIDs_sorted += id_in_old
                        error_sorted += [error[id_in_old[0]]]
                        nb_cams_excluded_sorted += [nb_cams_excluded[id_in_old[0]]]
                        id_excluded_cams_sorted += [id_excluded_cams[id_in_old[0]]]
                    else:
                        # personsIDs_sorted += [-1]
                        # Keep original if no match found (might be a new person)
                        # Need to find an available original index 'j' not already mapped from id_in_old
                        original_indices_used = associated_tuples[:,1]
                        available_original_indices = [idx for idx in range(len(error)) if idx not in original_indices_used]
                        if available_original_indices:
                           original_idx_to_use = available_original_indices[0] # Take the first available
                           error_sorted += [error[original_idx_to_use]]
                           nb_cams_excluded_sorted += [nb_cams_excluded[original_idx_to_use]]
                           id_excluded_cams_sorted += [id_excluded_cams[original_idx_to_use]]
                        else: # Should not happen if len(Q) matches len(error)
                            # Fallback: use placeholder Nans or zeros? Or original index 'i'? Let's use 'i' but this indicates a potential issue.
                           error_sorted += [error[i]]
                           nb_cams_excluded_sorted += [nb_cams_excluded[i]]
                           id_excluded_cams_sorted += [id_excluded_cams[i]]
                error, nb_cams_excluded, id_excluded_cams = error_sorted, nb_cams_excluded_sorted, id_excluded_cams_sorted
        
        # TODO: if distance > threshold, new person
        
        # Add triangulated points, errors and excluded cameras to pandas dataframes
        Q_tot.append([np.concatenate(Q[n]) for n in range(nb_persons_to_detect)])
        error_tot.append([error[n] for n in range(nb_persons_to_detect)])
        nb_cams_excluded_tot.append([nb_cams_excluded[n] for n in range(nb_persons_to_detect)])
        id_excluded_cams = [[id_excluded_cams[n][k] for k in range(keypoints_nb)] for n in range(nb_persons_to_detect)]
        id_excluded_cams_tot.append(id_excluded_cams)
            
    # fill values for if a person that was not initially detected has entered the frame 
    # Adjusted fillvalue logic needed if nb_persons_to_detect changed due to tracking
    Q_tot = [list(tpl) for tpl in zip(*it.zip_longest(*Q_tot, fillvalue=[np.nan]*keypoints_nb*3))]
    error_tot = [list(tpl) for tpl in zip(*it.zip_longest(*error_tot, fillvalue=[np.nan]*keypoints_nb*3))]
    nb_cams_excluded_tot = [list(tpl) for tpl in zip(*it.zip_longest(*nb_cams_excluded_tot, fillvalue=[np.nan]*keypoints_nb*3))]
    # Check if id_excluded_cams_tot needs similar fill value (it's a list of lists of lists)
    # Example fill value assuming it needs to match keypoint structure
    id_excluded_cams_fill = [[np.nan]*n_cams]*keypoints_nb 
    id_excluded_cams_tot = [list(tpl) for tpl in zip(*it.zip_longest(*id_excluded_cams_tot, fillvalue=[id_excluded_cams_fill]))]


    # dataframes for each person
    Q_tot = [pd.DataFrame([Q_tot_f[n] for Q_tot_f in Q_tot]) for n in range(nb_persons_to_detect)]
    error_tot = [pd.DataFrame([error_tot_f[n] for error_tot_f in error_tot]) for n in range(nb_persons_to_detect)]
    nb_cams_excluded_tot = [pd.DataFrame([nb_cams_excluded_tot_f[n] for nb_cams_excluded_tot_f in nb_cams_excluded_tot]) for n in range(nb_persons_to_detect)]
    # Corrected processing for id_excluded_cams_tot which is nested differently
    id_excluded_cams_processed = [[] for _ in range(nb_persons_to_detect)]
    for frame_data in id_excluded_cams_tot:
        for person_id in range(nb_persons_to_detect):
            if person_id < len(frame_data):
                id_excluded_cams_processed[person_id].append(frame_data[person_id])
            else: # Append fill value if person data missing for this frame (shouldn't happen with zip_longest?)
                 id_excluded_cams_processed[person_id].append(id_excluded_cams_fill) 

    id_excluded_cams_tot_df = [pd.DataFrame(id_excluded_cams_processed[n]) for n in range(nb_persons_to_detect)]
    
    
    for n in range(nb_persons_to_detect):
        error_tot[n]['mean'] = error_tot[n].mean(axis = 1)
        nb_cams_excluded_tot[n]['mean'] = nb_cams_excluded_tot[n].mean(axis = 1)
    
    # Delete participants with less than 4 valid triangulated frames
    # for each person, for each keypoint, frames to interpolate
    zero_nan_frames = [np.where( Q_tot[n].iloc[:,::3].T.eq(0) | ~np.isfinite(Q_tot[n].iloc[:,::3].T) ) for n in range(nb_persons_to_detect)]
    zero_nan_frames_per_kpt = [[zero_nan_frames[n][1][np.where(zero_nan_frames[n][0]==k)[0]] for k in range(keypoints_nb)] for n in range(nb_persons_to_detect)]
    non_nan_nb_first_kpt = [frame_nb - len(zero_nan_frames_per_kpt[n][0]) for n in range(nb_persons_to_detect)]
    deleted_person_id = [n for n in range(len(non_nan_nb_first_kpt)) if non_nan_nb_first_kpt[n]<4]

    Q_tot = [Q_tot[n] for n in range(len(Q_tot)) if n not in deleted_person_id]
    error_tot = [error_tot[n] for n in range(len(error_tot)) if n not in deleted_person_id]
    nb_cams_excluded_tot = [nb_cams_excluded_tot[n] for n in range(len(nb_cams_excluded_tot)) if n not in deleted_person_id]
    id_excluded_cams_tot_df = [id_excluded_cams_tot_df[n] for n in range(len(id_excluded_cams_tot_df)) if n not in deleted_person_id]
    nb_persons_to_detect = len(Q_tot)

    if nb_persons_to_detect ==0:
        raise Exception('No persons have been triangulated. Please check your calibration and your synchronization, or the triangulation parameters in Config.toml.')

    # IDs of excluded cameras
    # Need to carefully flatten the id_excluded_cams_tot_df structure
    cam_excluded_count = []
    for n in range(nb_persons_to_detect):
        # Flatten the list of lists of lists/arrays stored in the DataFrame column
        all_excluded_ids_person = []
        for frame_list in id_excluded_cams_tot_df[n].values.tolist(): # Iterate through rows (frames)
            for kpt_list in frame_list: # Iterate through keypoints in a frame
                # Check if kpt_list is iterable and not just nan
                if hasattr(kpt_list, '__iter__'):
                   all_excluded_ids_person.extend(kpt_list) # Add excluded cam IDs for this keypoint
        # Remove potential NaNs introduced by filling/errors before counting
        all_excluded_ids_person_cleaned = [int(id) for id in all_excluded_ids_person if not np.isnan(id)]
        count_dict = dict(Counter(all_excluded_ids_person_cleaned))
        # Normalize counts
        total_counts = frame_nb * keypoints_nb
        for cam_id in count_dict:
            count_dict[cam_id] /= total_counts if total_counts > 0 else 1
        cam_excluded_count.append(count_dict)

    
    # Optionally, for each person, for each keypoint, show indices of frames that should be interpolated
    if show_interp_indices:
        gaps = [[np.where(np.diff(zero_nan_frames_per_kpt[n][k]) > 1)[0] + 1 for k in range(keypoints_nb)] for n in range(nb_persons_to_detect)]
        sequences = [[np.split(zero_nan_frames_per_kpt[n][k], gaps[n][k]) for k in range(keypoints_nb)] for n in range(nb_persons_to_detect)]
        interp_frames = [[[f'{seq[0]}:{seq[-1]}' for seq in seq_kpt if len(seq)<=interp_gap_smaller_than and len(seq)>0] for seq_kpt in sequences[n]] for n in range(nb_persons_to_detect)]
        non_interp_frames = [[[f'{seq[0]}:{seq[-1]}' for seq in seq_kpt if len(seq)>interp_gap_smaller_than] for seq_kpt in sequences[n]] for n in range(nb_persons_to_detect)]
    else:
        interp_frames = [[[[] for _ in range(keypoints_nb)]] for _ in range(nb_persons_to_detect)] # Provide default structure
        non_interp_frames = [[[[] for _ in range(keypoints_nb)]] for _ in range(nb_persons_to_detect)] # Provide default structure


    # Interpolate missing values
    if interpolation_kind != 'none':
        for n in range(nb_persons_to_detect):
            try:
                Q_tot[n] = Q_tot[n].apply(interpolate_zeros_nans, axis=0, args=[interp_gap_smaller_than, interpolation_kind])
            except:
                logging.info(f'Interpolation was not possible for person {n}. This means that not enough points are available, which is often due to a bad calibration.')
    # Fill non-interpolated values with last valid one
    if fill_large_gaps_with == 'last_value':
        for n in range(nb_persons_to_detect): 
            Q_tot[n] = Q_tot[n].ffill(axis=0).bfill(axis=0)
    elif fill_large_gaps_with == 'zeros':
        for n in range(nb_persons_to_detect): 
            Q_tot[n].replace(np.nan, 0, inplace=True)
    
    # Create TRC file
    trc_paths = [make_trc(config_dict, Q_tot[n], keypoints_names, f_range, id_person=n) for n in range(len(Q_tot))]
    if make_c3d:
        c3d_paths = [convert_to_c3d(t) for t in trc_paths]
        
    # # Reorder TRC files - This logic might need review if MAE tracking guarantees person order
    # if multi_person and reorder_trc and len(trc_paths)>1:
    #     trc_id = retrieve_right_trc_order(trc_paths)
    #     [os.rename(t, t+'.old') for t in trc_paths]
    #     [os.rename(t+'.old', trc_paths[i]) for i, t in zip(trc_id,trc_paths)]
    #     if make_c3d:
    #         [os.rename(c, c+'.old') for c in c3d_paths]
    #         [os.rename(c+'.old', c3d_paths[i]) for i, c in zip(trc_id,c3d_paths)]
    #     error_tot = [error_tot[i] for i in trc_id]
    #     nb_cams_excluded_tot = [nb_cams_excluded_tot[i] for i in trc_id]
    #     cam_excluded_count = [cam_excluded_count[i] for i in trc_id]
    #     interp_frames = [interp_frames[i] for i in trc_id]
    #     non_interp_frames = [non_interp_frames[i] for i in trc_id]
    #     
    #     logging.info('\nThe trc and c3d files have been renamed to match the order of the static sequences.')


    # Recap message
    recap_triangulate(config_dict, error_tot, nb_cams_excluded_tot, keypoints_names, cam_excluded_count, interp_frames, non_interp_frames, trc_paths)
