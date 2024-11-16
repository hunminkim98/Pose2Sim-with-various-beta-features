#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
###########################################################################
## CAMERAS CALIBRATION                                                   ##
###########################################################################

Use this module to calibrate your cameras and save results to a .toml file.

It either converts a Qualisys calibration .qca.txt file,
Or calibrates cameras from checkerboard images.

Checkerboard calibration is based on 
https://docs.opencv.org/3.4.15/dc/dbb/tutorial_py_calibration.html.

INPUTS: 
- a calibration file in the 'calibration' folder (.qca.txt extension)
- OR folders 'calibration/intrinsics' (populated with video or about 30 images) and 'calibration/extrinsics' (populated with video or one image)
- a Config.toml file in the 'User' folder

OUTPUTS: 
- a calibration file in the 'calibration' folder (.toml extension)
'''

# TODO: DETECT WHEN WINDOW IS CLOSED
# TODO: WHEN 'Y', CATCH IF NUMBER OF IMAGE POINTS CLICKED NOT EQUAL TO NB OBJ POINTS


## INIT
from Pose2Sim.common import world_to_camera_persp, rotate_cam, quat2mat, euclidean_distance, natural_sort_key, zup2yup

import os
import logging
import pickle
import numpy as np
import pandas as pd
os.environ["OPENCV_LOG_LEVEL"]="FATAL"
import cv2
import glob
import toml
import re
from lxml import etree
import warnings
import matplotlib.pyplot as plt
from mpl_interactions import zoom_factory, panhandler
from PIL import Image
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
import scipy.optimize as optimize
import scipy.sparse as sparse
import copy


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.9.4"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def calib_qca_fun(file_to_convert_path, binning_factor=1):
    '''
    Convert a Qualisys .qca.txt calibration file
    Converts from camera view to object view, Pi rotates cameras, 
    and converts rotation with Rodrigues formula

    INPUTS:
    - file_to_convert_path: path of the .qca.text file to convert
    - binning_factor: when filming in 540p, one out of 2 pixels is binned so that the full frame is used

    OUTPUTS:
    - ret: residual reprojection error in _mm_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats
    - T: extrinsic translation: list of arrays of floats
    '''
    
    logging.info(f'Converting {file_to_convert_path} to .toml calibration file...')
    ret, C, S, D, K, R, T = read_qca(file_to_convert_path, binning_factor)
    
    RT = [world_to_camera_persp(r,t) for r, t in zip(R, T)]
    R = [rt[0] for rt in RT]
    T = [rt[1] for rt in RT]

    RT = [rotate_cam(r, t, ang_x=np.pi, ang_y=0, ang_z=0) for r, t in zip(R, T)]
    R = [rt[0] for rt in RT]
    T = [rt[1] for rt in RT]

    R = [np.array(cv2.Rodrigues(r)[0]).flatten() for r in R]
    T = np.array(T)
      
    return ret, C, S, D, K, R, T

    
def read_qca(qca_path, binning_factor):
    '''
    Reads a Qualisys .qca.txt calibration file
    Returns 6 lists of size N (N=number of cameras)
    
    INPUTS: 
    - qca_path: path to .qca.txt calibration file: string
    - binning_factor: usually 1: integer

    OUTPUTS:
    - ret: residual reprojection error in _mm_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of 3x3 arrays of floats
    - T: extrinsic translation: list of arrays of floats
    '''

    root = etree.parse(qca_path).getroot()
    ret, C, S, D, K, R, T = [], [], [], [], [], [], []
    res = []
    vid_id = []
    
    # Camera name
    for i, tag in enumerate(root.findall('cameras/camera')):
        ret += [float(tag.attrib.get('avg-residual'))]
        C += [tag.attrib.get('serial')]
        res += [int(tag.attrib.get('video_resolution')[:-1]) if tag.attrib.get('video_resolution') is not None else 1080]
        if tag.attrib.get('model') in ('Miqus Video', 'Miqus Video UnderWater', 'none'):
            vid_id += [i]
    
    # Image size
    for i, tag in enumerate(root.findall('cameras/camera/fov_video')):
        w = (float(tag.attrib.get('right')) - float(tag.attrib.get('left')) +1) /binning_factor \
            / (1080/res[i]) 
        h = (float(tag.attrib.get('bottom')) - float(tag.attrib.get('top')) +1) /binning_factor \
            / (1080/res[i])
        S += [[w, h]]
    
    # Intrinsic parameters: distorsion and intrinsic matrix
    for i, tag in enumerate(root.findall('cameras/camera/intrinsic')):
        k1 = float(tag.get('radialDistortion1'))/64/binning_factor
        k2 = float(tag.get('radialDistortion2'))/64/binning_factor
        p1 = float(tag.get('tangentalDistortion1'))/64/binning_factor
        p2 = float(tag.get('tangentalDistortion2'))/64/binning_factor
        D+= [np.array([k1, k2, p1, p2])]
        
        fu = float(tag.get('focalLengthU'))/64/binning_factor \
            / (1080/res[i])
        fv = fu / float(tag.attrib.get('pixelAspectRatio'))
        cu = (float(tag.get('centerPointU'))/64/binning_factor \
            - float(root.findall('cameras/camera/fov_video')[i].attrib.get('left'))) \
            / (1080/res[i])
        cv = (float(tag.get('centerPointV'))/64/binning_factor \
            - float(root.findall('cameras/camera/fov_video')[i].attrib.get('top'))) \
            / (1080/res[i])
        K += [np.array([fu, 0., cu, 0., fv, cv, 0., 0., 1.]).reshape(3,3)]

    # Extrinsic parameters: rotation matrix and translation vector
    for tag in root.findall('cameras/camera/transform'):
        tx = float(tag.get('x'))/1000
        ty = float(tag.get('y'))/1000
        tz = float(tag.get('z'))/1000
        r11 = float(tag.get('r11'))
        r12 = float(tag.get('r12'))
        r13 = float(tag.get('r13'))
        r21 = float(tag.get('r21'))
        r22 = float(tag.get('r22'))
        r23 = float(tag.get('r23'))
        r31 = float(tag.get('r31'))
        r32 = float(tag.get('r32'))
        r33 = float(tag.get('r33'))

        # Rotation (by-column to by-line)
        R += [np.array([r11, r12, r13, r21, r22, r23, r31, r32, r33]).reshape(3,3).T]
        T += [np.array([tx, ty, tz])]
   
    # Cameras names by natural order
    C_vid = [C[v] for v in vid_id]
    C_vid_id = [C_vid.index(c) for c in sorted(C_vid, key=natural_sort_key)]
    C_id = [vid_id[c] for c in C_vid_id]
    C = [C[c] for c in C_id]
    ret = [ret[c] for c in C_id]
    S = [S[c] for c in C_id]
    D = [D[c] for c in C_id]
    K = [K[c] for c in C_id]
    R = [R[c] for c in C_id]
    T = [T[c] for c in C_id]
   
    return ret, C, S, D, K, R, T


def read_qca(qca_path, binning_factor):
    '''
    Reads a Qualisys .qca.txt calibration file
    Returns 6 lists of size N (N=number of cameras)
    
    INPUTS: 
    - qca_path: path to .qca.txt calibration file: string
    - binning_factor: usually 1: integer

    OUTPUTS:
    - ret: residual reprojection error in _mm_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of 3x3 arrays of floats
    - T: extrinsic translation: list of arrays of floats
    '''

    root = etree.parse(qca_path).getroot()
    ret, C, S, D, K, R, T = [], [], [], [], [], [], []
    res = []
    vid_id = []
    
    # Camera name
    for i, tag in enumerate(root.findall('cameras/camera')):
        ret += [float(tag.attrib.get('avg-residual'))]
        C += [tag.attrib.get('serial')]
        res += [int(tag.attrib.get('video_resolution')[:-1]) if tag.attrib.get('video_resolution') is not None else 1080]
        if tag.attrib.get('model') in ('Miqus Video', 'Miqus Video UnderWater', 'none'):
            vid_id += [i]
    
    # Image size
    for i, tag in enumerate(root.findall('cameras/camera/fov_video')):
        w = (float(tag.attrib.get('right')) - float(tag.attrib.get('left')) +1) /binning_factor \
            / (1080/res[i]) 
        h = (float(tag.attrib.get('bottom')) - float(tag.attrib.get('top')) +1) /binning_factor \
            / (1080/res[i])
        S += [[w, h]]
    
    # Intrinsic parameters: distorsion and intrinsic matrix
    for i, tag in enumerate(root.findall('cameras/camera/intrinsic')):
        k1 = float(tag.get('radialDistortion1'))/64/binning_factor
        k2 = float(tag.get('radialDistortion2'))/64/binning_factor
        p1 = float(tag.get('tangentalDistortion1'))/64/binning_factor
        p2 = float(tag.get('tangentalDistortion2'))/64/binning_factor
        D+= [np.array([k1, k2, p1, p2])]
        
        fu = float(tag.get('focalLengthU'))/64/binning_factor \
            / (1080/res[i])
        fv = fu / float(tag.attrib.get('pixelAspectRatio'))
        cu = (float(tag.get('centerPointU'))/64/binning_factor \
            - float(root.findall('cameras/camera/fov_video')[i].attrib.get('left'))) \
            / (1080/res[i])
        cv = (float(tag.get('centerPointV'))/64/binning_factor \
            - float(root.findall('cameras/camera/fov_video')[i].attrib.get('top'))) \
            / (1080/res[i])
        K += [np.array([fu, 0., cu, 0., fv, cv, 0., 0., 1.]).reshape(3,3)]

    # Extrinsic parameters: rotation matrix and translation vector
    for tag in root.findall('cameras/camera/transform'):
        tx = float(tag.get('x'))/1000
        ty = float(tag.get('y'))/1000
        tz = float(tag.get('z'))/1000
        r11 = float(tag.get('r11'))
        r12 = float(tag.get('r12'))
        r13 = float(tag.get('r13'))
        r21 = float(tag.get('r21'))
        r22 = float(tag.get('r22'))
        r23 = float(tag.get('r23'))
        r31 = float(tag.get('r31'))
        r32 = float(tag.get('r32'))
        r33 = float(tag.get('r33'))

        # Rotation (by-column to by-line)
        R += [np.array([r11, r12, r13, r21, r22, r23, r31, r32, r33]).reshape(3,3).T]
        T += [np.array([tx, ty, tz])]
   
    # Cameras names by natural order
    C_vid = [C[v] for v in vid_id]
    C_vid_id = [C_vid.index(c) for c in sorted(C_vid, key=natural_sort_key)]
    C_id = [vid_id[c] for c in C_vid_id]
    C = [C[c] for c in C_id]
    ret = [ret[c] for c in C_id]
    S = [S[c] for c in C_id]
    D = [D[c] for c in C_id]
    K = [K[c] for c in C_id]
    R = [R[c] for c in C_id]
    T = [T[c] for c in C_id]
   
    return ret, C, S, D, K, R, T


def calib_optitrack_fun(config_dict, binning_factor=1):
    '''
    Convert an Optitrack calibration file 

    INPUTS:
    - a Config.toml file

    OUTPUTS:
    - ret: residual reprojection error in _mm_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats
    - T: extrinsic translation: list of arrays of floats
    '''
    
    logging.warning('See Readme.md to retrieve Optitrack calibration values.')
    raise NameError("See Readme.md to retrieve Optitrack calibration values.")


def calib_vicon_fun(file_to_convert_path, binning_factor=1):
    '''
    Convert a Vicon .xcp calibration file 
    Converts from camera view to object view, 
    and converts rotation with Rodrigues formula

    INPUTS:
    - file_to_convert_path: path of the .xcp file to convert
    - binning_factor: always 1 with Vicon calibration

    OUTPUTS:
    - ret: residual reprojection error in _mm_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats
    - T: extrinsic translation: list of arrays of floats
    '''
   
    logging.info(f'Converting {file_to_convert_path} to .toml calibration file...')
    ret, C, S, D, K, R, T = read_vicon(file_to_convert_path)
    
    RT = [world_to_camera_persp(r,t) for r, t in zip(R, T)]
    R = [rt[0] for rt in RT]
    T = [rt[1] for rt in RT]

    R = [np.array(cv2.Rodrigues(r)[0]).flatten() for r in R]
    T = np.array(T)
    
    return ret, C, S, D, K, R, T


def read_vicon(vicon_path):
    '''
    Reads a Vicon .xcp calibration file 
    Returns 6 lists of size N (N=number of cameras)
    
    INPUTS: 
    - vicon_path: path to .xcp calibration file: string

    OUTPUTS:
    - ret: residual reprojection error in _mm_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of 3x3 arrays of floats
    - T: extrinsic translation: list of arrays of floats
    '''

    root = etree.parse(vicon_path).getroot()
    ret, C, S, D, K, R, T = [], [], [], [], [], [], []
    vid_id = []
    
    # Camera name and image size
    for i, tag in enumerate(root.findall('Camera')):
        C += [tag.attrib.get('DEVICEID')]
        S += [[float(t) for t in tag.attrib.get('SENSOR_SIZE').split()]]
        ret += [float(tag.findall('KeyFrames/KeyFrame')[0].attrib.get('WORLD_ERROR'))]
        vid_id += [i]
        
    # Intrinsic parameters: distorsion and intrinsic matrix
    for cam_elem in root.findall('Camera'):
        try:
            dist = cam_elem.findall('KeyFrames/KeyFrame')[0].attrib.get('VICON_RADIAL2').split()[3:5]
        except:
            dist = cam_elem.findall('KeyFrames/KeyFrame')[0].attrib.get('VICON_RADIAL').split()
        D += [[float(d) for d in dist] + [0.0, 0.0]]

        fu = float(cam_elem.findall('KeyFrames/KeyFrame')[0].attrib.get('FOCAL_LENGTH'))
        fv = fu / float(cam_elem.attrib.get('PIXEL_ASPECT_RATIO'))
        cam_center = cam_elem.findall('KeyFrames/KeyFrame')[0].attrib.get('PRINCIPAL_POINT').split()
        cu, cv = [float(c) for c in cam_center]
        K += [np.array([fu, 0., cu, 0., fv, cv, 0., 0., 1.]).reshape(3,3)]

    # Extrinsic parameters: rotation matrix and translation vector
    for cam_elem in root.findall('Camera'):
        rot = cam_elem.findall('KeyFrames/KeyFrame')[0].attrib.get('ORIENTATION').split()
        R_quat = [float(r) for r in rot]
        R_mat = quat2mat(R_quat, scalar_idx=3)
        R += [R_mat]

        trans = cam_elem.findall('KeyFrames/KeyFrame')[0].attrib.get('POSITION').split()
        T += [[float(t)/1000 for t in trans]]

    # Camera names by natural order
    C_vid_id = [v for v in vid_id if ('VIDEO' or 'Video') in root.findall('Camera')[v].attrib.get('TYPE')]
    C_vid = [root.findall('Camera')[v].attrib.get('DEVICEID') for v in C_vid_id]
    C = sorted(C_vid, key=natural_sort_key)
    C_id_sorted = [i for v_sorted in C for i,v in enumerate(root.findall('Camera')) if v.attrib.get('DEVICEID')==v_sorted]
    S = [S[c] for c in C_id_sorted]
    D = [D[c] for c in C_id_sorted]
    K = [K[c] for c in C_id_sorted]
    R = [R[c] for c in C_id_sorted]
    T = [T[c] for c in C_id_sorted]

    return ret, C, S, D, K, R, T


def read_intrinsic_yml(intrinsic_path):
    '''
    Reads an intrinsic .yml calibration file
    Returns 3 lists of size N (N=number of cameras):
    - S (image size)
    - K (intrinsic parameters)
    - D (distorsion)

    N.B. : Size is calculated as twice the position of the optical center. Please correct in the .toml file if needed.
    '''
    intrinsic_yml = cv2.FileStorage(intrinsic_path, cv2.FILE_STORAGE_READ)
    cam_number = intrinsic_yml.getNode('names').size()
    N, S, D, K = [], [], [], []
    for i in range(cam_number):
        name = intrinsic_yml.getNode('names').at(i).string()
        N.append(name)
        K.append(intrinsic_yml.getNode(f'K_{name}').mat())
        D.append(intrinsic_yml.getNode(f'dist_{name}').mat().flatten()[:-1])
        S.append([K[i][0,2]*2, K[i][1,2]*2])
    return N, S, K, D


def read_extrinsic_yml(extrinsic_path):
    '''
    Reads an intrinsic .yml calibration file
    Returns 3 lists of size N (N=number of cameras):
    - R (extrinsic rotation, Rodrigues vector)
    - T (extrinsic translation)
    '''
    extrinsic_yml = cv2.FileStorage(extrinsic_path, cv2.FILE_STORAGE_READ)
    cam_number = extrinsic_yml.getNode('names').size()
    N, R, T = [], [], []
    for i in range(cam_number):
        name = extrinsic_yml.getNode('names').at(i).string()
        N.append(name)
        R.append(extrinsic_yml.getNode(f'R_{name}').mat().flatten()) # R_1 pour Rodrigues, Rot_1 pour matrice
        T.append(extrinsic_yml.getNode(f'T_{name}').mat().flatten())
    return N, R, T


def calib_easymocap_fun(files_to_convert_paths, binning_factor=1):
    '''
    Reads EasyMocap .yml calibration files

    INPUTS:
    - files_to_convert_paths: paths of the intri.yml and extri.yml calibration files to convert
    - binning_factor: always 1 with easymocap calibration

    OUTPUTS:
    - ret: residual reprojection error in _mm_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats
    - T: extrinsic translation: list of arrays of floats
    '''

    extrinsic_path, intrinsic_path = files_to_convert_paths
    C, S, K, D = read_intrinsic_yml(intrinsic_path)
    _, R, T = read_extrinsic_yml(extrinsic_path)
    ret = [np.nan]*len(C)
    
    return ret, C, S, D, K, R, T


def calib_biocv_fun(files_to_convert_paths, binning_factor=1):
    '''
    Convert bioCV calibration files.

    INPUTS:
    - files_to_convert_paths: paths of the calibration files to convert (no extension)
    - binning_factor: always 1 with biocv calibration

    OUTPUTS:
    - ret: residual reprojection error in _mm_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats
    - T: extrinsic translation: list of arrays of floats
    '''
    
    logging.info(f'Converting {[os.path.basename(f) for f in files_to_convert_paths]} to .toml calibration file...')

    ret, C, S, D, K, R, T = [], [], [], [], [], [], []
    for i, f_path in enumerate(files_to_convert_paths):
        with open(f_path) as f:
            calib_data = f.read().split('\n')
            ret += [np.nan]
            C += [f'cam_{str(i).zfill(2)}']
            S += [[int(calib_data[0]), int(calib_data[1])]]
            D += [[float(d) for d in calib_data[-2].split(' ')[:4]]]
            K += [np.array([k.strip().split(' ') for k in calib_data[2:5]], np.float32)]
            RT = np.array([k.strip().split(' ') for k in calib_data[6:9]], np.float32)
            R+=[cv2.Rodrigues(RT[:,:3])[0].squeeze()]
            T+=[RT[:,3]/1000]
                        
    return ret, C, S, D, K, R, T
    

def calib_opencap_fun(files_to_convert_paths, binning_factor=1):
    '''
    Convert OpenCap calibration files.
    
    Extrinsics in OpenCap are calculated with a vertical board for the world frame.
    As we want the world frame to be horizontal, we need to rotate cameras by -Pi/2 around x in the world frame. 
    T is good the way it is.

    INPUTS:
    - files_to_convert_paths: paths of the .pickle calibration files to convert
    - binning_factor: always 1 with opencap calibration

    OUTPUTS:
    - ret: residual reprojection error in _mm_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats
    - T: extrinsic translation: list of arrays of floats
    '''
    
    logging.info(f'Converting {[os.path.basename(f) for f in files_to_convert_paths]} to .toml calibration file...')
    
    ret, C, S, D, K, R, T = [], [], [], [], [], [], []
    for i, f_path in enumerate(files_to_convert_paths):
        with open(f_path, 'rb') as f_pickle:
            calib_data = pickle.load(f_pickle)
            ret += [np.nan]
            C += [f'cam_{str(i).zfill(2)}']
            S += [list(calib_data['imageSize'].squeeze()[::-1])]
            D += [list(calib_data['distortion'][0][:-1])]
            K += [calib_data['intrinsicMat']]
            R_cam = calib_data['rotation']
            T_cam = calib_data['translation'].squeeze()
            
            # Rotate cameras by Pi/2 around x in world frame -> could have just switched some columns in matrix
            # camera frame to world frame
            R_w, T_w = world_to_camera_persp(R_cam, T_cam)
            # x_rotate -Pi/2 and z_rotate Pi
            R_w_90, T_w_90 = rotate_cam(R_w, T_w, ang_x=-np.pi/2, ang_y=0, ang_z=np.pi)
            # world frame to camera frame
            R_c_90, T_c_90 = world_to_camera_persp(R_w_90, T_w_90)
            # store in R and T
            R+=[cv2.Rodrigues(R_c_90)[0].squeeze()]
            T+=[T_cam/1000]

    return ret, C, S, D, K, R, T
    

def calib_calc_fun(calib_dir, intrinsics_config_dict, extrinsics_config_dict):
    '''
    Calibrates intrinsic and extrinsic parameters
    from images or videos of a checkerboard
    or retrieve them from a file

    INPUTS:
    - calib_dir: directory containing intrinsic and extrinsic folders, each populated with camera directories
    - intrinsics_config_dict: dictionary of intrinsics parameters (overwrite_intrinsics, show_detection_intrinsics, intrinsics_extension, extract_every_N_sec, intrinsics_corners_nb, intrinsics_square_size, intrinsics_marker_size, intrinsics_aruco_dict)
    - extrinsics_config_dict: dictionary of extrinsics parameters (calculate_extrinsics, show_detection_extrinsics, extrinsics_extension, extrinsics_corners_nb, extrinsics_square_size, extrinsics_marker_size, extrinsics_aruco_dict, object_coords_3d)

    OUTPUTS:
    - ret: residual reprojection error in _px_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats (Rodrigues)
    - T: extrinsic translation: list of arrays of floats
    '''
    
    overwrite_intrinsics = intrinsics_config_dict.get('overwrite_intrinsics')
    calculate_extrinsics = extrinsics_config_dict.get('calculate_extrinsics')

    # retrieve intrinsics if calib_file found and if overwrite_intrinsics=False
    try:
        calib_file = glob.glob(os.path.join(calib_dir, f'Calib*.toml'))[0]
    except:
        pass
    if not overwrite_intrinsics and 'calib_file' in locals():
        logging.info(f'\nPreexisting calibration file found: \'{calib_file}\'.')
        logging.info(f'\nRetrieving intrinsic parameters from file. Set "overwrite_intrinsics" to true in Config.toml to recalculate them.')
        calib_file = glob.glob(os.path.join(calib_dir, f'Calib*.toml'))[0]
        calib_data = toml.load(calib_file)

        ret, C, S, D, K, R, T = [], [], [], [], [], [], []
        for cam in calib_data:
            if cam != 'metadata':
                ret += [0.0]
                C += [calib_data[cam]['name']]
                S += [calib_data[cam]['size']]
                K += [np.array(calib_data[cam]['matrix'])]
                D += [calib_data[cam]['distortions']]
                R += [[0.0, 0.0, 0.0]]
                T += [[0.0, 0.0, 0.0]]
        nb_cams_intrinsics = len(C)
    
    # calculate intrinsics otherwise
    else:
        logging.info(f'\nCalculating intrinsic parameters...')
        ret, C, S, D, K, R, T = calibrate_intrinsics(calib_dir, intrinsics_config_dict)
        
        # Ensure D is initialized properly
        for i in range(len(D)):
            if D[i] is None or len(D[i]) == 0:
                D[i] = np.zeros(4, dtype=np.float64)  # Default distortion coefficients

    # calculate extrinsics
    if calculate_extrinsics:
        logging.info(f'\nCalculating extrinsic parameters...')
        
        # check that the number of cameras is consistent
        nb_cams_extrinsics = len(next(os.walk(os.path.join(calib_dir, 'extrinsics')))[1])
        if nb_cams_intrinsics != nb_cams_extrinsics:
            raise Exception(f'Error: The number of cameras is not consistent:\
                    Found {nb_cams_intrinsics} cameras based on the number of intrinsic folders or on calibration file data,\
                    and {nb_cams_extrinsics} cameras based on the number of extrinsic folders.')
        ret, C, S, D, K, R, T = calibrate_extrinsics(calib_dir, extrinsics_config_dict, C, S, K, D)
    else:
        logging.info(f'\nExtrinsic parameters won\'t be calculated. Set "calculate_extrinsics" to true in Config.toml to calculate them.')

    return ret, C, S, D, K, R, T


def project_points_safely(object_points, rvec, tvec, camera_matrix, dist_coeffs):
    """
    Safely project 3D points to 2D image points.
    
    Args:
        object_points: 3D points in world coordinates
        rvec: Rotation vector
        tvec: Translation vector  
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        
    Returns:
        Projected 2D points or None if projection fails
    """
    try:
        dist_coeffs = np.array(dist_coeffs).flatten()
        return cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    except Exception as e:
        logging.error(f"Error in point projection: {str(e)}")
        return None

def compute_reprojection_error(projected_points, observed_points):
    """
    Compute reprojection error between projected and observed points.
    This is the standard method used throughout the code.
    
    Args:
        projected_points: Projected 2D points (N x 2)
        observed_points: Observed 2D points (N x 2)
        
    Returns:
        float: RMS error in pixels
    """
    # Ensure points are in the right shape
    projected_points = np.array(projected_points).reshape(-1, 2)
    observed_points = np.array(observed_points).reshape(-1, 2)
    
    # Calculate error for each point
    errors = projected_points - observed_points
    
    # Calculate Euclidean distance for each point
    distances = np.sqrt(np.sum(errors * errors, axis=1))
    
    # Return RMS error
    return float(np.sqrt(np.mean(distances * distances)))

def compute_standardized_rms_error(proj_points, image_points):
    """
    Compute standardized RMS error consistently across all stages.
    Returns error in pixels.
    """
    diff = proj_points - image_points
    # Calculate Euclidean distance for each point
    distances = np.sqrt(np.sum(diff * diff, axis=1))
    # Calculate RMS
    rms = np.sqrt(np.mean(distances * distances))
    return rms

def create_objective_function(cameras_params, object_points, image_points, initial_K, initial_D):
    """
    Create objective function for bundle adjustment optimization.
    
    Args:
        cameras_params: List of camera parameters
        object_points: 3D object points
        image_points: Observed 2D points
        initial_K: Initial camera matrices
        initial_D: Initial distortion coefficients
        
    Returns:
        Objective function for optimization
    """
    n_cams = len(cameras_params)
    
    def objective(params):
        total_error = []
        
        # For each camera
        for i in range(n_cams):
            # Extract camera parameters
            cam_params = params[i * 9:(i + 1) * 9]
            rvec = cam_params[0:3]
            tvec = cam_params[3:6]
            
            # Extract and reconstruct intrinsic parameters
            fx = cam_params[6]
            fy = cam_params[7]
            cx = cam_params[8]
            
            K = np.array([
                [fx, 0, cx],
                [0, fy, initial_K[i][1,1]],  # Keep skew parameter fixed
                [0, 0, 1]
            ])
            
            # Project points
            projected_pts = project_points_safely(object_points, rvec, tvec, K, initial_D[i])[0]
            
            if projected_pts is None:
                return np.full(len(image_points[i].flatten()), 1e6)
            
            # Compute reprojection error
            error = (projected_pts - image_points[i]).ravel()
            total_error.extend(error)
        
        return np.array(total_error)
    
    return objective

def initialize_camera_parameters(cameras_params):
    """
    Initialize camera parameters for optimization including intrinsics.
    
    Args:
        cameras_params: List of camera parameters containing R, T, and K
        
    Returns:
        initial_params: Flattened array of rotation, translation, and intrinsic parameters
    """
    n_cams = len(cameras_params)
    initial_params = np.zeros(n_cams * 9)  # 3 for R, 3 for T, 3 for K
    
    for i in range(n_cams):
        # Rotation and translation
        r = cv2.Rodrigues(cameras_params[i]['R'])[0].flatten()
        t = cameras_params[i]['T'].flatten()
        
        # Intrinsic parameters (fx, fy, cx)
        K = cameras_params[i]['K']
        k_params = np.array([K[0,0], K[1,1], K[0,2]])
        
        # Combine parameters
        initial_params[i*9:(i+1)*9] = np.concatenate([r, t, k_params])
    
    return initial_params

def update_camera_parameters(cameras_params, optimized_params):
    """
    Update camera parameters with optimization results.
    
    Args:
        cameras_params: List of camera parameters to update
        optimized_params: Optimized parameters from bundle adjustment
        
    Returns:
        Updated camera parameters
    """
    n_cams = len(cameras_params)
    for i in range(n_cams):
        params = optimized_params[i*9:(i+1)*9]
        rvec = params[0:3]
        tvec = params[3:6]
        fx = params[6]
        fy = params[7]
        cx = params[8]
        
        # Update rotation and translation
        cameras_params[i]['R'] = cv2.Rodrigues(rvec)[0]
        cameras_params[i]['T'] = tvec.reshape(3, 1)
        
        # Update intrinsic parameters
        K = cameras_params[i]['K'].copy()
        K[0,0] = fx  # fx
        K[1,1] = fy  # fy
        K[0,2] = cx  # cx
        cameras_params[i]['K'] = K
    
    return cameras_params

def bundle_adjust_cameras(cameras_params, object_points, image_points, initial_K, initial_D):
    """
    Perform bundle adjustment to optimize camera extrinsic parameters.
    """
    # Validate and prepare inputs
    n_cams = len(cameras_params)
    if n_cams == 0:
        raise ValueError("No cameras to optimize")
    
    if len(object_points) == 0:
        raise ValueError("No object points provided")
        
    try:
        # Store previous errors and best parameters for each camera
        prev_errors = [float('inf')] * n_cams
        best_params = [None] * n_cams
        best_errors = [float('inf')] * n_cams
        
        # Initialize optimization parameters
        initial_params = initialize_camera_parameters(cameras_params)
        params_per_camera = len(initial_params) // n_cams
        
        # Create objective function
        def objective(params):
            total_error = 0
            camera_errors = []
            all_errors = []
            
            for i in range(n_cams):
                start_idx = i * params_per_camera
                cam_params = params[start_idx:start_idx + params_per_camera]
                
                # Extract camera parameters
                R = cv2.Rodrigues(cam_params[:3])[0]
                T = cam_params[3:6]
                
                # Project points
                proj_points = project_points_safely(object_points, cam_params[:3], T, initial_K[i], initial_D[i])[0]
                
                # Compute error using standard method
                rms_error = compute_reprojection_error(proj_points, image_points[i])
                camera_errors.append(rms_error)
                
                # For optimization, collect all point errors
                point_errors = (proj_points - image_points[i]).ravel()
                all_errors.extend(point_errors)
                
                # Track improvement and store best parameters
                if rms_error < best_errors[i]:
                    best_errors[i] = rms_error
                    best_params[i] = cam_params.copy()
                    logging.info(f"Camera {i+1}: RMS error improved to {rms_error:.3f} pixels")
            
            # Print summary every few iterations
            if np.random.random() < 0.1:  # Print roughly every 10th iteration
                error_summary = " | ".join([f"Cam{i+1}: {err:.3f}px" for i, err in enumerate(camera_errors)])
                logging.info(f"Current RMS errors - {error_summary}")
            
            return np.array(all_errors)
        
        logging.info("\nStarting bundle adjustment optimization...")
        logging.info(f"Number of cameras: {n_cams}")
        logging.info(f"Number of points per camera: {len(object_points)}")
        
        # Run optimization with improved parameters
        result = optimize.least_squares(
            objective,
            initial_params,
            method='lm',
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=5000,
            verbose=2
        )
        
        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")
        
        # Construct final parameters using best results for each camera
        final_params = np.zeros_like(initial_params)
        for i in range(n_cams):
            if best_params[i] is not None:
                start_idx = i * params_per_camera
                final_params[start_idx:start_idx + params_per_camera] = best_params[i]
            else:
                # If no improvement was found, use the original optimization result
                start_idx = i * params_per_camera
                final_params[start_idx:start_idx + params_per_camera] = result.x[start_idx:start_idx + params_per_camera]
        
        # Final error calculation using best parameters
        final_errors = []
        for i in range(n_cams):
            start_idx = i * params_per_camera
            cam_params = final_params[start_idx:start_idx + params_per_camera]
            proj_points = project_points_safely(object_points, cam_params[:3], cam_params[3:6], initial_K[i], initial_D[i])[0]
            rms_error = compute_reprojection_error(proj_points, image_points[i])
            final_errors.append(rms_error)
        
        logging.info("\nFinal RMS errors (using best parameters):")
        for i, error in enumerate(final_errors):
            logging.info(f"Camera {i+1}: {error:.3f} pixels")
            
        # Update camera parameters using best results
        cameras_params = update_camera_parameters(cameras_params, final_params)
        
        return cameras_params

    except Exception as e:
        logging.error(f"Bundle adjustment failed: {str(e)}")
        logging.error(f"object_points shape: {object_points.shape}")
        logging.error(f"Number of cameras: {len(cameras_params)}")
        for i, pts in enumerate(image_points):
            logging.error(f"image_points[{i}] shape: {pts.shape}")
        return cameras_params

def get_initial_camera_estimate(object_coords_3d, imgp, K_mat, D_vec):
    """
    Get initial camera pose estimate using PnP.
    
    Args:
        object_coords_3d: 3D object points
        imgp: 2D image points
        K_mat: Camera matrix
        D_vec: Distortion coefficients
        
    Returns:
        Tuple of (success, rotation vector, translation vector)
    """
    # Try EPNP first
    success, r, t = cv2.solvePnP(object_coords_3d, imgp, K_mat, D_vec, flags=cv2.SOLVEPNP_EPNP)
    
    if not success:
        # Try ITERATIVE if EPNP fails
        success, r, t = cv2.solvePnP(object_coords_3d, imgp, K_mat, D_vec, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return False, None, None
    
    # Refine initial estimate
    success, r, t = cv2.solvePnP(object_coords_3d, imgp, K_mat, D_vec,
                               rvec=r, tvec=t,
                               useExtrinsicGuess=True,
                               flags=cv2.SOLVEPNP_ITERATIVE)
    
    return success, r, t

def visualize_reprojection(img_path, image_points, projected_points, error):
    """
    Visualize reprojection results.
    
    Args:
        img_path: Path to image file
        image_points: Original 2D points
        projected_points: Reprojected 2D points
        error: Reprojection error
    """
    img = cv2.imread(img_path)
    if img is None:
        cap = cv2.VideoCapture(img_path)
        ret_vid, img = cap.read()
        cap.release()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Draw points
    for pt_img, pt_proj in zip(image_points, projected_points):
        cv2.circle(img, tuple(map(int, pt_img)), 3, (0,255,0), -1)  # Original points in green
        cv2.circle(img, tuple(map(int, pt_proj)), 3, (0,0,255), 1)  # Projected points in red
    
    # Add legend
    cv2.putText(img, f'Reprojection error: {error:.2f} px', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    # Display
    plt.figure(figsize=(12,8))
    plt.imshow(img)
    plt.title(f'Reprojection Results')
    plt.axis('off')
    plt.show()

def process_camera_extrinsics(cam, img_vid_files, object_coords_3d, K, D, show_reprojection_error, extrinsics_method):
    """
    Process extrinsic calibration for a single camera.
    
    Args:
        cam: Camera name/identifier
        img_vid_files: List of image/video files
        object_coords_3d: 3D object points
        K: Camera matrix
        D: Distortion coefficients
        show_reprojection_error: Whether to show reprojection visualization
        extrinsics_method: Method for extrinsic calibration ('board' or 'scene')
        
    Returns:
        Tuple of (image points, camera parameters)
    """
    logging.info(f'\nCamera {cam}:')
    
    if len(img_vid_files) == 0:
        raise ValueError(f'No files found for camera {cam}')
        
    # Get image points
    if extrinsics_method == 'board':
        imgp, objp = findCorners(img_vid_files[0], extrinsics_corners_nb, objp=object_coords_3d, show=show_reprojection_error)
    else:
        img = cv2.imread(img_vid_files[0])
        if img is None:
            cap = cv2.VideoCapture(img_vid_files[0])
            ret_vid, img = cap.read()
            cap.release()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgp, objp = imgp_objp_visualizer_clicker(img, imgp=[], objp=object_coords_3d, img_path=img_vid_files[0])
    
    if len(imgp) == 0:
        raise ValueError(f'No points found for camera {cam}')
    
    # Ensure proper numpy array formatting
    objp = np.array(objp, dtype=np.float32)
    imgp = np.array(imgp, dtype=np.float32)
    K_mat = np.array(K, dtype=np.float32)
    D_vec = np.array(D, dtype=np.float32).flatten()
    
    # Get initial extrinsic estimate
    success, r, t = get_initial_camera_estimate(objp, imgp, K_mat, D_vec)
    if not success:
        raise RuntimeError(f'PnP failed for camera {cam}')
    
    return imgp, {'K': K_mat, 'D': D_vec, 'R': cv2.Rodrigues(r)[0], 'T': t.flatten()}

def calibrate_extrinsics(calib_dir, extrinsics_config_dict, C, S, K, D):
    '''
    Calibrates extrinsic parameters with bundle adjustment
    from an image or the first frame of a video
    of a checkerboard or of measured clues on the scene
    '''
    try:
        extrinsics_cam_listdirs_names = next(os.walk(os.path.join(calib_dir, 'extrinsics')))[1]
    except StopIteration:
        logging.exception(f'Error: No {os.path.join(calib_dir, "extrinsics")} folder found.')
        raise Exception(f'Error: No {os.path.join(calib_dir, "extrinsics")} folder found.')
    
    extrinsics_method = extrinsics_config_dict.get('extrinsics_method')
    ret = []
    R = []
    T = []
    
    if extrinsics_method in ('board', 'scene'):
        # Define 3D object points
        if extrinsics_method == 'board':
            extrinsics_corners_nb = extrinsics_config_dict.get('board').get('extrinsics_corners_nb')
            extrinsics_square_size = extrinsics_config_dict.get('board').get('extrinsics_square_size') / 1000
            object_coords_3d = np.zeros((extrinsics_corners_nb[0] * extrinsics_corners_nb[1], 3), np.float32)
            object_coords_3d[:, :2] = np.mgrid[0:extrinsics_corners_nb[0], 0:extrinsics_corners_nb[1]].T.reshape(-1, 2)
            object_coords_3d[:, :2] = object_coords_3d[:, 0:2] * extrinsics_square_size
        else:
            object_coords_3d = np.array(extrinsics_config_dict.get('scene').get('object_coords_3d'), np.float32)
                
        # Save reference 3D coordinates as trc
        calib_output_path = os.path.join(calib_dir, f'Object_points.trc')
        trc_write(object_coords_3d, calib_output_path)
    
        # Process each camera
        cameras_params = []
        image_points = []
        
        for i, cam in enumerate(extrinsics_cam_listdirs_names):
            try:
                # Get image/video files
                extrinsics_extension = [extrinsics_config_dict.get('board').get('extrinsics_extension') if extrinsics_method == 'board'
                                      else extrinsics_config_dict.get('scene').get('extrinsics_extension')][0]
                show_reprojection_error = [extrinsics_config_dict.get('board').get('show_reprojection_error') if extrinsics_method == 'board'
                                         else extrinsics_config_dict.get('scene').get('show_reprojection_error')][0]
                
                img_vid_files = glob.glob(os.path.join(calib_dir, 'extrinsics', cam, f'*.{extrinsics_extension}'))
                
                # Process camera extrinsics
                imgp, camera_params = process_camera_extrinsics(
                    cam, img_vid_files, object_coords_3d, K[i], D[i],
                    show_reprojection_error, extrinsics_method
                )
                
                cameras_params.append(camera_params)
                image_points.append(imgp.reshape(-1, 2))
                
            except Exception as e:
                logging.error(f"Error processing camera {cam}: {str(e)}")
                continue
        
        # Perform bundle adjustment
        logging.info('\nPerforming global bundle adjustment...')
        try:
            optimized_params = bundle_adjust_cameras(cameras_params, object_coords_3d, image_points, initial_K=K, initial_D=D)
            
            # Extract optimized parameters and compute errors
            for i, params in enumerate(optimized_params):
                r = cv2.Rodrigues(params['R'])[0].flatten()
                t = params['T'].flatten()
                
                # Compute reprojection error
                proj_points = project_points_safely(object_coords_3d, r, t, K[i], D[i])[0].reshape(-1, 2)
                error = compute_reprojection_error(proj_points, image_points[i])
                
                ret.append(float(error))
                R.append(r)
                T.append(t)
                
                # Visualize results if requested
                if show_reprojection_error:
                    visualize_reprojection(img_vid_files[0], image_points[i], proj_points, error)
            
            logging.info('Bundle adjustment completed successfully')
            
        except Exception as e:
            logging.error(f"Bundle adjustment failed: {str(e)}")
            # Use initial PnP estimates if bundle adjustment fails
            for i, params in enumerate(cameras_params):
                r = cv2.Rodrigues(params['R'])[0].flatten()
                t = params['T'].flatten()
                
                # Compute reprojection error
                proj_points = project_points_safely(object_coords_3d, r, t, K[i], D[i])[0].reshape(-1, 2)
                error = compute_reprojection_error(proj_points, image_points[i])
                
                ret.append(float(error))
                R.append(r)
                T.append(t)
            
            logging.warning('Using initial PnP estimates due to bundle adjustment failure')
    
    elif extrinsics_method == 'keypoints':
        logging.info('Calibration based on keypoints is not available yet.')
        
    else:
        raise ValueError('Wrong value for extrinsics_method')

    return ret, C, S, D, K, R, T

def findCorners(img_path, corner_nb, objp=[], show=True):
    '''
    Find corners in the photo of a checkerboard.
    Press 'Y' to accept detection, 'N' to dismiss this image, 'C' to click points by hand.
    Left click to add a point, right click to remove the last point.
    Use mouse wheel to zoom in and out and to pan.
    
    Make sure that: 
    - the checkerboard is surrounded by a white border
    - rows != lines, and row is even if lines is odd (or conversely)
    - it is flat and without reflections
    - corner_nb correspond to _internal_ corners
    
    INPUTS:
    - img_path: path to image (or video)
    - corner_nb: [H, W] internal corners in checkerboard: list of two integers [4,7]
    - optionnal: show: choose whether to show corner detections
    - optionnal: objp: array [3d corner coordinates]

    OUTPUTS:
    - imgp_confirmed: array of [[2d corner coordinates]]
    - only if objp!=[]: objp_confirmed: array of [3d corner coordinates]
    '''

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # stop refining after 30 iterations or if error less than 0.001px
    
    img = cv2.imread(img_path)
    if img is None:
        cap = cv2.VideoCapture(img_path)
        ret, img = cap.read()
        cap.release()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Find corners
    ret, corners = cv2.findChessboardCorners(gray, corner_nb, None)
    # If corners are found, refine corners
    if ret == True: 
        imgp = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        logging.info(f'{os.path.basename(img_path)}: Corners found.')
        
        if show:
            # Draw corners
            cv2.drawChessboardCorners(img, corner_nb, imgp, ret)
            # Add corner index 
            for i, corner in enumerate(imgp):
                if i in [0, corner_nb[0]-1, corner_nb[0]*(corner_nb[1]-1), corner_nb[0]*corner_nb[1] -1]:
                    x, y = corner.ravel()
                    cv2.putText(img, str(i+1), (int(x)-5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 7) 
                    cv2.putText(img, str(i+1), (int(x)-5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,0), 2) 
            
            # Visualizer and key press event handler
            for var_to_delete in ['imgp_confirmed', 'objp_confirmed']:
                if var_to_delete in globals():
                    del globals()[var_to_delete]
            imgp_objp_confirmed = imgp_objp_visualizer_clicker(img, imgp=imgp, objp=objp, img_path=img_path)
        else:
            imgp_objp_confirmed = imgp

    # If corners are not found, dismiss or click points by hand
    else:
        logging.info(f'{os.path.basename(img_path)}: Corners not found. To label them by hand, set "show_detection_intrinsics" to true in the Config.toml file.')
        if show:
            # Visualizer and key press event handler
            imgp_objp_confirmed = imgp_objp_visualizer_clicker(img, imgp=[], objp=objp, img_path=img_path)
        else:
            imgp_objp_confirmed = []

    return imgp_objp_confirmed


def imgp_objp_visualizer_clicker(img, imgp=[], objp=[], img_path=''):
    '''
    Shows image img. 
    If imgp is given, displays them in green
    If objp is given, can be displayed in a 3D plot if 'C' is pressed.
    If img_path is given, just uses it to name the window

    If 'Y' is pressed, closes all and returns confirmed imgp and (if given) objp
    If 'N' is pressed, closes all and returns nothing
    If 'C' is pressed, allows clicking imgp by hand. If objp is given:
        Displays them in 3D as a helper. 
        Left click to add a point, right click to remove the last point.
        Press 'H' to indicate that one of the objp is not visible on image
        Closes all and returns imgp and objp if all points have been clicked
    Allows for zooming and panning with middle click
    
    INPUTS:
    - img: image opened with openCV
    - optional: imgp: detected image points, to be accepted or not. Array of [[2d corner coordinates]]
    - optionnal: objp: array of [3d corner coordinates]
    - optional: img_path: path to image

    OUTPUTS:
    - imgp_confirmed: image points that have been correctly identified. array of [[2d corner coordinates]]
    - only if objp!=[]: objp_confirmed: array of [3d corner coordinates]
    '''
    global old_image_path
    old_image_path = img_path
                                 
    def on_key(event):
        '''
        Handles key press events:
        'Y' to return imgp, 'N' to dismiss image, 'C' to click points by hand.
        Left click to add a point, 'H' to indicate it is not visible, right click to remove it.
        '''

        global imgp_confirmed, objp_confirmed, objp_confirmed_notok, scat, ax_3d, fig_3d, events, count
        
        if event.key == 'y':
            # If 'y', close all
            # If points have been clicked, imgp_confirmed is returned, else imgp
            # If objp is given, objp_confirmed is returned in addition
            if 'scat' not in globals() or 'imgp_confirmed' not in globals():
                imgp_confirmed = imgp
                objp_confirmed = objp
            else:
                imgp_confirmed = np.array([imgp.astype('float32') for imgp in imgp_confirmed])
                objp_confirmed = objp_confirmed
            # OpenCV needs at leas 4 correspondance points to calibrate
            if len(imgp_confirmed) < 6:
                objp_confirmed = []
                imgp_confirmed = []
            # close all, del all global variables except imgp_confirmed and objp_confirmed
            plt.close('all')
            if len(objp) == 0:
                if 'objp_confirmed' in globals():
                    del objp_confirmed

        if event.key == 'n' or event.key == 'q':
            # If 'n', close all and return nothing
            plt.close('all')
            imgp_confirmed = []
            objp_confirmed = []

        if event.key == 'c':
            # TODO: RIGHT NOW, IF 'C' IS PRESSED ANOTHER TIME, OBJP_CONFIRMED AND IMGP_CONFIRMED ARE RESET TO []
            # We should reopen a figure without point on it
            img_for_pointing = cv2.imread(old_image_path)
            if img_for_pointing is None:
                cap = cv2.VideoCapture(old_image_path)
                ret, img_for_pointing = cap.read()
                cap.release()
            img_for_pointing = cv2.cvtColor(img_for_pointing, cv2.COLOR_BGR2RGB)
            ax.imshow(img_for_pointing)
            # To update the image
            plt.draw()

            if 'objp_confirmed' in globals():
                del objp_confirmed
            # If 'c', allows retrieving imgp_confirmed by clicking them on the image
            scat = ax.scatter([],[],s=100,marker='+',color='g')
            plt.connect('button_press_event', on_click)
            # If objp is given, display 3D object points in black
            if len(objp) != 0 and not plt.fignum_exists(2):
                fig_3d = plt.figure()
                fig_3d.tight_layout()
                fig_3d.canvas.manager.set_window_title('Object points to be clicked')
                ax_3d = fig_3d.add_subplot(projection='3d')
                plt.rc('xtick', labelsize=5)
                plt.rc('ytick', labelsize=5)
                for i, (xs,ys,zs) in enumerate(np.float32(objp)):
                    ax_3d.scatter(xs,ys,zs, marker='.', color='k')
                    ax_3d.text(xs,ys,zs,  f'{str(i+1)}', size=10, zorder=1, color='k') 
                set_axes_equal(ax_3d)
                ax_3d.set_xlabel('X')
                ax_3d.set_ylabel('Y')
                ax_3d.set_zlabel('Z')
                if np.all(objp[:,2] == 0):
                    ax_3d.view_init(elev=-90, azim=0)
                fig_3d.show()

        if event.key == 'h':
            # If 'h', indicates that one of the objp is not visible on image
            # Displays it in red on 3D plot
            if len(objp) != 0  and 'ax_3d' in globals():
                count = [0 if 'count' not in globals() else count+1][0]
                if 'events' not in globals():
                    # retrieve first objp_confirmed_notok and plot 3D
                    events = [event]
                    objp_confirmed_notok = objp[count]
                    ax_3d.scatter(*objp_confirmed_notok, marker='o', color='r')
                    fig_3d.canvas.draw()
                elif count == len(objp)-1:
                    # if all objp have been clicked or indicated as not visible, close all
                    objp_confirmed = np.array([[objp[count]] if 'objp_confirmed' not in globals() else objp_confirmed+[objp[count]]][0])[:-1]
                    imgp_confirmed = np.array(np.expand_dims(scat.get_offsets(), axis=1), np.float32) 
                    plt.close('all')
                    for var_to_delete in ['events', 'count', 'scat', 'fig_3d', 'ax_3d', 'objp_confirmed_notok']:
                        if var_to_delete in globals():
                            del globals()[var_to_delete]
                else:
                    # retrieve other objp_confirmed_notok and plot 3D
                    events.append(event)
                    objp_confirmed_notok = objp[count]
                    ax_3d.scatter(*objp_confirmed_notok, marker='o', color='r')
                    fig_3d.canvas.draw()
            else:
                pass


    def on_click(event):
        '''
        Detect click position on image
        If right click, last point is removed
        '''
        
        global imgp_confirmed, objp_confirmed, objp_confirmed_notok, scat, ax_3d, fig_3d, events, count, xydata
        
        # Left click: Add clicked point to imgp_confirmed
        # Display it on image and on 3D plot
        if event.button == 1: 
            # To remember the event to cancel after right click
            if 'events' in globals():
                events.append(event)
            else:
                events = [event]

            # Add clicked point to image
            xydata = scat.get_offsets()
            new_xydata = np.concatenate((xydata,[[event.xdata,event.ydata]]))
            scat.set_offsets(new_xydata)
            imgp_confirmed = np.expand_dims(scat.get_offsets(), axis=1)    
            plt.draw()

            # Add clicked point to 3D object points if given
            if len(objp) != 0:
                count = [0 if 'count' not in globals() else count+1][0]
                if count==0:
                    # retrieve objp_confirmed and plot 3D
                    objp_confirmed = [objp[count]]
                    ax_3d.scatter(*objp[count], marker='o', color='g')
                    fig_3d.canvas.draw()
                elif count == len(objp)-1:
                    # close all
                    plt.close('all')
                    # retrieve objp_confirmed
                    objp_confirmed = np.array([[objp[count]] if 'objp_confirmed' not in globals() else objp_confirmed+[objp[count]]][0])
                    imgp_confirmed = np.array(imgp_confirmed, np.float32)
                    # delete all
                    for var_to_delete in ['events', 'count', 'scat', 'scat_3d', 'fig_3d', 'ax_3d', 'objp_confirmed_notok']:
                        if var_to_delete in globals():
                            del globals()[var_to_delete]
                else:
                    # retrieve objp_confirmed and plot 3D
                    objp_confirmed = [[objp[count]] if 'objp_confirmed' not in globals() else objp_confirmed+[objp[count]]][0]
                    ax_3d.scatter(*objp[count], marker='o', color='g')
                    fig_3d.canvas.draw()
                

        # Right click: 
        # If last event was left click, remove last point and if objp given, from objp_confirmed
        # If last event was 'H' and objp given, remove last point from objp_confirmed_notok
        elif event.button == 3: # right click
            if 'events' in globals():
                # If last event was left click: 
                if 'button' in dir(events[-1]):
                    if events[-1].button == 1: 
                        # Remove lastpoint from image
                        new_xydata = scat.get_offsets()[:-1]
                        scat.set_offsets(new_xydata)
                        plt.draw()
                        # Remove last point from imgp_confirmed
                        imgp_confirmed = imgp_confirmed[:-1]
                        if len(objp) != 0:
                            if count >= 0: 
                                count -= 1
                            # Remove last point from objp_confirmed
                            objp_confirmed = objp_confirmed[:-1]
                            # remove from plot 
                            if len(ax_3d.collections) > len(objp):
                                ax_3d.collections[-1].remove()
                                fig_3d.canvas.draw()
                            
                # If last event was 'h' key
                elif events[-1].key == 'h':
                    if len(objp) != 0:
                        if count >= 1: count -= 1
                        # Remove last point from objp_confirmed_notok
                        objp_confirmed_notok = objp_confirmed_notok[:-1]
                        # remove from plot  
                        if len(ax_3d.collections) > len(objp):
                            ax_3d.collections[-1].remove()
                            fig_3d.canvas.draw()                
    

    def set_axes_equal(ax):
        '''
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc.
        From https://stackoverflow.com/questions/13685386/how-to-set-the-equal-aspect-ratio-for-all-axes-x-y-z

        Input
        ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    # Write instructions
    cv2.putText(img, 'Type "Y" to accept point detection.', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, 'Type "Y" to accept point detection.', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, 'If points are wrongfully (or not) detected:', (20, 43), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, 'If points are wrongfully (or not) detected:', (20, 43), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, '- type "N" to dismiss this image,', (20, 66), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, '- type "N" to dismiss this image,', (20, 66), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, '- type "C" to click points by hand (beware of their order).', (20, 89), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, '- type "C" to click points by hand (beware of their order).', (20, 89), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, '   left click to add a point, right click to remove it, "H" to indicate it is not visible. ', (20, 112), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, '   left click to add a point, right click to remove it, "H" to indicate it is not visible. ', (20, 112), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, '   Confirm with "Y", cancel with "N".', (20, 135), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, '   Confirm with "Y", cancel with "N".', (20, 135), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, 'Use mouse wheel to zoom in and out and to pan', (20, 158), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, 'Use mouse wheel to zoom in and out and to pan', (20, 158), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    
    # Put image in a matplotlib figure for more controls
    plt.rcParams['toolbar'] = 'None'
    fig, ax = plt.subplots()
    fig = plt.gcf()
    fig.canvas.manager.set_window_title(os.path.basename(img_path))
    ax.axis("off")
    for corner in imgp:
        x, y = corner.ravel()
        cv2.drawMarker(img, (int(x),int(y)), (128,128,128), cv2.MARKER_CROSS, 10, 2)
    ax.imshow(img)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.tight_layout()
    
    # Allow for zoom and pan in image
    zoom_factory(ax)
    ph = panhandler(fig, button=2)

    # Handles key presses to Accept, dismiss, or click points by hand
    cid = fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.draw()
    plt.show(block=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.rcParams['toolbar'] = 'toolmanager'

    for var_to_delete in ['events', 'count', 'scat', 'fig_3d', 'ax_3d', 'objp_confirmed_notok']:
        if var_to_delete in globals():
            del globals()[var_to_delete]

    if 'imgp_confirmed' in globals() and 'objp_confirmed' in globals():
        return imgp_confirmed, objp_confirmed
    elif 'imgp_confirmed' in globals() and not 'objp_confirmed' in globals():
        return imgp_confirmed
    else:
        return


def extract_frames(video_path, extract_every_N_sec=1, overwrite_extraction=False):
    '''
    Extract frames from video 
    if has not been done yet or if overwrite==True
    
    INPUT:
    - video_path: path to video whose frames need to be extracted
    - extract_every_N_sec: extract one frame every N seconds (can be <1)
    - overwrite_extraction: if True, overwrite even if frames have already been extracted
    
    OUTPUT:
    - extracted frames in folder
    '''
    
    if not os.path.exists(os.path.splitext(video_path)[0] + '_00000.png') or overwrite_extraction:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = round(cap.get(cv2.CAP_PROP_FPS))
            frame_nb = 0
            logging.info(f'Extracting frames...')
            while cap.isOpened():
                ret, frame = cap.read()
                if ret == True:
                    if frame_nb % (fps*extract_every_N_sec) == 0:
                        img_path = (os.path.splitext(video_path)[0] + '_' +str(frame_nb).zfill(5)+'.png')
                        cv2.imwrite(str(img_path), frame)
                    frame_nb+=1
                else:
                    break


def trc_write(object_coords_3d, trc_path):
    '''
    Make Opensim compatible trc file from a dataframe with 3D coordinates

    INPUT:
    - object_coords_3d: list of 3D point lists
    - trc_path: output path of the trc file

    OUTPUT:
    - trc file with 2 frames of the same 3D points
    '''

    #Header
    DataRate = CameraRate = OrigDataRate = 1
    NumFrames = 2
    NumMarkers = len(object_coords_3d)
    keypoints_names = np.arange(NumMarkers)
    header_trc = ['PathFileType\t4\t(X/Y/Z)\t' + os.path.basename(trc_path), 
            'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames', 
            '\t'.join(map(str,[DataRate, CameraRate, NumFrames, NumMarkers, 'm', OrigDataRate, NumFrames])),
            'Frame#\tTime\t' + '\t\t\t'.join(str(k) for k in keypoints_names) + '\t\t',
            '\t\t'+'\t'.join([f'X{i+1}\tY{i+1}\tZ{i+1}' for i in range(NumMarkers)])]
    
    # Zup to Yup coordinate system
    object_coords_3d = pd.DataFrame([np.array(object_coords_3d).flatten(), np.array(object_coords_3d).flatten()])
    object_coords_3d = zup2yup(object_coords_3d)
    
    #Add Frame# and Time columns
    object_coords_3d.index = np.array(range(0, NumFrames)) + 1
    object_coords_3d.insert(0, 't', object_coords_3d.index / DataRate)

    #Write file
    with open(trc_path, 'w') as trc_o:
        [trc_o.write(line+'\n') for line in header_trc]
        object_coords_3d.to_csv(trc_o, sep='\t', index=True, header=None, lineterminator='\n')

    return trc_path


def toml_write(calib_path, C, S, D, K, R, T):
    '''
    Writes calibration parameters to a .toml file

    INPUTS:
    - calib_path: path to the output calibration file: string
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats (Rodrigues)
    - T: extrinsic translation: list of arrays of floats

    OUTPUTS:
    - a .toml file cameras calibrations
    '''

    with open(os.path.join(calib_path), 'w+') as cal_f:
        for c in range(len(C)):
            cam=f'[{C[c]}]\n'
            name = f'name = "{C[c]}"\n'
            size = f'size = [ {S[c][0]}, {S[c][1]}]\n' 
            mat = f'matrix = [ [ {K[c][0,0]}, 0.0, {K[c][0,2]}], [ 0.0, {K[c][1,1]}, {K[c][1,2]}], [ 0.0, 0.0, 1.0]]\n'
            dist = f'distortions = [ {D[c][0]}, {D[c][1]}, {D[c][2]}, {D[c][3]}]\n' 
            rot = f'rotation = [ {R[c][0]}, {R[c][1]}, {R[c][2]}]\n'
            tran = f'translation = [ {T[c][0]}, {T[c][1]}, {T[c][2]}]\n'
            fish = f'fisheye = false\n\n'
            cal_f.write(cam + name + size + mat + dist + rot + tran + fish)
        meta = '[metadata]\nadjusted = false\nerror = 0.0\n'
        cal_f.write(meta)


def recap_calibrate(ret, calib_path, calib_full_type):
    '''
    Print a log message giving calibration results. Also stored in User/logs.txt.

    OUTPUT:
    - Message in console
    '''
    
    calib = toml.load(calib_path)
    
    ret_m, ret_px = [], []
    for c, cam in enumerate(calib.keys()):
        if cam != 'metadata':
            f_px = calib[cam]['matrix'][0][0]
            Dm = euclidean_distance(calib[cam]['translation'], [0,0,0])
            if calib_full_type in ['convert_qualisys', 'convert_vicon','convert_opencap', 'convert_biocv']:
                ret_m.append( np.around(ret[c], decimals=3) )
                ret_px.append( np.around(ret[c] / (Dm*1000) * f_px, decimals=3) )
            elif calib_full_type=='calculate':
                ret_px.append( np.around(ret[c], decimals=3) )
                ret_m.append( np.around(ret[c]*Dm*1000 / f_px, decimals=3) )
                
    logging.info(f'\n--> Residual (RMS) calibration errors for each camera are respectively {ret_px} px, \nwhich corresponds to {ret_m} mm.\n')
    logging.info(f'Calibration file is stored at {calib_path}.')


def calibrate_cams_all(config_dict):
    '''
    Either converts a preexisting calibration file, 
    or calculates calibration from scratch (from a board or from points).
    Stores calibration in a .toml file
    Prints recap.
    
    INPUTS:
    - a config_dict dictionary

    OUTPUT:
    - a .toml camera calibration file
    '''

    # Read config_dict
    project_dir = config_dict.get('project').get('project_dir')
    calib_dir = [os.path.join(project_dir, c) for c in os.listdir(project_dir) if ('Calib' in c or 'calib' in c)][0]
    calib_type = config_dict.get('calibration').get('calibration_type')

    if calib_type=='convert':
        convert_filetype = config_dict.get('calibration').get('convert').get('convert_from')
        try:
            if convert_filetype=='qualisys':
                convert_ext = '.qca.txt'
                file_to_convert_path = glob.glob(os.path.join(calib_dir, f'*{convert_ext}*'))[0]
                binning_factor = config_dict.get('calibration').get('convert').get('qualisys').get('binning_factor')
            elif convert_filetype=='optitrack':
                file_to_convert_path = ['']
                binning_factor = 1
            elif convert_filetype=='vicon':
                convert_ext = '.xcp'
                file_to_convert_path = glob.glob(os.path.join(calib_dir, f'*{convert_ext}'))[0]
                binning_factor = 1
            elif convert_filetype=='opencap': # all files with .pickle extension
                convert_ext = '.pickle'
                file_to_convert_path = sorted(glob.glob(os.path.join(calib_dir, f'*{convert_ext}')))
                binning_factor = 1
            elif convert_filetype=='easymocap': #intri.yml and intri.yml
                convert_ext = '.yml'
                file_to_convert_path = sorted(glob.glob(os.path.join(calib_dir, f'*{convert_ext}')))
                binning_factor = 1
            elif convert_filetype=='biocv': # all files without extension -> now with .calib extension
                # convert_ext = 'no'
                # list_dir = os.listdir(calib_dir)
                # list_dir_noext = sorted([os.path.splitext(f)[0] for f in list_dir if os.path.splitext(f)[1]==''])
                # file_to_convert_path = [os.path.join(calib_dir,f) for f in list_dir_noext if os.path.isfile(os.path.join(calib_dir, f))]
                convert_ext = '.calib'
                file_to_convert_path = sorted(glob.glob(os.path.join(calib_dir, f'*{convert_ext}')))
                binning_factor = 1
            elif convert_filetype=='anipose' or convert_filetype=='freemocap' or convert_filetype=='caliscope': # no conversion needed, skips this stage
                logging.info(f'\n--> No conversion needed from Caliscope, AniPose, nor from FreeMocap. Calibration skipped.\n')
                return
            else:
                convert_ext = '???'
                file_to_convert_path = ['']
                raise NameError(f'Calibration conversion from {convert_filetype} is not supported.') from None
            assert file_to_convert_path!=[]
        except:
            raise NameError(f'No file with {convert_ext} extension found in {calib_dir}.')
        
        calib_output_path = os.path.join(calib_dir, f'Calib_{convert_filetype}.toml')
        calib_full_type = '_'.join([calib_type, convert_filetype])
        args_calib_fun = [file_to_convert_path, binning_factor]
        
    elif calib_type=='calculate':
        intrinsics_config_dict = config_dict.get('calibration').get('calculate').get('intrinsics')
        extrinsics_config_dict = config_dict.get('calibration').get('calculate').get('extrinsics')
        extrinsics_method = config_dict.get('calibration').get('calculate').get('extrinsics').get('extrinsics_method')

        calib_output_path = os.path.join(calib_dir, f'Calib_{extrinsics_method}.toml')
        calib_full_type = calib_type
        args_calib_fun = [calib_dir, intrinsics_config_dict, extrinsics_config_dict]

    else:
        logging.info('Wrong calibration_type in Config.toml')
    
    # Map calib function
    calib_mapping = {
        'convert_qualisys': calib_qca_fun,
        'convert_optitrack': calib_optitrack_fun,
        'convert_vicon': calib_vicon_fun,
        'convert_opencap': calib_opencap_fun,
        'convert_easymocap': calib_easymocap_fun,
        'convert_biocv': calib_biocv_fun,
        'calculate': calib_calc_fun,
        }
    calib_fun = calib_mapping[calib_full_type]

    # Calibrate
    ret, C, S, D, K, R, T = calib_fun(*args_calib_fun)

    # Write calibration file
    toml_write(calib_output_path, C, S, D, K, R, T)
    
    # Recap message
    recap_calibrate(ret, calib_output_path, calib_full_type)
