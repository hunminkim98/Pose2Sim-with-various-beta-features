#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
#########################################
## SYNCHRONIZE CAMERAS                 ##
#########################################

    Post-synchronize your cameras in case they are not natively synchronized.

    For each camera, computes mean vertical speed for the chosen keypoints, 
    and find the time offset for which their correlation is highest. 

    Depending on the analysed motion, all keypoints can be taken into account, 
    or a list of them, or the right or left side.
    All frames can be considered, or only those around a specific time (typically, 
    the time when there is a single participant in the scene performing a clear vertical motion).
    Has also been successfully tested for synchronizing random walkswith random walks.

    Keypoints whose likelihood is too low are filtered out; and the remaining ones are 
    filtered with a butterworth filter.

    INPUTS: 
    - json files from each camera folders
    - a Config.toml file
    - a skeleton model

    OUTPUTS: 
    - synchronized json files for each camera
'''

## INIT
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import patheffects
from scipy import signal
from scipy import interpolate
import json
import os
import glob
import fnmatch
import re
import shutil
from anytree import RenderTree
from anytree.importer import DictImporter
from matplotlib.widgets import TextBox, Button
import logging

from Pose2Sim.common import sort_stringlist_by_last_number, bounding_boxes
from Pose2Sim.skeletons import *


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon, HunMin Kim"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.9.4"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


# FUNCTIONS
def load_frame_and_bounding_boxes(cap, frame_number, frame_to_json, pose_dir, json_dir_name):
    '''
    Given a video capture object or a list of image files and a frame number, 
    load the frame (or image) and corresponding bounding boxes.

    INPUTS:
    - cap: cv2.VideoCapture object or list of image file paths.
    - frame_number: int. The frame number to load.
    - frame_to_json: dict. Mapping from frame numbers to JSON file names.
    - pose_dir: str. Path to the directory containing pose data.
    - json_dir_name: str. Name of the JSON directory for the current camera.

    OUTPUTS:
    - frame_rgb: The RGB image of the frame or image.
    - bounding_boxes_list: List of bounding boxes for the frame/image.
    '''

    # Case 1: If input is a video file (cv2.VideoCapture object)
    if isinstance(cap, cv2.VideoCapture):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            return None, []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Case 2: If input is a list of image file paths
    elif isinstance(cap, list):
        if frame_number >= len(cap):
            return None, []
        image_path = cap[frame_number]
        frame = cv2.imread(image_path)
        if frame is None:
            return None, []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    else:
        raise ValueError("Input must be either a video capture object or a list of image file paths.")

    # Get the corresponding JSON file for bounding boxes
    json_file_name = frame_to_json.get(frame_number)
    bounding_boxes_list = []
    if json_file_name:
        json_file_path = os.path.join(pose_dir, json_dir_name, json_file_name)
        bounding_boxes_list.extend(bounding_boxes(json_file_path))

    return frame_rgb, bounding_boxes_list


def draw_bounding_boxes_and_annotations(ax, bounding_boxes_list, rects, annotations):
    '''
    Draws the bounding boxes and annotations on the given axes.

    INPUTS:
    - ax: The axes object to draw on.
    - bounding_boxes_list: list of tuples. Each tuple contains (x_min, y_min, x_max, y_max) of a bounding box.
    - rects: List to store rectangle patches representing bounding boxes.
    - annotations: List to store text annotations for each bounding box.

    OUTPUTS:
    - None. Modifies rects and annotations in place.
    '''

    # Clear existing rectangles and annotations
    for items in [rects, annotations]:
            for item in items:
                item.remove()
            items.clear()

    # Draw bounding boxes and annotations
    for idx, (x_min, y_min, x_max, y_max) in enumerate(bounding_boxes_list):
        if not np.isfinite([x_min, y_min, x_max, y_max]).all():
            continue  # Skip invalid bounding boxes for solve issue(posx and posy should be finite values)

        rect = plt.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=1, edgecolor='white', facecolor=(1, 1, 1, 0.1),
            linestyle='-', path_effects=[patheffects.withSimplePatchShadow()], zorder=2
        ) # add shadow
        ax.add_patch(rect)
        rects.append(rect)

        annotation = ax.text(
            x_min, y_min - 10, f'Person {idx}', color='white', fontsize=7, fontweight='normal',
            bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'), zorder=3
        )
        annotations.append(annotation)


def reset_styles(rects, annotations):
    '''
    Resets the styles of the rectangles and annotations to default.

    INPUTS:
    - rects: List of rectangle patches representing bounding boxes.
    - annotations: List of text annotations for each bounding box.

    OUTPUTS:
    - None. Modifies rects and annotations in place.
    '''

    for rect, annotation in zip(rects, annotations):
        rect.set_linewidth(1)
        rect.set_edgecolor('white')
        rect.set_facecolor((1, 1, 1, 0.1))
        annotation.set_fontsize(7)
        annotation.set_fontweight('normal')


def highlight_bounding_box(rect, annotation):
    '''
    Highlights a rectangle and its annotation.

    INPUTS:
    - rect: Rectangle patch to highlight.
    - annotation: Text annotation to highlight.

    OUTPUTS:
    - None. Modifies rect and annotation in place.
    '''

    rect.set_linewidth(2)
    rect.set_edgecolor('yellow')
    rect.set_facecolor((1, 1, 0, 0.2))
    annotation.set_fontsize(8)
    annotation.set_fontweight('bold')


def on_hover(event, fig, rects, annotations, bounding_boxes_list):
    '''
    Highlights the bounding box and annotation when the mouse hovers over a person in the plot.
    
    INPUTS:
    - event: The hover event.
    - fig:  The figure object.
    - rects: The rectangles representing bounding boxes.
    - annotations: The annotations corresponding to each bounding box.
    - bounding_boxes_list: List of tuples containing bounding box coordinates.

    OUTPUTS:
    - None. This function updates the plot in place.
    '''

    if event.xdata is None or event.ydata is None:
        return

    # Reset styles of all rectangles and annotations
    reset_styles(rects, annotations)

    # Find and highlight the bounding box under the mouse cursor
    # remove NaN bounding boxes for make sure matching with rects
    bounding_boxes_list = [bbox for bbox in bounding_boxes_list if np.all(np.isfinite(bbox)) and not np.any(np.isnan(bbox))]
    for idx, (x_min, y_min, x_max, y_max) in enumerate(bounding_boxes_list):
        if x_min <= event.xdata <= x_max and y_min <= event.ydata <= y_max:
            highlight_bounding_box(rects[idx], annotations[idx])
            break

    fig.canvas.draw_idle()


def on_click(event, ax, bounding_boxes_list, selected_idx_container, person_textbox):
    '''
    Detects if a bounding box is clicked and records the index of the selected person.

    INPUTS:
    - event: The click event.
    - ax: The axes object of the plot.
    - bounding_boxes_list: List of tuples containing bounding box coordinates.
    - selected_idx_container: List with one element to store the selected person's index.
    - person_textbox: TextBox. The person selection text box widget.

    OUTPUTS:
    - None. Updates selected_idx_container[0] and person_textbox with the index of the selected person.
    '''

    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return

    for idx, (x_min, y_min, x_max, y_max) in enumerate(bounding_boxes_list):
        if x_min <= event.xdata <= x_max and y_min <= event.ydata <= y_max:
            selected_idx_container[0] = idx
            person_textbox.set_val(str(idx))  # Update the person number text box
            break


def update_play(cap, image, frame_number, frame_to_json, pose_dir, json_dir_name, rects, annotations, bounding_boxes_list, ax, fig):
    '''
    Updates the plot with a new frame.

    INPUTS:
    - cap: cv2.VideoCapture. The video capture object.
    - image: The image object in the plot.
    - frame_number: int. The frame number to display.
    - frame_to_json: dict. Mapping from frame numbers to JSON file names.
    - pose_dir: str. Path to the directory containing pose data.
    - json_dir_name: str. Name of the JSON directory for the current camera.
    - rects: List of rectangle patches representing bounding boxes.
    - annotations: List of text annotations for each bounding box.
    - bounding_boxes_list: List of tuples to store bounding boxes for the current frame.
    - ax: The axes object of the plot.
    - fig: The figure object containing the plot.

    OUTPUTS:
    - None. Updates the plot with the new frame, bounding boxes, and annotations.
    '''

    frame_rgb, bounding_boxes_list_new = load_frame_and_bounding_boxes(cap, frame_number, frame_to_json, pose_dir, json_dir_name)
    if frame_rgb is None:
        return

    # Update frame image
    image.set_data(frame_rgb)

    # Update bounding boxes
    bounding_boxes_list.clear()
    bounding_boxes_list.extend(bounding_boxes_list_new)

    # Draw bounding boxes and annotations
    draw_bounding_boxes_and_annotations(ax, bounding_boxes_list, rects, annotations)
    fig.canvas.draw_idle()


def handle_ok_button():
    '''
    Handle OK button click. Closes the window and confirms the selection.
    '''
    plt.close()


def handle_toggle_labels(event, keypoint_texts, show_labels):
    '''
    Handle toggle labels button click.
    
    INPUTS:
    - event: The button click event.
    - keypoint_texts: List of text objects showing keypoint labels.
    - show_labels: List containing boolean flag for label visibility.
    '''
    show_labels[0] = not show_labels[0]  # Toggle visibility state
    for text in keypoint_texts:
        text.set_visible(show_labels[0])
    plt.draw()


def select_person(vid_or_img_files, cam_names, json_files_names_range, search_around_frames, pose_dir, json_dirs_names, keypoints_names):
    '''
    Allows the user to select a person, start frame and keypoints from each camera by clicking on their bounding box in the video frames.

    INPUTS:
    - vid_or_img_files: list of str. Paths to the video files for each camera or to the image directories for each camera.
    - cam_names: list of str. Names of the cameras.
    - json_files_names_range: list of lists. Each sublist contains JSON file names for a camera.
    - search_around_frames: list of tuples. Each tuple contains (start_frame, end_frame) for searching frames.
    - pose_dir: str. Path to the directory containing pose data.
    - json_dirs_names: list of str. Names of the JSON directories for each camera.
    - keypoints_names: list of str. List of keypoint names from the skeleton model.

    OUTPUTS:
    - selected_id_list: list of int or None. List of the selected person indices for each camera.
    - keypoints_to_consider: list of str. List of keypoint names to consider for synchronization.
    - approx_time_maxspeed: list of int. List of frame numbers for synchronization.
    '''

    logging.info('Manual mode: selecting the person, start frame and keypoints to synchronize on for each camera.')
    selected_id_list = []
    approx_time_maxspeed = []
    keypoints_to_consider = []

    try: # video files
        video_files_dict = {cam_name: file for cam_name in cam_names for file in vid_or_img_files if cam_name in os.path.basename(file)}
    except: # image directories
        video_files_dict = {cam_name: files for cam_name in cam_names for files in vid_or_img_files if cam_name in os.path.basename(files[0])}

    for i, cam_name in enumerate(cam_names):
        # Initialize containers for this camera
        selected_idx_container = [0]  # Container for selected person index
        keypoints_to_consider_container = ['R Wrist']  # Container for selected keypoints
        
        vid_or_img_files_cam = video_files_dict.get(cam_name)
        if not vid_or_img_files_cam:
            logging.warning(f'No video file nor image directory found for camera {cam_name}')
            selected_id_list.append(None)
            continue
        try:
            cap = cv2.VideoCapture(vid_or_img_files_cam)
            if not cap.isOpened():
                raise
        except:
            cap = vid_or_img_files_cam

        frame_to_json = {int(re.split(r'(\d+)', name)[-2]): name for name in json_files_names_range[i]}
        frame_number = search_around_frames[i][0]

        frame_rgb, bounding_boxes_list = load_frame_and_bounding_boxes(cap, frame_number, frame_to_json, pose_dir, json_dirs_names[i])
        if frame_rgb is None:
            logging.warning(f'Cannot read frame {frame_number} from video {vid_or_img_files_cam}')
            selected_id_list.append(None)
            if isinstance(cap, cv2.VideoCapture):
                cap.release()
            continue
        
        # Initialize plot with a larger figure and two subplots
        frame_height, _ = frame_rgb.shape[:2]
        fig_height = frame_height/250
        fig = plt.figure(figsize=(12, fig_height))  # 더 넓은 figure
        
        # Main video display (left side)
        ax_video = plt.axes([0.05, 0.2, 0.6, 0.7])  # 비디오 영역 조정
        ax_video.imshow(frame_rgb)
        ax_video.set_title(f'Camera name: {cam_name}', fontsize=10, pad=10)
        ax_video.axis('off')

        # Keypoints selection area (right side)
        ax_keypoints = plt.axes([0.7, 0.2, 0.25, 0.7])
        
        # Define relative positions for keypoints in human form
        keypoints_positions = {
            # Head
            'Head': (0.5, 0.8),
            'Neck': (0.5, 0.7),
            'Nose': (0.5, 0.75),           
            # Torso
            'Hip': (0.5, 0.3),
            'RHip': (0.4, 0.3),
            'LHip': (0.6, 0.3),          
            # Right side
            'RShoulder': (0.4, 0.7),
            'RElbow': (0.3, 0.6),
            'RWrist': (0.2, 0.5),
            'RKnee': (0.35, 0.2),
            'RAnkle': (0.35, 0.05),
            'RSmallToe': (0.3, 0.0),
            'RBigToe': (0.4, 0.0),
            'RHeel': (0.35, 0.02),
            
            # Left side
            'LShoulder': (0.6, 0.7),
            'LElbow': (0.7, 0.6),
            'LWrist': (0.8, 0.5),
            'LKnee': (0.65, 0.2),
            'LAnkle': (0.65, 0.05),
            'LSmallToe': (0.7, 0.0),
            'LBigToe': (0.6, 0.0),
            'LHeel': (0.65, 0.02)
        }

        # Create x, y coordinates
        keypoints_x = []
        keypoints_y = []
        for name in keypoints_names:
            pos = keypoints_positions.get(name, (0.5, 0.5))
            keypoints_x.append(pos[0])
            keypoints_y.append(pos[1])

        selected_keypoints = []
        show_labels = [True]  # Container for label visibility state
        keypoint_texts = []  # Store text objects

        # Plot keypoints as scatter points
        scatter = ax_keypoints.scatter(keypoints_x, keypoints_y, c='blue', picker=True)
        
        # Add keypoint labels
        for x, y, name in zip(keypoints_x, keypoints_y, keypoints_names):
            text = ax_keypoints.text(x + 0.02, y, name, va='center', fontsize=8)
            keypoint_texts.append(text)

        # Add toggle labels button
        ax_toggle = plt.axes([0.775, 0.2, 0.1, 0.04])  # 키포인트 영역 아래에 버튼 배치
        btn_toggle = Button(ax_toggle, 'Toggle Labels')
        btn_toggle.on_clicked(lambda event: handle_toggle_labels(event, keypoint_texts, show_labels))

        ax_keypoints.set_title('Click keypoints to select', pad=10)
        ax_keypoints.set_xlim(0, 1)
        ax_keypoints.set_ylim(-0.1, 1)
        ax_keypoints.axis('off')

        def on_pick(event):
            ind = event.ind[0]
            keypoint = keypoints_names[ind]
            if keypoint in selected_keypoints:
                selected_keypoints.remove(keypoint)
                scatter.set_facecolors([('blue' if n not in selected_keypoints else 'red') 
                                      for n in keypoints_names])
            else:
                selected_keypoints.append(keypoint)
                scatter.set_facecolors([('red' if n in selected_keypoints else 'blue') 
                                      for n in keypoints_names])
            keypoints_to_consider_container[0] = selected_keypoints
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('pick_event', on_pick)

        # Controls at the bottom
        ax_person = plt.axes([0.3, 0.1, 0.1, 0.04])
        person_textbox = TextBox(ax_person, 'Synchronize on person number', initial='0')
        
        # Frame number text box
        ax_frame = plt.axes([0.45, 0.1, 0.1, 0.04])
        frame_textbox = TextBox(ax_frame, 'Frame', initial=str(frame_number))

        # Draw bounding boxes and annotations
        rects, annotations = [], []
        draw_bounding_boxes_and_annotations(ax_video, bounding_boxes_list, rects, annotations)

        # Add hover event handler
        fig.canvas.mpl_connect('motion_notify_event', 
            lambda event: on_hover(event, fig, rects, annotations, bounding_boxes_list))

        # Add click event handler
        fig.canvas.mpl_connect('button_press_event', 
            lambda event: on_click(event, ax_video, bounding_boxes_list, selected_idx_container, person_textbox))

        # Event handlers connection
        person_textbox.on_submit(lambda text: handle_person_change(text, selected_idx_container, person_textbox))
        frame_textbox.on_submit(lambda text: handle_frame_change(text, frame_number, frame_textbox, cap, ax_video, 
                               frame_to_json, pose_dir, json_dirs_names[i], rects, annotations, 
                               bounding_boxes_list, fig, search_around_frames, i))

        # Navigation buttons and OK button
        btn_prev = plt.axes([0.565, 0.1, 0.02, 0.04])
        btn_next = plt.axes([0.59, 0.1, 0.02, 0.04])
        btn_ok = plt.axes([0.615, 0.1, 0.03, 0.04])
        
        btn_prev = plt.Button(btn_prev, '<')
        btn_next = plt.Button(btn_next, '>')
        btn_ok = plt.Button(btn_ok, 'OK')

        btn_prev.on_clicked(lambda event: handle_prev_frame(frame_textbox, search_around_frames, i, cap, ax_video,
                           frame_to_json, pose_dir, json_dirs_names[i], rects, annotations,
                           bounding_boxes_list, fig))
        btn_next.on_clicked(lambda event: handle_next_frame(frame_textbox, search_around_frames, i, cap, ax_video,
                           frame_to_json, pose_dir, json_dirs_names[i], rects, annotations,
                           bounding_boxes_list, fig))
        btn_ok.on_clicked(lambda event: handle_ok_button())

        # Keyboard navigation
        fig.canvas.mpl_connect('key_press_event', lambda event: handle_key_press(event, frame_textbox,
                              search_around_frames, i, cap, ax_video, frame_to_json, pose_dir,
                              json_dirs_names[i], rects, annotations, bounding_boxes_list, fig))

        # Show plot and wait for user input (window will close only when OK is clicked)
        plt.show()
        cap.release()

        # Store selected values after OK button is clicked
        selected_id_list.append(selected_idx_container[0])
        keypoints_to_consider.append(keypoints_to_consider_container[0])
        approx_time_maxspeed.append(int(frame_textbox.text))
        
        logging.info(f'--> Camera #{i}: selected person #{selected_idx_container[0]} '
                    f'with keypoints {keypoints_to_consider_container[0]} '
                    f'at frame #{frame_textbox.text}')

    return selected_id_list, keypoints_to_consider, approx_time_maxspeed

def convert_json2pandas(json_files, likelihood_threshold=0.6, keypoints_ids=[], multi_person=False, selected_id=None):
    '''
    Convert a list of JSON files to a pandas DataFrame.
    Only takes one person in the JSON file.

    INPUTS:
    - json_files: list of str. Paths of the the JSON files.
    - likelihood_threshold: float. Drop values if confidence is below likelihood_threshold.
    - keypoints_ids: list of int. Indices of the keypoints to extract.

    OUTPUTS:
    - df_json_coords: dataframe. Extracted coordinates in a pandas dataframe.
    '''

    nb_coords = len(keypoints_ids)
    json_coords = []
    for j_p in json_files:
        with open(j_p) as j_f:
            try:
                json_data_all = json.load(j_f)['people']

                # # previous approach takes person #0
                # json_data = json_data_all[0]
                # json_data = np.array([json_data['pose_keypoints_2d'][3*i:3*i+3] for i in keypoints_ids])
                
                # # approach based on largest mean confidence does not work if person in background is better detected
                # p_conf = [np.mean(np.array([p['pose_keypoints_2d'][3*i:3*i+3] for i in keypoints_ids])[:, 2])
                #         if 'pose_keypoints_2d' in p else 0
                #         for p in json_data_all]
                # max_confidence_person = json_data_all[np.argmax(p_conf)]
                # json_data = np.array([max_confidence_person['pose_keypoints_2d'][3*i:3*i+3] for i in keypoints_ids])

                # latest approach: uses person with largest bounding box
                if not multi_person:
                    bbox_area = [
                                (keypoints[:, 0].max() - keypoints[:, 0].min()) * (keypoints[:, 1].max() - keypoints[:, 1].min())
                                if 'pose_keypoints_2d' in p else 0
                                for p in json_data_all
                                for keypoints in [np.array([p['pose_keypoints_2d'][3*i:3*i+3] for i in keypoints_ids])]
                                ]
                    max_area_person = json_data_all[np.argmax(bbox_area)]
                    json_data = np.array([max_area_person['pose_keypoints_2d'][3*i:3*i+3] for i in keypoints_ids])

                elif multi_person:
                    if selected_id is not None: # We can sfely assume that selected_id is always not greater than len(json_data_all) because padding with 0 was done in the previous step
                        selected_person = json_data_all[selected_id]
                        json_data = np.array([selected_person['pose_keypoints_2d'][3*i:3*i+3] for i in keypoints_ids])
                    else:
                        json_data = [np.nan] * nb_coords * 3
                
                # Remove points with low confidence
                json_data = np.array([j if j[2]>likelihood_threshold else [np.nan, np.nan, np.nan] for j in json_data]).ravel().tolist() 
            except:
                # print(f'No person found in {os.path.basename(json_dir)}, frame {i}')
                json_data = [np.nan] * nb_coords*3
        json_coords.append(json_data)
    df_json_coords = pd.DataFrame(json_coords)

    return df_json_coords


def drop_col(df, col_nb):
    '''
    Drops every nth column from a DataFrame.

    INPUTS:
    - df: dataframe. The DataFrame from which columns will be dropped.
    - col_nb: int. The column number to drop.

    OUTPUTS:
    - dataframe: DataFrame with dropped columns.
    '''

    idx_col = list(range(col_nb-1, df.shape[1], col_nb)) 
    df_dropped = df.drop(idx_col, axis=1)
    df_dropped.columns = range(df_dropped.columns.size)
    return df_dropped


def vert_speed(df, axis='y'):
    '''
    Calculate the vertical speed of a DataFrame along a specified axis.

    INPUTS:
    - df: dataframe. DataFrame of 2D coordinates.
    - axis: str. The axis along which to calculate speed. 'x', 'y', or 'z', default is 'y'.

    OUTPUTS:
    - df_vert_speed: DataFrame of vertical speed values.
    '''

    axis_dict = {'x':0, 'y':1, 'z':2}
    df_diff = df.diff()
    df_diff = df_diff.fillna(df_diff.iloc[1]*2)
    df_vert_speed = pd.DataFrame([df_diff.loc[:, 2*k + axis_dict[axis]] for k in range(int(df_diff.shape[1] / 2))]).T # modified ( df_diff.shape[1]*2 to df_diff.shape[1] / 2 )
    df_vert_speed.columns = np.arange(len(df_vert_speed.columns))
    return df_vert_speed


def interpolate_zeros_nans(col, kind):
    '''
    Interpolate missing points (of value nan)

    INPUTS:
    - col: pandas column of coordinates
    - kind: 'linear', 'slinear', 'quadratic', 'cubic'. Default 'cubic'

    OUTPUTS:
    - col_interp: interpolated pandas column
    '''
    
    mask = ~(np.isnan(col) | col.eq(0)) # true where nans or zeros
    idx_good = np.where(mask)[0]
    try: 
        f_interp = interpolate.interp1d(idx_good, col[idx_good], kind=kind, bounds_error=False)
        col_interp = np.where(mask, col, f_interp(col.index))
        return col_interp 
    except:
        # print('No good values to interpolate')
        return col


def time_lagged_cross_corr(camx, camy, lag_range, show=True, ref_cam_name='0', cam_name='1'):
    '''
    Compute the time-lagged cross-correlation between two pandas series.

    INPUTS:
    - camx: pandas series. Coordinates of reference camera.
    - camy: pandas series. Coordinates of camera to compare.
    - lag_range: int or list. Range of frames for which to compute cross-correlation.
    - show: bool. If True, display the cross-correlation plot.
    - ref_cam_name: str. The name of the reference camera.
    - cam_name: str. The name of the camera to compare with.

    OUTPUTS:
    - offset: int. The time offset for which the correlation is highest.
    - max_corr: float. The maximum correlation value.
    '''

    if isinstance(lag_range, int):
        lag_range = [-lag_range, lag_range]

    pearson_r = [camx.corr(camy.shift(lag)) for lag in range(lag_range[0], lag_range[1])]
    offset = int(np.floor(len(pearson_r)/2)-np.argmax(pearson_r))
    if not np.isnan(pearson_r).all():
        max_corr = np.nanmax(pearson_r)

        if show:
            f, ax = plt.subplots(2,1)
            # speed
            camx.plot(ax=ax[0], label = f'Reference: {ref_cam_name}')
            camy.plot(ax=ax[0], label = f'Compared: {cam_name}')
            ax[0].set(xlabel='Frame', ylabel='Speed (px/frame)')
            ax[0].legend()
            # time lagged cross-correlation
            ax[1].plot(list(range(lag_range[0], lag_range[1])), pearson_r)
            ax[1].axvline(np.ceil(len(pearson_r)/2) + lag_range[0],color='k',linestyle='--')
            ax[1].axvline(np.argmax(pearson_r) + lag_range[0],color='r',linestyle='--',label='Peak synchrony')
            plt.annotate(f'Max correlation={np.round(max_corr,2)}', xy=(0.05, 0.9), xycoords='axes fraction')
            ax[1].set(title=f'Offset = {offset} frames', xlabel='Offset (frames)',ylabel='Pearson r')
            
            plt.legend()
            f.tight_layout()
            plt.show()
    else:
        max_corr = 0
        offset = 0
        if show:
            # print('No good values to interpolate')
            pass

    return offset, max_corr


def synchronize_cams_all(config_dict):
    '''
    Post-synchronize your cameras in case they are not natively synchronized.

    For each camera, computes mean vertical speed for the chosen keypoints, 
    and find the time offset for which their correlation is highest. 

    Depending on the analysed motion, all keypoints can be taken into account, 
    or a list of them, or the right or left side.
    All frames can be considered, or only those around a specific time (typically, 
    the time when there is a single participant in the scene performing a clear vertical motion).
    Has also been successfully tested for synchronizing random walks without a specific movement.

    Keypoints whose likelihood is too low are filtered out; and the remaining ones are 
    filtered with a butterworth filter.

    INPUTS: 
    - json files from each camera folders
    - a Config.toml file
    - a skeleton model

    OUTPUTS: 
    - synchronized json files for each camera
    '''
    
    # Get parameters from Config.toml
    project_dir = config_dict.get('project').get('project_dir')
    pose_dir = os.path.realpath(os.path.join(project_dir, 'pose'))
    pose_model = config_dict.get('pose').get('pose_model')
    multi_person = config_dict.get('project').get('multi_person')
    fps =  config_dict.get('project').get('frame_rate')
    frame_range = config_dict.get('project').get('frame_range')
    display_sync_plots = config_dict.get('synchronization').get('display_sync_plots')
    keypoints_to_consider = config_dict.get('synchronization').get('keypoints_to_consider')
    approx_time_maxspeed = config_dict.get('synchronization').get('approx_time_maxspeed') 
    time_range_around_maxspeed = config_dict.get('synchronization').get('time_range_around_maxspeed')
    manual_selection = config_dict.get('synchronization').get('manual_person_selection')

    likelihood_threshold = config_dict.get('synchronization').get('likelihood_threshold')
    filter_cutoff = int(config_dict.get('synchronization').get('filter_cutoff'))
    filter_order = int(config_dict.get('synchronization').get('filter_order'))

    # Determine frame rate
    video_dir = os.path.join(project_dir, 'videos')
    vid_img_extension = config_dict['pose']['vid_img_extension']
    vid_or_img_files = glob.glob(os.path.join(video_dir, '*'+vid_img_extension))
    if not vid_or_img_files: # video_files is then img_dirs
        image_folders = [f for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f))]
        for image_folder in image_folders:
            vid_or_img_files.append(glob.glob(os.path.join(video_dir, image_folder, '*'+vid_img_extension)))

    if fps == 'auto': 
        try:
            cap = cv2.VideoCapture(vid_or_img_files[0])
            cap.read()
            if cap.read()[0] == False:
                raise
            fps = round(cap.get(cv2.CAP_PROP_FPS))
        except:
            fps = 60  
    lag_range = time_range_around_maxspeed*fps # frames

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

    # List json files
    try:
        pose_listdirs_names = next(os.walk(pose_dir))[1]
        os.listdir(os.path.join(pose_dir, pose_listdirs_names[0]))[0]
    except:
        raise ValueError(f'No json files found in {pose_dir} subdirectories. Make sure you run Pose2Sim.poseEstimation() first.')
    pose_listdirs_names = sort_stringlist_by_last_number(pose_listdirs_names)
    json_dirs_names = [k for k in pose_listdirs_names if 'json' in k]
    json_dirs = [os.path.join(pose_dir, j_d) for j_d in json_dirs_names] # list of json directories in pose_dir
    json_files_names = [fnmatch.filter(os.listdir(os.path.join(pose_dir, js_dir)), '*.json') for js_dir in json_dirs_names]
    json_files_names = [sort_stringlist_by_last_number(j) for j in json_files_names]
    nb_frames_per_cam = [len(fnmatch.filter(os.listdir(os.path.join(json_dir)), '*.json')) for json_dir in json_dirs]
    cam_nb = len(json_dirs)
    cam_list = list(range(cam_nb))
    cam_names = [os.path.basename(j_dir).split('_')[0] for j_dir in json_dirs]
    
    # frame range selection
    f_range = [[0, min([len(j) for j in json_files_names])] if frame_range==[] else frame_range][0]
    # json_files_names = [[j for j in json_files_cam if int(re.split(r'(\d+)',j)[-2]) in range(*f_range)] for json_files_cam in json_files_names]

    # Determine frames to consider for synchronization
    if isinstance(approx_time_maxspeed, list): # search around max speed
        approx_frame_maxspeed = [int(fps * t) for t in approx_time_maxspeed]
        nb_frames_per_cam = [len(fnmatch.filter(os.listdir(os.path.join(json_dir)), '*.json')) for json_dir in json_dirs]
        search_around_frames = [[int(a-lag_range) if a-lag_range>0 else 0, int(a+lag_range) if a+lag_range<nb_frames_per_cam[i] else nb_frames_per_cam[i]+f_range[0]] for i,a in enumerate(approx_frame_maxspeed)]
        logging.info(f'Synchronization is calculated around the times {approx_time_maxspeed} +/- {time_range_around_maxspeed} s.')
    elif approx_time_maxspeed == 'auto': # search on the whole sequence (slower if long sequence)
        search_around_frames = [[f_range[0], f_range[0]+nb_frames_per_cam[i]] for i in range(cam_nb)]
        logging.info('Synchronization is calculated on the whole sequence. This may take a while.')
    else:
        raise ValueError('approx_time_maxspeed should be a list of floats or "auto"')
    
    if keypoints_to_consider == 'right':
        logging.info(f'Keypoints used to compute the best synchronization offset: right side.')
    elif keypoints_to_consider == 'left':
        logging.info(f'Keypoints used to compute the best synchronization offset: left side.')
    elif isinstance(keypoints_to_consider, list):
        logging.info(f'Keypoints used to compute the best synchronization offset: {keypoints_to_consider}.')
    elif keypoints_to_consider == 'all':
        logging.info(f'All keypoints are used to compute the best synchronization offset.')
    logging.info(f'These keypoints are filtered with a Butterworth filter (cut-off frequency: {filter_cutoff} Hz, order: {filter_order}).')
    logging.info(f'They are removed when their likelihood is below {likelihood_threshold}.\n')

    # Extract, interpolate, and filter keypoint coordinates
    logging.info('Synchronizing...')
    df_coords = []
    b, a = signal.butter(filter_order/2, filter_cutoff/(fps/2), 'low', analog = False) 
    json_files_names_range = [[j for j in json_files_cam if int(re.split(r'(\d+)',j)[-2]) in range(*frames_cam)] for (json_files_cam, frames_cam) in zip(json_files_names,search_around_frames)]
    json_files_range = [[os.path.join(pose_dir, j_dir, j_file) for j_file in json_files_names_range[j]] for j, j_dir in enumerate(json_dirs_names)]
    
    if np.array([j==[] for j in json_files_names_range]).any():
        raise ValueError(f'No json files found within the specified frame range ({frame_range}) at the times {approx_time_maxspeed} +/- {time_range_around_maxspeed} s.')
    
    # Handle manual selection if multi person is True
    if manual_selection:
        selected_id_list, keypoints_to_consider, approx_time_maxspeed = select_person(
            vid_or_img_files, cam_names, json_files_names_range, search_around_frames, 
            pose_dir, json_dirs_names, keypoints_names)
    else:
        selected_id_list = [None] * cam_nb

    for i in range(cam_nb):
        df_coords.append(convert_json2pandas(json_files_range[i], likelihood_threshold=likelihood_threshold, keypoints_ids=keypoints_ids, multi_person=multi_person, selected_id=selected_id_list[i]))
        df_coords[i] = drop_col(df_coords[i],3) # drop likelihood
        if keypoints_to_consider == 'right':
            kpt_indices = [i for i in range(len(keypoints_ids)) if keypoints_names[i].startswith('R') or keypoints_names[i].startswith('right')]
        elif keypoints_to_consider == 'left':
            kpt_indices = [i for i in range(len(keypoints_ids)) if keypoints_names[i].startswith('L') or keypoints_names[i].startswith('left')]
        elif isinstance(keypoints_to_consider, list):
            kpt_indices = [i for i in range(len(keypoints_ids)) if keypoints_names[i] in keypoints_to_consider]
        elif keypoints_to_consider == 'all':
            kpt_indices = [i for i in range(len(keypoints_ids))]
        else:
            raise ValueError('keypoints_to_consider should be "all", "right", "left", or a list of keypoint names.\n\
                            If you specified keypoints, make sure that they exist in your pose_model.')
        
        kpt_indices = np.sort(np.concatenate([np.array(kpt_indices)*2, np.array(kpt_indices)*2+1]))
        df_coords[i] = df_coords[i][kpt_indices]
        df_coords[i] = df_coords[i].apply(interpolate_zeros_nans, axis=0, args = ['linear'])
        df_coords[i] = df_coords[i].bfill().ffill()
        df_coords[i] = pd.DataFrame(signal.filtfilt(b, a, df_coords[i], axis=0))


    # Compute sum of speeds
    df_speed = []
    sum_speeds = []
    for i in range(cam_nb):
        df_speed.append(vert_speed(df_coords[i]))
        sum_speeds.append(abs(df_speed[i]).sum(axis=1))
        # nb_coords = df_speed[i].shape[1]
        # sum_speeds[i][ sum_speeds[i]>vmax*nb_coords ] = 0
        
        # # Replace 0 by random values, otherwise 0 padding may lead to unreliable correlations
        # sum_speeds[i].loc[sum_speeds[i] < 1] = sum_speeds[i].loc[sum_speeds[i] < 1].apply(lambda x: np.random.normal(0,1))
        
        sum_speeds[i] = pd.DataFrame(signal.filtfilt(b, a, sum_speeds[i], axis=0)).squeeze()


    # Compute offset for best synchronization:
    # Highest correlation of sum of absolute speeds for each cam compared to reference cam
    ref_cam_id = nb_frames_per_cam.index(min(nb_frames_per_cam)) # ref cam: least amount of frames
    ref_cam_name = cam_names[ref_cam_id]
    ref_frame_nb = len(df_coords[ref_cam_id])
    lag_range = int(ref_frame_nb/2)
    cam_list.pop(ref_cam_id)
    cam_names.pop(ref_cam_id)
    offset = []
    for cam_id, cam_name in zip(cam_list, cam_names):
        offset_cam_section, max_corr_cam = time_lagged_cross_corr(sum_speeds[ref_cam_id], sum_speeds[cam_id], lag_range, show=display_sync_plots, ref_cam_name=ref_cam_name, cam_name=cam_name)
        offset_cam = offset_cam_section - (search_around_frames[ref_cam_id][0] - search_around_frames[cam_id][0])
        if isinstance(approx_time_maxspeed, list):
            logging.info(f'--> Camera {ref_cam_name} and {cam_name}: {offset_cam} frames offset ({offset_cam_section} on the selected section), correlation {round(max_corr_cam, 2)}.')
        else:
            logging.info(f'--> Camera {ref_cam_name} and {cam_name}: {offset_cam} frames offset, correlation {round(max_corr_cam, 2)}.')
        offset.append(offset_cam)
    offset.insert(ref_cam_id, 0)

    # rename json files according to the offset and copy them to pose-sync
    sync_dir = os.path.abspath(os.path.join(pose_dir, '..', 'pose-sync'))
    os.makedirs(sync_dir, exist_ok=True)
    for d, j_dir in enumerate(json_dirs):
        os.makedirs(os.path.join(sync_dir, os.path.basename(j_dir)), exist_ok=True)
        for j_file in json_files_names[d]:
            j_split = re.split(r'(\d+)',j_file)
            j_split[-2] = f'{int(j_split[-2])-offset[d]:06d}'
            if int(j_split[-2]) > 0:
                json_offset_name = ''.join(j_split)
                shutil.copy(os.path.join(pose_dir, os.path.basename(j_dir), j_file), os.path.join(sync_dir, os.path.basename(j_dir), json_offset_name))

    logging.info(f'Synchronized json files saved in {sync_dir}.')


def handle_person_change(text, selected_idx_container, person_textbox):
    '''
    Handle changes to the person selection text box.
    
    INPUTS:
    - text: str. The text entered in the person selection box.
    - selected_idx_container: list. Container for the selected person index.
    - person_textbox: TextBox. The person selection text box widget.
    '''
    try:
        selected_idx_container[0] = int(text)
    except ValueError:
        person_textbox.set_val('0')
        selected_idx_container[0] = 0


def handle_keypoints_change(text, keypoints_to_consider_container):
    '''
    Handle changes to the keypoints selection text box.
    
    INPUTS:
    - text: str. The text entered in the keypoints selection box.
    - keypoints_to_consider_container: list. Container for the selected keypoints.
    '''
    keypoints_to_consider_container[0] = text.split(',')


def handle_frame_change(text, frame_number, frame_textbox, cap, ax_video, frame_to_json, 
                       pose_dir, json_dir_name, rects, annotations, bounding_boxes_list, 
                       fig, search_around_frames, i):
    '''
    Handle changes to the frame number text box.
    
    INPUTS:
    - text: str. The text entered in the frame number box.
    - frame_number: int. The current frame number.
    - frame_textbox: TextBox. The frame number text box widget.
    - Other parameters: Same as in update_play function.
    '''
    try:
        frame_num = int(text)
        if search_around_frames[i][0] <= frame_num <= search_around_frames[i][1]:
            update_play(cap, ax_video.images[0], frame_num, frame_to_json, 
                       pose_dir, json_dir_name, rects, annotations, 
                       bounding_boxes_list, ax_video, fig)
    except ValueError:
        frame_textbox.set_val(str(frame_number))


def handle_prev_frame(frame_textbox, search_around_frames, i, cap, ax_video, frame_to_json,
                     pose_dir, json_dir_name, rects, annotations, bounding_boxes_list, fig):
    '''
    Handle previous frame button click.
    
    INPUTS:
    - frame_textbox: TextBox. The frame number text box widget.
    - Other parameters: Same as in update_play function.
    '''
    current = int(frame_textbox.text)
    if current > search_around_frames[i][0]:
        new_frame = str(current - 1)
        frame_textbox.set_val(new_frame)
        handle_frame_change(new_frame, current, frame_textbox, cap, ax_video, frame_to_json,
                          pose_dir, json_dir_name, rects, annotations, bounding_boxes_list,
                          fig, search_around_frames, i)


def handle_next_frame(frame_textbox, search_around_frames, i, cap, ax_video, frame_to_json,
                     pose_dir, json_dir_name, rects, annotations, bounding_boxes_list, fig):
    '''
    Handle next frame button click.
    
    INPUTS:
    - frame_textbox: TextBox. The frame number text box widget.
    - Other parameters: Same as in update_play function.
    '''
    current = int(frame_textbox.text)
    if current < search_around_frames[i][1]:
        new_frame = str(current + 1)
        frame_textbox.set_val(new_frame)
        handle_frame_change(new_frame, current, frame_textbox, cap, ax_video, frame_to_json,
                          pose_dir, json_dir_name, rects, annotations, bounding_boxes_list,
                          fig, search_around_frames, i)


def handle_key_press(event, frame_textbox, search_around_frames, i, cap, ax_video, frame_to_json,
                    pose_dir, json_dir_name, rects, annotations, bounding_boxes_list, fig):
    '''
    Handle keyboard navigation events.
    
    INPUTS:
    - event: Event. The keyboard event.
    - Other parameters: Same as in update_play function.
    '''
    if event.key == 'left':
        handle_prev_frame(frame_textbox, search_around_frames, i, cap, ax_video, frame_to_json,
                         pose_dir, json_dir_name, rects, annotations, bounding_boxes_list, fig)
    elif event.key == 'right':
        handle_next_frame(frame_textbox, search_around_frames, i, cap, ax_video, frame_to_json,
                         pose_dir, json_dir_name, rects, annotations, bounding_boxes_list, fig)