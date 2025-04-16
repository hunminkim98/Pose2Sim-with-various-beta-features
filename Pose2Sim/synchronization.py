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
    Has also been successfully tested for synchronizing random walks with random walks.

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

from Pose2Sim.common import sort_stringlist_by_last_number, bounding_boxes, interpolate_zeros_nans
from Pose2Sim.skeletons import *

from .Track_person import animate_pre_post_tracking


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon, HunMin Kim"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
from importlib.metadata import version
__version__ = version('pose2sim')
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


# UI FUNCTIONS
# Global matplotlib settings - remove toolbar
plt.rcParams['toolbar'] = 'none'

# Define keypoint UI parameters
TITLE_SIZE = 12
LABEL_SIZE_KEYPOINTS = 8 # defined twice
BTN_WIDTH_KEYPOINTS = 0.16 # defined twice
BTN_HEIGHT = 0.04
BTN_Y = 0.02
CENTER_X = 0.5
SELECTED_COLOR = 'darkorange'
UNSELECTED_COLOR = 'blue'
NONE_COLOR = 'silver'
BTN_COLOR = 'white'  # Light gray
BTN_HOVER_COLOR = '#D3D3D3'  # Darker gray on hover

# Define person UI parameters
BACKGROUND_COLOR = 'white'
TEXT_COLOR = 'black'
CONTROL_COLOR = 'white'
CONTROL_HOVER_COLOR = '#D3D3D3'
SLIDER_COLOR = '#4682B4'
SLIDER_HIGHLIGHT_COLOR = 'moccasin'
SLIDER_EDGE_COLOR = (0.5, 0.5, 0.5, 0.5)
LABEL_SIZE_PERSON = 10
TEXT_SIZE = 9.5
BUTTON_SIZE = 10
TEXTBOX_WIDTH = 0.09
BTN_WIDTH_PERSON = 0.04
CONTROL_HEIGHT = 0.04
Y_POSITION = 0.1


def reset_styles(rect, annotation):
    '''
    Resets the visual style of a bounding box and its annotation to default.
    
    INPUTS:
    - rect: Matplotlib Rectangle object representing a bounding box
    - annotation: Matplotlib Text object containing the label for the bounding box
    '''

    rect.set_linewidth(1)
    rect.set_edgecolor('white')
    rect.set_facecolor((1, 1, 1, 0.1))
    annotation.set_fontsize(7)
    annotation.set_fontweight('normal')


def create_textbox(ax_pos, label, initial, UI_PARAMS):
    '''
    Creates a textbox widget with consistent styling.
    
    INPUTS:
    - ax_pos: List or tuple containing the position of the axes in the figure [left, bottom, width, height]
    - label: String label for the textbox
    - initial: Initial text value
    - UI_PARAMS: Dictionary containing UI parameters with colors and sizes settings
    
    OUTPUTS:
    - textbox: The created TextBox widget
    '''

    ax = plt.axes(ax_pos)
    ax.set_facecolor(UI_PARAMS['colors']['control'])
    textbox = TextBox(
        ax, 
        label,
        initial=initial,
        color=UI_PARAMS['colors']['control'],
        hovercolor=UI_PARAMS['colors']['control_hover'],
        label_pad=0.1
    )
    textbox.label.set_color(UI_PARAMS['colors']['text'])
    textbox.label.set_fontsize(UI_PARAMS['sizes']['label'])
    textbox.text_disp.set_color(UI_PARAMS['colors']['text'])
    textbox.text_disp.set_fontsize(UI_PARAMS['sizes']['text'])

    return textbox


## Handlers
def handle_ok_button(ui):
    '''
    Handles the OK button click event.
    
    INPUTS:
    - ui: Dictionary containing all UI elements and state
    - fps: Frames per second of the video
    - i: Current camera index
    - selected_id_list: List to store the selected person ID for each camera
    - approx_time_maxspeed: List to store the approximate time of maximum speed for each camera
    '''

    try:
        float(ui['controls']['main_time_textbox'].text)
        float(ui['controls']['time_RAM_textbox'].text)
        int(ui['controls']['person_textbox'].text)
        plt.close(ui['fig'])
    except ValueError:
        logging.warning('Invalid input in textboxes.')
        

def handle_person_change(text, selected_idx_container, person_textbox):
    '''
    Handles changes to the person selection text box.
    
    INPUTS:
    - text: Text from the person selection text box
    - selected_idx_container: List with one element to store the selected person's index
    - person_textbox: TextBox widget for displaying and editing the selected person number
    '''

    try:
        selected_idx_container[0] = int(text)
    except ValueError:
        person_textbox.set_val('0')
        selected_idx_container[0] = 0


def handle_frame_navigation(direction, frame_textbox, search_around_frames, i, cap, ax_video, frame_to_json,
                           pose_dir, json_dir_name, rects, annotations, bounding_boxes_list, fig,
                           time_range_around_maxspeed, fps, ui):
    '''
    Handles frame navigation (previous or next frame).
    
    INPUTS:
    - direction: Integer, -1 for previous frame, 1 for next frame
    - frame_textbox: TextBox widget for displaying and editing the frame number
    - search_around_frames: Frame ranges to search around for each camera
    - i: Current camera index
    - cap: Video capture object
    - ax_video: Axes for video display
    - frame_to_json: Mapping from frame numbers to JSON files
    - pose_dir: Directory containing pose data
    - json_dir_name: Name of the JSON directory for the current camera
    - rects: List of rectangle patches representing bounding boxes
    - annotations: List of text annotations for each bounding box
    - bounding_boxes_list: List of bounding boxes for detected persons
    - fig: The figure object to update
    - time_range_around_maxspeed: Time range to consider around max speed
    - fps: Frames per second of the video
    - ui: Dictionary containing all UI elements and state
    '''

    time_val = float(frame_textbox.text.split(' ±')[0])
    current = round(time_val * fps)
    
    # Check bounds based on direction
    if (direction < 0 and current > search_around_frames[i][0]) or \
       (direction > 0 and current < search_around_frames[i][1]):
        next_frame = current + direction
        handle_frame_change(next_frame, frame_textbox, cap, ax_video, frame_to_json,
                            pose_dir, json_dir_name, rects, annotations, bounding_boxes_list,
                            fig, search_around_frames, i, time_range_around_maxspeed, fps, ui)


def handle_frame_change(frame_number, frame_textbox, cap, ax_video, frame_to_json, 
                        pose_dir, json_dir_name, rects, annotations, bounding_boxes_list, 
                        fig, search_around_frames, i, time_range_around_maxspeed, fps, ui):
    '''
    Handles changes to the frame number text box.
    
    INPUTS:
    - text: Text from the frame number text box
    - frame_number: The current frame number
    - frame_textbox: TextBox widget for displaying and editing the frame number
    - cap: Video capture object
    - ax_video: Axes for video display
    - frame_to_json: Mapping from frame numbers to JSON files
    - pose_dir: Directory containing pose data
    - json_dir_name: Name of the JSON directory for the current camera
    - rects: List of rectangle patches representing bounding boxes
    - annotations: List of text annotations for each bounding box
    - bounding_boxes_list: List of bounding boxes for detected persons
    - fig: The figure object to update
    - search_around_frames: Frame ranges to search around for each camera
    - i: Current camera index
    - time_range_around_maxspeed: Time range to consider around max speed
    - fps: Frames per second of the video
    - ui: Dictionary containing all UI elements and state
    '''

    if search_around_frames[i][0] <= frame_number <= search_around_frames[i][1]:
        # Update video frame first
        update_play(cap, ax_video.images[0], frame_number, frame_to_json, 
                    pose_dir, json_dir_name, rects, annotations, 
                    bounding_boxes_list, ax_video, fig)
        
        # Update UI elements
        frame_textbox.eventson = False
        new_time = frame_number / fps
        frame_textbox.set_val(f"{new_time:.2f} ±{time_range_around_maxspeed}")
        frame_textbox.eventson = True
        
        # Update slider and highlight
        ui['controls']['frame_slider'].set_val(frame_number)
        update_highlight(frame_number, time_range_around_maxspeed, fps, search_around_frames, i, ui['axes']['slider'], ui['controls'])
        fig.canvas.draw_idle()


def handle_key_press(event, frame_textbox, search_around_frames, i, cap, ax_video, frame_to_json,
                     pose_dir, json_dir_name, rects, annotations, bounding_boxes_list, fig,
                     time_range_around_maxspeed, fps, ui):
    '''
    Handles keyboard navigation through video frames.
    
    INPUTS:
    - event: Matplotlib keyboard event object
    - frame_textbox: TextBox widget for displaying and editing the frame number
    - search_around_frames: Frame ranges to search around for each camera
    - i: Current camera index
    - cap: Video capture object
    - ax_video: Axes for video display
    - frame_to_json: Mapping from frame numbers to JSON files
    - pose_dir: Directory containing pose data
    - json_dir_name: Name of the JSON directory for the current camera
    - rects: List of rectangle patches representing bounding boxes
    - annotations: List of text annotations for each bounding box
    - bounding_boxes_list: List of bounding boxes for detected persons
    - fig: The figure object to update
    - time_range_around_maxspeed: Time range to consider around max speed
    - fps: Frames per second of the video
    - ui: Dictionary containing all UI elements and state
    '''

    direction = 0
    if event.key == 'left':
        direction = -1
    elif event.key == 'right':
        direction = 1
    if direction != 0:
        handle_frame_navigation(direction, frame_textbox, search_around_frames, i, cap, ax_video, frame_to_json,
                              pose_dir, json_dir_name, rects, annotations, bounding_boxes_list, fig,
                              time_range_around_maxspeed, fps, ui)


def handle_toggle_labels(keypoint_texts, containers, btn_toggle):
    '''
    Handle toggle labels button click.
    
    INPUTS:
    - event: Matplotlib event object
    - keypoint_texts: List of text objects for keypoint labels
    - containers: Dictionary of container objects
    - btn_toggle: Button object for toggling label visibility
    '''

    containers['show_labels'][0] = not containers['show_labels'][0]  # Toggle visibility state
    for text in keypoint_texts:
        text.set_visible(containers['show_labels'][0])
    # Update button text
    btn_toggle.label.set_text('Hide names' if containers['show_labels'][0] else 'Show names')
    plt.draw()


## Highlighters
def highlight_selected_box(rect, annotation):
    '''
    Highlights a selected rectangle and its annotation with bold orange style.
    
    INPUTS:
    - rect: Matplotlib Rectangle object to highlight
    - annotation: Matplotlib Text object to highlight
    '''

    rect.set_linewidth(2)
    rect.set_edgecolor(SELECTED_COLOR)
    rect.set_facecolor((1, 1, 1, 0.1))
    annotation.set_fontsize(8)
    annotation.set_fontweight('bold')


def highlight_hover_box(rect, annotation):
    '''
    Highlights a hovered rectangle and its annotation with yellow-orange style.
    
    INPUTS:
    - rect: Matplotlib Rectangle object to apply hover effect to
    - annotation: Matplotlib Text object to style for hover state
    '''

    rect.set_linewidth(2)
    rect.set_edgecolor(SELECTED_COLOR)
    rect.set_facecolor((1, 1, 0, 0.2))
    annotation.set_fontsize(8)
    annotation.set_fontweight('bold')


## on_family
def on_hover(event, fig, rects, annotations, bounding_boxes_list, selected_idx_container=None):
    '''
    Manages hover effects for bounding boxes in the video frame.
    
    INPUTS:
    - event: Matplotlib event object containing mouse position
    - fig: Matplotlib figure to update
    - rects: List of rectangle patches representing bounding boxes
    - annotations: List of text annotations for each bounding box
    - bounding_boxes_list: List of bounding box coordinates (x_min, y_min, x_max, y_max)
    - selected_idx_container: Optional container holding the index of the currently selected box
    '''

    if event.xdata is None or event.ydata is None:
        return

    # First reset all boxes to default style
    for idx, (rect, annotation) in enumerate(zip(rects, annotations)):
        if selected_idx_container and idx == selected_idx_container[0]:
            # Keep the selected box highlighted with bold white style
            highlight_selected_box(rect, annotation)
        else:
            reset_styles(rect, annotation)

    # Then apply hover effect to the box under cursor (even if it's selected)
    bounding_boxes_list = [bbox for bbox in bounding_boxes_list if np.all(np.isfinite(bbox)) and not np.any(np.isnan(bbox))]
    
    for idx, (x_min, y_min, x_max, y_max) in enumerate(bounding_boxes_list):
        if x_min <= event.xdata <= x_max and y_min <= event.ydata <= y_max:
            highlight_hover_box(rects[idx], annotations[idx])
            break

    fig.canvas.draw_idle()


def on_click(event, ax, bounding_boxes_list, selected_idx_container, person_textbox):
    '''
    Detects clicks on person bounding boxes and updates the selection state.
    
    INPUTS:
    - event: Matplotlib event object containing click information
    - ax: The axes object of the video frame
    - bounding_boxes_list: List of tuples containing bounding box coordinates (x_min, y_min, x_max, y_max)
    - selected_idx_container: List with one element to store the selected person's index
    - person_textbox: TextBox widget for displaying and editing the selected person number
    '''

    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return

    for idx, (x_min, y_min, x_max, y_max) in enumerate(bounding_boxes_list):
        if x_min <= event.xdata <= x_max and y_min <= event.ydata <= y_max:
            selected_idx_container[0] = idx
            person_textbox.set_val(str(idx))  # Update the person number text box
            break


def on_slider_change(val, fps, controls, fig, search_around_frames, cam_index, ax_slider):
    '''
    Updates UI elements when the frame slider value changes.
    
    INPUTS:
    - val: The current slider value (frame number)
    - fps: Frames per second of the video
    - controls: Dictionary containing UI control elements
    - fig: Matplotlib figure to update
    - search_around_frames: Frame ranges to search within
    - cam_index: Current camera index
    - ax_slider: The slider axes object
    '''

    frame_number = int(val)
    main_time = frame_number / fps
    controls['main_time_textbox'].set_val(f"{main_time:.2f}")
    try:
        time_RAM = float(controls['time_RAM_textbox'].text)
    except ValueError:
        time_RAM = 0
    update_highlight(frame_number, time_RAM, fps, search_around_frames, cam_index, ax_slider, controls)
    fig.canvas.draw_idle()


def on_key(event, ui, fps, cap, frame_to_json, pose_dir, json_dirs_names, i, search_around_frames, bounding_boxes_list):
    '''
    Handles keyboard navigation through video frames.
    
    INPUTS:
    - event: Matplotlib keyboard event object
    - ui: Dictionary containing all UI elements and state
    - fps: Frames per second of the video
    - cap: Video capture object
    - frame_to_json: Mapping of frame numbers to JSON files
    - pose_dir: Directory containing pose data
    - json_dirs_names: List of JSON directory names
    - i: Current camera index
    - search_around_frames: Frame ranges to search around for each camera
    - bounding_boxes_list: List of bounding boxes for detected persons
    '''

    if event.key == 'left':
        handle_frame_navigation(-1, ui['controls']['main_time_textbox'], search_around_frames, i, cap, ui['ax_video'], frame_to_json,
                              pose_dir, json_dirs_names[i], ui['containers']['rects'], ui['containers']['annotations'], bounding_boxes_list, ui['fig'],
                              float(ui['controls']['time_RAM_textbox'].text), fps, ui)
    elif event.key == 'right':
        handle_frame_navigation(1, ui['controls']['main_time_textbox'], search_around_frames, i, cap, ui['ax_video'], frame_to_json,
                              pose_dir, json_dirs_names[i], ui['containers']['rects'], ui['containers']['annotations'], bounding_boxes_list, ui['fig'],
                              float(ui['controls']['time_RAM_textbox'].text), fps, ui)


## UI Update Functions
def update_highlight(current_frame, time_RAM, fps, search_around_frames, cam_index, ax_slider, controls):
    '''
    Updates the highlighted range on the frame slider.
    
    INPUTS:
    - current_frame: The current frame number
    - time_RAM: Time range in seconds to highlight around the current frame
    - fps: Frames per second of the video
    - search_around_frames: Valid frame range limits for the current camera
    - cam_index: Current camera index
    - ax_slider: The slider axes object
    - controls: Dictionary containing UI controls and state
    '''

    if 'range_highlight' in controls:
        controls['range_highlight'].remove()
    range_start = max(current_frame - time_RAM * fps, search_around_frames[cam_index][0])
    range_end = min(current_frame + time_RAM * fps, search_around_frames[cam_index][1])
    controls['range_highlight'] = ax_slider.axvspan(range_start, range_end, 
                                                  ymin=0.20, ymax=0.80,
                                                  color=SLIDER_HIGHLIGHT_COLOR, alpha=0.5, zorder=4)


def update_main_time(text, fps, search_around_frames, i, ui, cap, frame_to_json, pose_dir, json_dirs_names, bounding_boxes_list):
    '''
    Updates the UI based on changes to the main time textbox.
    
    INPUTS:
    - text: Text from the main time textbox
    - fps: Frames per second of the video
    - search_around_frames: Valid frame range limits for each camera
    - i: Current camera index
    - ui: Dictionary containing all UI elements and state
    - cap: Video capture object
    - frame_to_json: Mapping of frame numbers to JSON files
    - pose_dir: Directory containing pose data
    - json_dirs_names: List of JSON directory names
    - bounding_boxes_list: List of bounding boxes for detected persons
    '''

    try:
        main_time = float(text)
        frame_num = int(round(main_time * fps))
        frame_num = max(search_around_frames[i][0], min(frame_num, search_around_frames[i][1]))
        ui['controls']['frame_slider'].set_val(frame_num)
        update_frame(frame_num, fps, ui, frame_to_json, pose_dir, json_dirs_names, i, search_around_frames, bounding_boxes_list)
    except ValueError:
        pass


def update_time_RAM(text, fps, search_around_frames, i, ui):
    '''
    time_RAM = time_range_around_maxspeed
    Updates the highlight range based on changes to the time_RAM textbox.

    INPUTS:
    - text: Text from the time_RAM textbox
    - fps: Frames per second of the video
    - search_around_frames: Valid frame range limits for each camera
    - i: Current camera index
    - ui: Dictionary containing UI elements and controls
    '''
    
    try:
        time_RAM = float(text)
        if time_RAM < 0:
            time_RAM = 0
        frame_num = int(ui['controls']['frame_slider'].val)
        update_highlight(frame_num, time_RAM, fps, search_around_frames, i, ui['axes']['slider'], ui['controls'])
    except ValueError:
        pass


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


def update_keypoint_selection(selected_keypoints, all_keypoints, keypoints_names, scatter, keypoint_texts, selected_text, btn_all_none,
                              SELECTED_COLOR, UNSELECTED_COLOR, NONE_COLOR):
    '''
    Updates the selected keypoints and their visualization on the scatter plot.
    
    INPUTS:
    - selected_keypoints: List of keypoints that are currently selected
    - all_keypoints: List of all available keypoints
    - keypoints_names: List of valid keypoint names
    - scatter: Scatter plot object to update
    - keypoint_texts: List of text objects for keypoint labels
    - selected_text: Text object displaying the currently selected keypoints
    - btn_all_none: Button object for toggling between "Select All" and "Select None"
    - SELECTED_COLOR: Color to use for selected keypoints
    - UNSELECTED_COLOR: Color to use for unselected keypoints
    - NONE_COLOR: Color to use for non-keypoint elements
    
    OUTPUTS:
    - None. Updates the visualization in place.
    '''
    # Update scatter colors
    colors = [
        SELECTED_COLOR if kp in selected_keypoints else UNSELECTED_COLOR if kp in keypoints_names else NONE_COLOR
        for kp in all_keypoints
    ]
    scatter.set_facecolors(colors)
    
    # Update text weights
    for text, kp in zip(keypoint_texts, all_keypoints):
        text.set_fontweight('bold' if kp in selected_keypoints else 'normal')
    
    # Update selected text and button label
    if selected_keypoints:
        text_parts = ['Selected: '] + [f'$\\bf{{{kp}}}$' if i == 0 else f', $\\bf{{{kp}}}$' for i, kp in enumerate(selected_keypoints)]
        selected_text.set_text(''.join(text_parts))
        btn_all_none.label.set_text('Select None')
    else:
        selected_text.set_text('Selected: None\nClick on keypoints to select them')
        btn_all_none.label.set_text('Select All')
    
    plt.draw()


def update_frame(val, fps, ui, frame_to_json, pose_dir, json_dirs_names, i, search_around_frames, bounding_boxes_list):
    '''
    Synchronizes all UI elements when the frame number changes.
    
    INPUTS:
    - val: The current frame value from the slider
    - fps: Frames per second of the video
    - ui: Dictionary containing UI elements and controls
    - cap: Video capture object
    - frame_to_json: Mapping of frame numbers to JSON files
    - pose_dir: Directory containing pose data
    - json_dirs_names: List of JSON directory names
    - i: Current camera index
    - search_around_frames: Frame ranges to search around for each camera
    - bounding_boxes_list: List of bounding boxes for detected persons
    '''

    frame_num = int(val)
    main_time = frame_num / fps
    ui['controls']['main_time_textbox'].set_val(f"{main_time:.2f}")
    
    # Update yellow highlight position
    try:
        time_RAM = float(ui['controls']['time_RAM_textbox'].text)
    except ValueError:
        time_RAM = 0
        
    # Update highlight
    update_highlight(frame_num, time_RAM, fps, search_around_frames, i, ui['axes']['slider'], ui['controls'])
    
    # Update video frame and bounding boxes
    update_play(ui['cap'], ui['ax_video'].images[0], frame_num, frame_to_json, 
            pose_dir, json_dirs_names[i], ui['containers']['rects'], 
            ui['containers']['annotations'], bounding_boxes_list, 
            ui['ax_video'], ui['fig'])
    
    # Update canvas
    ui['fig'].canvas.draw_idle()


def update_play(cap, image, frame_number, frame_to_json, pose_dir, json_dir_name, rects, annotations, bounding_boxes_list, ax, fig):
    '''
    Updates the video frame and bounding boxes for the given frame number.

    INPUTS:
    - cap: Video capture object or list of image file paths
    - image: The image object to update
    - frame_number: The frame number to display
    - frame_to_json: Mapping from frame numbers to JSON file names
    - pose_dir: Directory containing pose data
    - json_dir_name: Name of the JSON directory for the current camera
    - rects: List of rectangle patches representing bounding boxes
    - annotations: List of text annotations for each bounding box
    - bounding_boxes_list: List to store bounding box coordinates
    - ax: The axes object to draw on
    - fig: The figure object to update
    '''

    # Store the currently selected box index if any
    selected_idx = None
    for idx, rect in enumerate(rects):
        if rect.get_linewidth() > 1:  # If box is highlighted
            selected_idx = idx
            break

    frame_rgb, bounding_boxes_list_new = load_frame_and_bounding_boxes(cap, frame_number, frame_to_json, pose_dir, json_dir_name)
    if frame_rgb is None:
        return

    # Update image
    image.set_array(frame_rgb)
    
    # Clear existing boxes and annotations
    for rect in rects:
        rect.remove()
    for ann in annotations:
        ann.remove()
    rects.clear()
    annotations.clear()
    
    # Update bounding boxes list
    bounding_boxes_list.clear()
    bounding_boxes_list.extend(bounding_boxes_list_new)
    
    # Draw new boxes and annotations
    draw_bounding_boxes_and_annotations(ax, bounding_boxes_list, rects, annotations)
    
    # Restore highlight on the selected box if it still exists
    if selected_idx is not None and selected_idx < len(rects):
        highlight_selected_box(rects[selected_idx], annotations[selected_idx])
    
    fig.canvas.draw_idle()


def keypoints_ui(keypoints_to_consider, keypoints_names):
    '''
    Step 1: Initializes the UI for selecting keypoints.

    This function creates an interactive GUI for selecting keypoints. It displays
    a human figure with selectable keypoints, allows users to toggle keypoint names,
    select all or none, and confirm their selection. The GUI uses matplotlib for
    visualization and interaction.

    The function performs the following steps:
    1. Sets up the figure and axes for the GUI
    2. Defines keypoint positions and colors
    3. Creates interactive elements (scatter plot, buttons, text)
    4. Sets up event handlers for user interactions
    5. Displays the GUI and waits for user input
    6. Returns the list of selected keypoints

    INPUTS:
    - keypoints_names: List of strings. The names of the keypoints to select.

    OUTPUTS:
    - selected_keypoints: List of strings. The names of the selected keypoints.
    '''
    
    # Create figure
    fig = plt.figure(figsize=(6, 8), num='Synchronizing cameras')
    fig.patch.set_facecolor('white')

    # Keypoint selection area
    ax_keypoints = plt.axes([0.1, 0.2, 0.8, 0.7])
    ax_keypoints.set_facecolor('white')
    ax_keypoints.set_title('Select keypoints to synchronize on', fontsize=TITLE_SIZE, pad=10, color='black')
    
    # Define all keypoints and their positions
    all_keypoints = [
        'Hip', 'Neck', 'Head', 'Nose', 
        'RHip', 'RShoulder', 'RElbow', 'RWrist', 
        'RKnee', 'RAnkle', 'RSmallToe', 'RBigToe', 'RHeel', 
        'LHip', 'LShoulder', 'LElbow', 'LWrist', 
        'LKnee', 'LAnkle', 'LSmallToe', 'LBigToe', 'LHeel',
    ]
    keypoints_positions = {
        'Hip': (0.50, 0.42), 'Neck': (0.50, 0.75), 'Head': (0.50, 0.85), 'Nose': (0.53, 0.82), 
        'RHip': (0.42, 0.42), 'RShoulder': (0.40, 0.75), 'RElbow': (0.35, 0.65), 'RWrist': (0.25, 0.50),
        'LHip': (0.58, 0.42), 'LShoulder': (0.60, 0.75), 'LElbow': (0.65, 0.65), 'LWrist': (0.75, 0.50),
        'RKnee': (0.40, 0.25), 'RAnkle': (0.40, 0.05), 'RSmallToe': (0.35, 0.0), 'RBigToe': (0.42, 0.0), 'RHeel': (0.40, 0.02),
        'LKnee': (0.60, 0.25), 'LAnkle': (0.60, 0.05), 'LSmallToe': (0.65, 0.0), 'LBigToe': (0.58, 0.0), 'LHeel': (0.60, 0.02)
    }
    
    # Generate keypoint coordinates
    keypoints_x, keypoints_y = zip(*[keypoints_positions[name] for name in all_keypoints])
    
    # Set initial colors
    initial_colors = [SELECTED_COLOR if kp in keypoints_to_consider else UNSELECTED_COLOR if kp in keypoints_names else NONE_COLOR for kp in all_keypoints]
    
    # Create scatter plot
    selected_keypoints = keypoints_to_consider
    scatter = ax_keypoints.scatter(keypoints_x, keypoints_y, c=initial_colors, picker=True)
    
    # Add keypoint labels
    keypoint_texts = [ax_keypoints.text(x + 0.02, y, name, va='center', fontsize=LABEL_SIZE_KEYPOINTS, color='black', visible=False)
                      for x, y, name in zip(keypoints_x, keypoints_y, all_keypoints)]
    
    ax_keypoints.set_xlim(0, 1)
    ax_keypoints.set_ylim(-0.1, 1)
    ax_keypoints.axis('off')
    
    # Selected keypoints display area
    ax_selected = plt.axes([0.1, 0.08, 0.8, 0.04])
    ax_selected.axis('off')
    ax_selected.set_facecolor('black')
    text_parts = ['Selected: '] + [f'$\\bf{{{kp}}}$' if i == 0 else f', $\\bf{{{kp}}}$' for i, kp in enumerate(selected_keypoints)]
    selected_text = ax_selected.text(0.0, 0.5, ''.join(text_parts), 
                                    va='center', fontsize=BUTTON_SIZE, wrap=True, color='black')
    
    # Add buttons
    btn_all_none = plt.Button(plt.axes([CENTER_X - 1.5*BTN_WIDTH_KEYPOINTS - 0.01, BTN_Y, BTN_WIDTH_KEYPOINTS, BTN_HEIGHT]), 'Select All')
    btn_toggle = plt.Button(plt.axes([CENTER_X - BTN_WIDTH_KEYPOINTS/2, BTN_Y, BTN_WIDTH_KEYPOINTS, BTN_HEIGHT]), 'Show names')
    btn_ok = plt.Button(plt.axes([CENTER_X + 0.5*BTN_WIDTH_KEYPOINTS + 0.01, BTN_Y, BTN_WIDTH_KEYPOINTS, BTN_HEIGHT]), label='OK')
    btn_ok.label.set_fontweight('bold')
    
    # button colors
    for btn in [btn_all_none, btn_toggle, btn_ok]:
        btn.color = BTN_COLOR
        btn.hovercolor = BTN_HOVER_COLOR
    
    # Define containers for data
    containers = {
        'show_labels': [False],  # Label display status
        'selected_keypoints': selected_keypoints  # List of selected keypoints
    }

    # Connect button events
    btn_toggle.on_clicked(lambda event: handle_toggle_labels(keypoint_texts, containers, btn_toggle))
    btn_ok.on_clicked(lambda event: plt.close())
    btn_all_none.on_clicked(lambda event: (
        selected_keypoints.clear() if selected_keypoints else selected_keypoints.extend(keypoints_names),
        update_keypoint_selection(selected_keypoints, all_keypoints, keypoints_names, scatter, keypoint_texts, selected_text, btn_all_none,
        SELECTED_COLOR, UNSELECTED_COLOR, NONE_COLOR)
    )[-1])
    
    fig.canvas.mpl_connect('pick_event', lambda event: (
        (selected_keypoints.remove(all_keypoints[event.ind[0]]) 
         if all_keypoints[event.ind[0]] in selected_keypoints 
         else selected_keypoints.append(all_keypoints[event.ind[0]])) 
        if all_keypoints[event.ind[0]] in keypoints_names else None,
        update_keypoint_selection(selected_keypoints, all_keypoints, keypoints_names, scatter, keypoint_texts, selected_text, btn_all_none,
        SELECTED_COLOR, UNSELECTED_COLOR, NONE_COLOR)
    )[-1] if all_keypoints[event.ind[0]] in keypoints_names else None)
    
    plt.show()
    
    return selected_keypoints


def person_ui(frame_rgb, cam_name, frame_number, search_around_frames, time_range_around_maxspeed, fps, cam_index, frame_to_json, pose_dir, json_dirs_names):
    '''
    Step 2: Initializes the UI for person and frame selection.
    
    INPUTS:
    - frame_rgb: The initial RGB frame to display
    - cam_name: Name of the current camera
    - frame_number: Initial frame number to display
    - search_around_frames: Frame ranges to search around for each camera
    - time_range_around_maxspeed: Time range to consider around max speed
    - fps: Frames per second of the video
    - cam_index: Index of the current camera
    - frame_to_json: Mapping from frame numbers to JSON files
    - pose_dir: Directory containing pose data
    - json_dirs_names: Names of JSON directories for each camera
    
    OUTPUTS:
    - ui: Dictionary containing all UI elements and state
    '''
    
    # Set up UI based on frame size and orientation
    frame_height, frame_width = frame_rgb.shape[:2]
    is_vertical = frame_height > frame_width
    
    # Calculate appropriate figure height based on video orientation
    if is_vertical:
        fig_height = frame_height / 250  # For vertical videos
    else:
        fig_height = max(frame_height / 300, 6)  # For horizontal videos
    
    fig = plt.figure(figsize=(8, fig_height), num=f'Synchronizing cameras')
    fig.patch.set_facecolor(BACKGROUND_COLOR)

    # Adjust UI layout based on video orientation
    video_axes_height = 0.7 if is_vertical else 0.6
    slider_y = 0.15 if is_vertical else 0.2
    controls_y = Y_POSITION if is_vertical else 0.1
    lower_controls_y = controls_y - 0.05  # Y-coordinate for lower controls
    
    ax_video = plt.axes([0.1, 0.2, 0.8, video_axes_height])
    ax_video.imshow(frame_rgb)
    ax_video.axis('off')
    ax_video.set_facecolor(BACKGROUND_COLOR)

    # Create frame slider
    ax_slider = plt.axes([ax_video.get_position().x0, slider_y, ax_video.get_position().width, 0.04])
    ax_slider.set_facecolor(BACKGROUND_COLOR)
    frame_slider = Slider(
        ax=ax_slider,
        label='',
        valmin=search_around_frames[cam_index][0],
        valmax=search_around_frames[cam_index][1],
        valinit=frame_number,
        valstep=1,
        valfmt=None 
    )

    frame_slider.poly.set_edgecolor(SLIDER_EDGE_COLOR)
    frame_slider.poly.set_facecolor(SLIDER_COLOR)
    frame_slider.poly.set_linewidth(1)
    frame_slider.valtext.set_visible(False)

    # Add highlight for time range around max speed
    range_start = max(frame_number - time_range_around_maxspeed * fps, search_around_frames[cam_index][0])
    range_end = min(frame_number + time_range_around_maxspeed * fps, search_around_frames[cam_index][1])
    highlight = ax_slider.axvspan(range_start, range_end, 
                                  ymin=0.20, ymax=0.80,
                                  color=SLIDER_HIGHLIGHT_COLOR, alpha=0.5, zorder=4)

    # Save highlight for later updates
    controls = {'range_highlight': highlight}
    controls['frame_slider'] = frame_slider

    # Calculate positions for UI elements
    controls_y = Y_POSITION
    lower_controls_y = controls_y - 0.05  # Y-coordinate for lower controls
    
    # Create person textbox (centered)
    controls['person_textbox'] = create_textbox(
        [0.5 - TEXTBOX_WIDTH/2 + 0.17, controls_y, TEXTBOX_WIDTH, CONTROL_HEIGHT],
        f"{cam_name}: Synchronize on person number",
        '0',
        {'colors': {'background': BACKGROUND_COLOR, 'text': TEXT_COLOR, 'control': CONTROL_COLOR, 'control_hover': CONTROL_HOVER_COLOR},
         'sizes': {'label': LABEL_SIZE_PERSON, 'text': TEXT_SIZE}}
    )

    # Create main time textbox (lower left)
    controls['main_time_textbox'] = create_textbox(
        [0.5 - TEXTBOX_WIDTH/2 - 0.05, lower_controls_y, TEXTBOX_WIDTH, CONTROL_HEIGHT],
        'around time',
        f"{frame_number / fps:.2f}",
        {'colors': {'background': BACKGROUND_COLOR, 'text': TEXT_COLOR, 'control': CONTROL_COLOR, 'control_hover': CONTROL_HOVER_COLOR},
         'sizes': {'label': LABEL_SIZE_PERSON, 'text': TEXT_SIZE}}
    )

    # Create time RAM textbox (lower center)
    controls['time_RAM_textbox'] = create_textbox(
        [0.5 - TEXTBOX_WIDTH/2 + 0.07, lower_controls_y, TEXTBOX_WIDTH, CONTROL_HEIGHT],
        '±',
        f"{time_range_around_maxspeed:.2f}",
        {'colors': {'background': BACKGROUND_COLOR, 'text': TEXT_COLOR, 'control': CONTROL_COLOR, 'control_hover': CONTROL_HOVER_COLOR},
         'sizes': {'label': LABEL_SIZE_PERSON, 'text': TEXT_SIZE}}
    )
    
    # Create OK button (lower right)
    ok_ax = plt.axes([0.5 - TEXTBOX_WIDTH/2 + 0.17, lower_controls_y, BTN_WIDTH_PERSON * 1.5, CONTROL_HEIGHT])
    ok_ax.set_facecolor(CONTROL_COLOR)
    controls['btn_ok'] = Button(
        ok_ax, 
        label='OK', 
        color=CONTROL_COLOR,
        hovercolor=CONTROL_HOVER_COLOR
    )
    controls['btn_ok'].label.set_color(TEXT_COLOR)
    controls['btn_ok'].label.set_fontsize(BUTTON_SIZE)
    controls['btn_ok'].label.set_fontweight('bold')
    
    # Initialize containers for dynamic elements
    containers = {
        'rects': [],
        'annotations': [],
        'bounding_boxes_list': [],
        'selected_idx': [0]
    }

    # Create UI dictionary
    ui = {
        'fig': fig,
        'ax_video': ax_video,
        'controls': controls,
        'containers': containers,
        'axes': {'slider': ax_slider}
    }

    # Connect hover event
    fig.canvas.mpl_connect('motion_notify_event', 
        lambda event: on_hover(event, fig, containers['rects'], 
                             containers['annotations'], 
                             containers['bounding_boxes_list'],
                             containers['selected_idx']))

    # Connect event handlers using lambda
    frame_slider.on_changed(lambda val: on_slider_change(val, fps, controls, fig, search_around_frames, cam_index, ax_slider))
    controls['main_time_textbox'].on_submit(lambda text: update_main_time(text, fps, search_around_frames, cam_index, ui, ui['cap'], frame_to_json, pose_dir, json_dirs_names, containers['bounding_boxes_list']))
    controls['time_RAM_textbox'].on_submit(lambda text: update_time_RAM(text, fps, search_around_frames, cam_index, ui))

    return ui


def select_person(vid_or_img_files, cam_names, json_files_names_range, search_around_frames, pose_dir, json_dirs_names, keypoints_names, keypoints_to_consider, time_range_around_maxspeed, fps):
    '''
    This function manages the process of selecting keypoints and persons for each camera.
    It performs two main steps:
    1. Select keypoints to consider for all cameras
    2. For each camera, select a person ID and a specific frame

    INPUTS:
    - vid_or_img_files: List of video files or image directories
    - cam_names: List of camera names
    - json_files_names_range: Range of JSON file names for each camera
    - search_around_frames: Frame ranges to search around for each camera
    - pose_dir: Directory containing pose data
    - json_dirs_names: Names of JSON directories for each camera
    - keypoints_names: Names of keypoints to consider
    - time_range_around_maxspeed: Time range to consider around max speed
    - fps: Frames per second of the videos

    OUTPUTS:
    - selected_id_list: List of selected person IDs for each camera
    - keypoints_to_consider: List of keypoints selected for consideration
    - approx_time_maxspeed: List of approximate times of maximum speed for each camera
    - time_RAM_list: List of time ranges to consider around max speed times for each camera
    '''
    
    # Step 1
    selected_keypoints = keypoints_ui(keypoints_to_consider, keypoints_names)
    if len(selected_keypoints) == 0:
        logging.warning('Synchronization requires to select at least one reference keypoint, none were selected. Selecting all of them.')
    else:
        logging.info(f'Selected keypoints: {selected_keypoints}')
    
    # Step 2
    selected_id_list = []
    approx_time_maxspeed = []
    keypoints_to_consider = selected_keypoints
    time_RAM_list = []
    
    try: # video files
        video_files_dict = {cam_name: file for cam_name in cam_names for file in vid_or_img_files if cam_name in os.path.basename(file)}
        print(f'video_files_dict: {video_files_dict}')
    except: # image directories
        video_files_dict = {cam_name: files for cam_name in cam_names for files in vid_or_img_files if cam_name in os.path.basename(files[0])}

    for i, cam_name in enumerate(cam_names):
        vid_or_img_files_cam = video_files_dict.get(cam_name)
        if not vid_or_img_files_cam:
            logging.warning(f'No video file nor image directory found for camera {cam_name}')
            selected_id_list.append(None)
            time_RAM_list.append(time_range_around_maxspeed)  # Use default value for missing cameras
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
            time_RAM_list.append(time_range_around_maxspeed)  # Use default value for missing cameras
            if isinstance(cap, cv2.VideoCapture):
                cap.release()
            continue
        
        # Initialize UI for person/frame selection only (no keypoint selection)
        ui = person_ui(frame_rgb, cam_name, frame_number, search_around_frames, time_range_around_maxspeed, fps, i, frame_to_json, pose_dir, json_dirs_names)
        ui['cap'] = cap
        
        # Draw initial bounding boxes
        draw_bounding_boxes_and_annotations(ui['ax_video'], bounding_boxes_list, 
                                          ui['containers']['rects'], 
                                          ui['containers']['annotations'])
        ui['containers']['bounding_boxes_list'] = bounding_boxes_list 
        ui['controls']['frame_slider'].on_changed(lambda val: update_frame(val, fps, ui, frame_to_json, pose_dir, json_dirs_names, i, search_around_frames, bounding_boxes_list))
        
        # Update main time textbox to also update slider
        ui['controls']['main_time_textbox'].on_submit(lambda text: update_main_time(text, fps, search_around_frames, i, ui, ui['cap'], frame_to_json, pose_dir, json_dirs_names, ui['containers']['bounding_boxes_list']))
        
        # Update time_RAM textbox to update highlight
        ui['controls']['time_RAM_textbox'].on_submit(lambda text: update_time_RAM(text, fps, search_around_frames, i, ui))

        # Add click event handler
        ui['fig'].canvas.mpl_connect('button_press_event', 
            lambda event: on_click(event, ui['ax_video'], bounding_boxes_list, 
                                 ui['containers']['selected_idx'], ui['controls']['person_textbox']))

        # Event handlers connection
        ui['controls']['person_textbox'].on_submit(
            lambda text: handle_person_change(text, ui['containers']['selected_idx'], ui['controls']['person_textbox']))

        # OK button
        btn_ok = ui['controls']['btn_ok']
        btn_ok.on_clicked(lambda event: handle_ok_button(ui))

        # Keyboard navigation
        ui['fig'].canvas.mpl_connect('key_press_event', lambda event: handle_key_press(event, ui['controls']['main_time_textbox'],
                              search_around_frames, i, ui['cap'], ui['ax_video'], frame_to_json, pose_dir,
                              json_dirs_names[i], ui['containers']['rects'], ui['containers']['annotations'], bounding_boxes_list, ui['fig'],
                              time_range_around_maxspeed, fps, ui))

        # Show plot and wait for user input
        plt.show()
        cap.release()

        # Store selected values after OK button is clicked
        selected_id_list.append(int(ui['controls']['person_textbox'].text))
        current_frame = int(round(float(ui['controls']['main_time_textbox'].text) * fps))
        approx_time_maxspeed.append(current_frame / fps)
        current_time_RAM = float(ui['controls']['time_RAM_textbox'].text)
        time_RAM_list.append(current_time_RAM)  # Store the time_RAM for this camera
        logging.info(f'--> Camera #{i}: selected person #{ui["controls"]["person_textbox"].text} at time {current_frame / fps:.2f} ± {current_time_RAM:.2f} s')

    return selected_id_list, keypoints_to_consider, approx_time_maxspeed, time_RAM_list


# SYNC FUNCTIONS
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


def calculate_mae(keypoints1, keypoints2, likelihood_threshold=0.6):
    """
    Calculate the Mean Absolute Error (MAE) between two sets of keypoints.
    Only considers valid points (non-zero coordinates and likelihood above threshold).

    INPUTS:
    - keypoints1: np.array (N, 3). Keypoints for the first person (x, y, likelihood).
    - keypoints2: np.array (N, 3). Keypoints for the second person (x, y, likelihood).
    - likelihood_threshold: float. Minimum confidence score for a keypoint to be considered valid.

    OUTPUTS:
    - mae: float. Mean Absolute Error between the valid keypoints. Returns infinity if no valid pairs found.
    """
    # Extract coordinates and likelihoods
    coords1 = keypoints1[:, :2]
    likes1 = keypoints1[:, 2]
    coords2 = keypoints2[:, :2]
    likes2 = keypoints2[:, 2]

    # Identify valid points based on non-zero coordinates and likelihood threshold
    valid1 = (np.all(coords1 != 0, axis=1)) & (likes1 > likelihood_threshold)
    valid2 = (np.all(coords2 != 0, axis=1)) & (likes2 > likelihood_threshold)
    
    # Find common valid points
    common_valid = valid1 & valid2
    
    if not np.any(common_valid):
        return float('inf')  # No common valid keypoints to compare

    # Calculate MAE only on common valid points
    diff = np.abs(coords1[common_valid] - coords2[common_valid])
    mae = np.mean(diff)
    
    return mae


def convert_json2pandas(json_files, likelihood_threshold=0.3, keypoints_ids=[],
                        synchronization_gui=False, selected_id=None, mae_threshold= 80,
                        animation=True, mae_keypoint_indices=None): # Added mae_keypoint_indices
    '''
    Convert a list of JSON files to a pandas DataFrame.
    If synchronization_gui is True, tracks the selected person using MAE.
    Otherwise, uses the person with the largest bounding box (original behavior).

    INPUTS:
    - json_files: list of str. Paths of the the JSON files.
    - likelihood_threshold: float. Drop values if confidence is below likelihood_threshold.
    - keypoints_ids: list of int. Indices of the keypoints to extract (relative to the original model).
    - synchronization_gui: bool. Flag indicating if the GUI was used for selection.
    - selected_id: int or None. The index of the person selected in the GUI's first frame.
    - mae_threshold: float. Maximum MAE allowed for tracking a person between frames.
    - animation: bool. Whether to generate tracking animation.
    - mae_keypoint_indices: list of int or None. Indices (within the 0-based index of people_kps rows, corresponding to keypoints_ids order) of keypoints to use for MAE calculation. If None or empty, use all.


    OUTPUTS:
    - df_json_coords: dataframe. Extracted coordinates in a pandas dataframe.
    '''

    nb_keypoints_total = max(keypoints_ids) + 1 if keypoints_ids else 0 # Determine based on max id
    nb_coords = len(keypoints_ids)
    json_coords_list = []
    
    pre_tracking_data_list = [] # Store original 'people' data for animation
    data_to_track_np = None # Holds the keypoints (N, 3) of the person being tracked from the previous frame
    tracking_initialized = False
    currently_tracked_person_id = None # 현재 추적 중인 사람의 ID 저장

    if not synchronization_gui:
        # --- Original Non-GUI Logic: Use largest bounding box ---
        for j_p in json_files:
            with open(j_p) as j_f:
                try:
                    json_data_all = json.load(j_f)['people']
                    pre_tracking_data_list.append(json_data_all if json_data_all else []) # Store for animation
                    if not json_data_all:
                        json_data = [np.nan] * nb_coords * 3
                    else:
                        bbox_area = []
                        valid_people_data = []
                        for p in json_data_all:
                            if 'pose_keypoints_2d' in p and len(p['pose_keypoints_2d']) >= nb_keypoints_total * 3:
                                kps_all = np.array(p['pose_keypoints_2d']).reshape(-1, 3)
                                kps_relevant = kps_all[keypoints_ids] # Select only relevant keypoints
                                valid_kps = kps_relevant[(kps_relevant[:, 2] > likelihood_threshold) & np.all(kps_relevant[:, :2] != 0, axis=1)]
                                if valid_kps.shape[0] >= 2: # Need at least 2 valid points for bbox
                                    x_min, y_min = valid_kps[:, :2].min(axis=0)
                                    x_max, y_max = valid_kps[:, :2].max(axis=0)
                                    area = (x_max - x_min) * (y_max - y_min)
                                    bbox_area.append(area)
                                    valid_people_data.append(kps_relevant)
                                else:
                                    bbox_area.append(0)
                                    valid_people_data.append(None) # Placeholder
                            else:
                                bbox_area.append(0)
                                valid_people_data.append(None) # Placeholder

                        if not bbox_area or max(bbox_area) == 0:
                            json_data = [np.nan] * nb_coords * 3
                        else:
                            max_area_idx = np.argmax(bbox_area)
                            json_data_np = valid_people_data[max_area_idx]
                            # Apply likelihood threshold
                            json_data_np = np.array([kp if kp[2] > likelihood_threshold else [np.nan, np.nan, np.nan] for kp in json_data_np])
                            json_data = json_data_np.ravel().tolist()
                except Exception as e:
                    # logging.warning(f"Error processing {os.path.basename(j_p)} in non-GUI mode: {e}")
                    json_data = [np.nan] * nb_coords * 3
                    if not pre_tracking_data_list or len(pre_tracking_data_list) <= json_files.index(j_p): # Ensure list has place for this frame
                        pre_tracking_data_list.append([]) # Add empty list if error occurred before storing
            json_coords_list.append(json_data)

    else:
        # --- New GUI Logic: Track selected person using MAE ---
        initial_selected_id = selected_id # Store the initially selected ID
        print(f'initial_selected_id: {initial_selected_id}')
        
        for frame_idx, j_p in enumerate(json_files):
            current_frame_tracked = False
            frame_basename = os.path.basename(j_p)  # 파일 이름 추출
            with open(j_p) as j_f:
                try:
                    json_data_all = json.load(j_f)['people']
                    pre_tracking_data_list.append(json_data_all if json_data_all else []) # Store for animation
                    
                    if not json_data_all:
                        # No people detected in this frame
                        if tracking_initialized and currently_tracked_person_id is not None:
                            logging.warning(f"프레임 {frame_idx} ({frame_basename}): 사람 ID {currently_tracked_person_id} 추적 손실. 프레임에서 사람이 감지되지 않음.")
                        json_coords_list.append([np.nan] * nb_coords * 3)
                        data_to_track_np = None # Lost track
                        currently_tracked_person_id = None
                        continue

                    people_kps = [] # Store keypoints (N, 3) for all people in this frame
                    people_indices = [] # Store original index of people
                    for idx, p in enumerate(json_data_all):
                         if 'pose_keypoints_2d' in p and len(p['pose_keypoints_2d']) >= nb_keypoints_total * 3:
                             kps_person_all = np.array(p['pose_keypoints_2d']).reshape(-1, 3)
                             # Ensure kps_person_all has enough keypoints before indexing
                             if kps_person_all.shape[0] > max(keypoints_ids):
                                 kps_person_relevant = kps_person_all[keypoints_ids]
                                 people_kps.append(kps_person_relevant)
                                 people_indices.append(idx)
                             else:
                                 logging.debug(f"Frame {frame_idx} ({frame_basename}): Person {idx} has insufficient keypoints ({kps_person_all.shape[0]}) required by keypoints_ids (max: {max(keypoints_ids)}). Skipping person.")
                         # else: Person data is invalid or incomplete

                    if not people_kps: # No valid people found after filtering
                         if tracking_initialized and currently_tracked_person_id is not None:
                             logging.warning(f"프레임 {frame_idx} ({frame_basename}): 사람 ID {currently_tracked_person_id} 추적 손실. 유효한 키포인트가 없음.")
                         json_coords_list.append([np.nan] * nb_coords * 3)
                         data_to_track_np = None # Lost track
                         currently_tracked_person_id = None
                         continue

                    if not tracking_initialized:
                        # Try to initialize tracking with the initially selected ID
                        if initial_selected_id is not None and initial_selected_id in people_indices:
                             selected_person_frame_idx = people_indices.index(initial_selected_id)
                             data_to_track_np = people_kps[selected_person_frame_idx]
                             # Apply likelihood threshold to the initial tracked data
                             tracked_kps = np.array([kp if kp[2] > likelihood_threshold else [np.nan, np.nan, np.nan] for kp in data_to_track_np])
                             json_coords_list.append(tracked_kps.ravel().tolist())
                             tracking_initialized = True
                             currently_tracked_person_id = initial_selected_id  # 추적 시작 시 ID 설정
                             current_frame_tracked = True
                             logging.info(f"프레임 {frame_idx} ({frame_basename}): 사람 ID {currently_tracked_person_id} 추적 시작.")
                        # else: Cannot initialize yet, wait for a frame where the selected ID is present

                    elif data_to_track_np is not None:
                        # --- Perform MAE-based tracking ---
                        mae_list = []
                        for kps_candidate in people_kps:
                             # Filter keypoints before calculating MAE if indices are provided
                             if mae_keypoint_indices: # Check if list is not None and not empty
                                # print(f"mae_keypoint_indices: {mae_keypoint_indices}")
                                try:
                                     # Ensure indices are within bounds for both arrays
                                     if max(mae_keypoint_indices) < data_to_track_np.shape[0] and max(mae_keypoint_indices) < kps_candidate.shape[0]:
                                         keypoints1_filtered = data_to_track_np[mae_keypoint_indices, :]
                                         keypoints2_filtered = kps_candidate[mae_keypoint_indices, :]
                                         # Ensure we still have points to compare after filtering
                                         if keypoints1_filtered.shape[0] > 0 and keypoints2_filtered.shape[0] > 0:
                                             mae = calculate_mae(keypoints1_filtered, keypoints2_filtered, likelihood_threshold)
                                         else:
                                             mae = float('inf') # Cannot compare if filtering removed all points
                                     else:
                                         logging.error(f"Frame {frame_idx} ({frame_basename}): MAE keypoint indices out of bounds. Using all keypoints.")
                                         mae = calculate_mae(data_to_track_np, kps_candidate, likelihood_threshold)
                                except IndexError:
                                     logging.error(f"Frame {frame_idx} ({frame_basename}): IndexError during MAE keypoint filtering. Using all keypoints for this comparison.")
                                     mae = calculate_mae(data_to_track_np, kps_candidate, likelihood_threshold)
                             else: # Original behavior: use all extracted keypoints
                                 mae = calculate_mae(data_to_track_np, kps_candidate, likelihood_threshold)
                             mae_list.append(mae)

                        # if not mae_list or min(mae_list) == float('inf'): # Original check
                        if not mae_list or np.all(np.isinf(mae_list)): # Check if all comparisons resulted in inf
                            # No valid comparisons possible
                            if currently_tracked_person_id is not None:
                                logging.warning(f"프레임 {frame_idx} ({frame_basename}): 사람 ID {currently_tracked_person_id} 추적 손실. 유효한 MAE 비교가 불가능.")
                            data_to_track_np = None # Lost track
                            currently_tracked_person_id = None
                        else:
                             # Find the index of the minimum *finite* MAE value
                             finite_mae_indices = [idx for idx, m in enumerate(mae_list) if not np.isinf(m)]
                             if not finite_mae_indices:
                                 # All were infinite, handled above, but for safety:
                                 logging.warning(f"프레임 {frame_idx} ({frame_basename}): 사람 ID {currently_tracked_person_id} 추적 손실. 모든 MAE 비교가 무한대.")
                                 data_to_track_np = None
                                 currently_tracked_person_id = None
                             else:
                                 min_finite_mae_local_idx = np.argmin([mae_list[i] for i in finite_mae_indices])
                                 min_mae_idx = finite_mae_indices[min_finite_mae_local_idx] # Index in original people_kps/people_indices
                                 min_mae = mae_list[min_mae_idx]
                                 new_person_id = people_indices[min_mae_idx]  # 새로운 사람 ID

                                 if min_mae <= mae_threshold:
                                     # Found a match, update tracked data
                                     if currently_tracked_person_id is not None and new_person_id != currently_tracked_person_id:
                                         logging.warning(f"프레임 {frame_idx} ({frame_basename}): 추적 대상이 ID {currently_tracked_person_id}에서 ID {new_person_id}로 변경됨 (MAE: {min_mae:.2f}).")
                                     
                                     data_to_track_np = people_kps[min_mae_idx]
                                     currently_tracked_person_id = new_person_id  # 추적 중인 ID 업데이트
                                     # Apply likelihood threshold before saving
                                     tracked_kps = np.array([kp if kp[2] > likelihood_threshold else [np.nan, np.nan, np.nan] for kp in data_to_track_np])
                                     json_coords_list.append(tracked_kps.ravel().tolist())
                                     current_frame_tracked = True
                                 else:
                                     # MAE too high, tracking lost
                                     # Include the potential ID in the log message
                                     logging.warning(f"프레임 {frame_idx} ({frame_basename}): 사람 ID {currently_tracked_person_id} 추적 손실. 최소 MAE ({min_mae:.2f} -> ID {new_person_id})가 임계값({mae_threshold:.2f})을 초과.")
                                     data_to_track_np = None
                                     currently_tracked_person_id = None
                except Exception as e:
                    logging.error(f"프레임 {frame_idx} ({frame_basename}) 처리 중 오류: {e}")
                    if tracking_initialized and currently_tracked_person_id is not None:
                        logging.warning(f"프레임 {frame_idx} ({frame_basename}): 오류로 인해 사람 ID {currently_tracked_person_id} 추적 손실.")
                    data_to_track_np = None # Lost track
                    currently_tracked_person_id = None
                    if not pre_tracking_data_list or len(pre_tracking_data_list) <= frame_idx: # Ensure list has place for this frame
                        pre_tracking_data_list.append([]) # Add empty list if error occurred before storing

            # If tracking was not initialized or lost in this frame, append NaNs
            if not current_frame_tracked:
                 json_coords_list.append([np.nan] * nb_coords * 3)
                 # Keep data_to_track_np as None until re-initialized or matched again

    # --- Final DataFrame creation (common for both modes) ---
    df_json_coords = pd.DataFrame(json_coords_list)

    # Optional: Add animation call here if desired
    if animation:
        try:
            post_tracking_data_np = np.array(json_coords_list)
            folder_name = os.path.basename(os.path.dirname(json_files[0])) if json_files else "Unknown"
            # Check if function exists and dependencies are met before calling
            if 'animate_pre_post_tracking' in globals():
                 animate_pre_post_tracking(pre_tracking_data_list, post_tracking_data_np, folder_name=folder_name)
            else:
                 logging.warning("Tracking animation function not found or dependencies missing.")
        except ImportError:
            logging.warning("Could not import animation dependencies (matplotlib, scipy). Skipping tracking animation.")
        except Exception as e:
            logging.warning(f"Could not generate tracking animation: {e}")


    if df_json_coords.isnull().all().all():
        logging.error('No valid coordinates found after processing/tracking. Check JSON files, selected person, likelihood threshold, and MAE threshold.')
        raise ValueError('No valid coordinates found after processing/tracking.')

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
            f, ax = plt.subplots(2,1, num='Synchronizing cameras')
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
    # multi_person = config_dict.get('project').get('multi_person')
    fps =  config_dict.get('project').get('frame_rate')
    frame_range = config_dict.get('project').get('frame_range')
    display_sync_plots = config_dict.get('synchronization').get('display_sync_plots')
    keypoints_to_consider = config_dict.get('synchronization').get('keypoints_to_consider')
    approx_time_maxspeed = config_dict.get('synchronization').get('approx_time_maxspeed') 
    time_range_around_maxspeed = config_dict.get('synchronization').get('time_range_around_maxspeed')
    synchronization_gui = config_dict.get('synchronization').get('synchronization_gui')

    likelihood_threshold = config_dict.get('synchronization').get('likelihood_threshold')
    filter_cutoff = int(config_dict.get('synchronization').get('filter_cutoff'))
    filter_order = int(config_dict.get('synchronization').get('filter_order'))

    display_tracking_animation = config_dict.get('synchronization').get('display_tracking_animation')

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
    cam_names = []
    for j_dir in json_dirs:
        base_name = os.path.basename(j_dir)
        parts = base_name.split('_')
        cam_name_found = None
        for part in parts:
            if re.match(r'cam\d+', part):
                cam_name_found = part
                break
        if cam_name_found:
            cam_names.append(cam_name_found)
        else:
            # Fallback to original logic if 'cam<number>' pattern not found
            cam_names.append(parts[0]) 
            logging.warning(f"Could not find a camera name matching 'cam<number>' in '{base_name}'. Using '{parts[0]}' as camera name.")
    
    # frame range selection
    f_range = [[0, min([len(j) for j in json_files_names])] if frame_range==[] else frame_range][0]
    # json_files_names = [[j for j in json_files_cam if int(re.split(r'(\d+)',j)[-2]) in range(*f_range)] for json_files_cam in json_files_names]

    # Determine frames to consider for synchronization
    if isinstance(approx_time_maxspeed, list): # search around max speed
        logging.info(f'Synchronization is calculated around the times {approx_time_maxspeed} +/- {time_range_around_maxspeed} s.')

        if len(approx_time_maxspeed) == 1 and cam_nb > 1:
            approx_time_maxspeed *= cam_nb

        approx_frame_maxspeed = [int(fps * t) for t in approx_time_maxspeed]
        nb_frames_per_cam = [len(fnmatch.filter(os.listdir(os.path.join(json_dir)), '*.json')) for json_dir in json_dirs]

        search_around_frames = []
        for i, frame in enumerate(approx_frame_maxspeed):
            start_frame = max(int(frame - lag_range), 0)
            end_frame = min(int(frame + lag_range), nb_frames_per_cam[i] + f_range[0])

            if start_frame != frame - lag_range:
                logging.warning(f'Frame range start adjusted for camera {i}: {frame - lag_range} -> {start_frame}')
            if end_frame != frame + lag_range:
                logging.warning(f'Frame range end adjusted for camera {i}: {frame + lag_range} -> {end_frame}')

            search_around_frames.append([start_frame, end_frame])

    elif approx_time_maxspeed == 'auto': # search on the whole sequence (slower if long sequence)
        search_around_frames = [[f_range[0], f_range[0]+nb_frames_per_cam[i]] for i in range(cam_nb)]
        logging.info('Synchronization is calculated on the whole sequence. This may take a while.')
    else:
        raise ValueError('approx_time_maxspeed should be a list of floats or "auto"')
    
    if keypoints_to_consider == 'right':
        keypoints_to_consider = [keypoints_names[i] for i in range(len(keypoints_ids)) if keypoints_names[i].startswith('R') or keypoints_names[i].startswith('right')]
        logging.info(f'Keypoints used to compute the best synchronization offset: right side.')
    elif keypoints_to_consider == 'left':
        keypoints_to_consider = [keypoints_names[i] for i in range(len(keypoints_ids)) if keypoints_names[i].startswith('L') or keypoints_names[i].startswith('left')]
        logging.info(f'Keypoints used to compute the best synchronization offset: left side.')
    elif isinstance(keypoints_to_consider, list):
        logging.info(f'Keypoints used to compute the best synchronization offset: {keypoints_to_consider}.')
    elif keypoints_to_consider == 'all':
        keypoints_to_consider = [keypoints_names[i] for i in range(len(keypoints_ids))]
        logging.info(f'All keypoints are used to compute the best synchronization offset.')
    else:
        raise ValueError('keypoints_to_consider should be "all", "right", "left", or a list of keypoint names.\n\
                        If you specified keypoints, make sure that they exist in your pose_model.')
    logging.info(f'These keypoints are filtered with a Butterworth filter (cut-off frequency: {filter_cutoff} Hz, order: {filter_order}).')
    logging.info(f'They are removed when their likelihood is below {likelihood_threshold}.\n')

    # Extract, interpolate, and filter keypoint coordinates
    logging.info('Synchronizing...')
    df_coords = []
    b, a = signal.butter(int(filter_order/2), filter_cutoff/(fps/2), 'low', analog = False)
    json_files_names_range = [[j for j in json_files_cam if int(re.split(r'(\d+)',j)[-2]) in range(*frames_cam)] for (json_files_cam, frames_cam) in zip(json_files_names,search_around_frames)]
    
    if np.array([j==[] for j in json_files_names_range]).any():
        raise ValueError(f'No json files found within the specified frame range ({frame_range}) at the times {approx_time_maxspeed} +/- {time_range_around_maxspeed} s.')
    
    json_files_range = [[os.path.join(pose_dir, j_dir, j_file) for j_file in json_files_names_range[j]] for j, j_dir in enumerate(json_dirs_names)]
    
    # Initial calculation based on config (might be overwritten by GUI)
    kpt_indices = [i for i,k in zip(keypoints_ids, keypoints_names) if k in keypoints_to_consider]
    kpt_id_in_df = np.array([[keypoints_ids.index(k)*2,keypoints_ids.index(k)*2+1]  for k in kpt_indices]).ravel()
    
    # Define model_nodes here, needed later
    model_nodes = {node.id: node for _, _, node in RenderTree(model) if node.id is not None}

    # Handle manual selection if synchronization_gui is True
    # This call might update keypoints_to_consider
    if synchronization_gui:
        # Initial keypoints_names needed for the UI
        initial_keypoints_names_for_ui = [node.name for _, _, node in RenderTree(model) if node.id is not None]
        selected_id_list, keypoints_to_consider, approx_time_maxspeed, time_RAM_list = select_person(
            vid_or_img_files, cam_names, json_files_names_range, search_around_frames, 
            pose_dir, json_dirs_names, initial_keypoints_names_for_ui, keypoints_to_consider, time_range_around_maxspeed, fps)
        
        # Update kpt_indices and kpt_id_in_df based on GUI selection
        kpt_indices = [i for i, k in zip(keypoints_ids, keypoints_names) if k in keypoints_to_consider]
        kpt_id_in_df = np.array([[keypoints_ids.index(k)*2,keypoints_ids.index(k)*2+1] for k in kpt_indices]).ravel()

        # Calculate lag_ranges using time_RAM_list
        lag_ranges = [int(dt * fps) for dt in time_RAM_list]
        
        # Update search_around_frames if approx_time_maxspeed is a list
        if isinstance(approx_time_maxspeed, list):
            approx_frame_maxspeed = [int(fps * t) for t in approx_time_maxspeed]
            search_around_frames = [[int(a-lag_ranges[i]) if a-lag_ranges[i]>0 else 0, 
                                    int(a+lag_ranges[i]) if a+lag_ranges[i]<nb_frames_per_cam[i] else nb_frames_per_cam[i]+f_range[0]] 
                                    for i,a in enumerate(approx_frame_maxspeed)]
            
            # Recalculate json_files_names_range and json_files_range with updated search_around_frames
            json_files_names_range = [[j for j in json_files_cam if int(re.split(r'(\d+)',j)[-2]) in range(*frames_cam)] 
                                     for (json_files_cam, frames_cam) in zip(json_files_names,search_around_frames)]
            json_files_range = [[os.path.join(pose_dir, j_dir, j_file) for j_file in json_files_names_range[j]] 
                               for j, j_dir in enumerate(json_dirs_names)]
                               
    else:
        selected_id_list = [None] * cam_nb

    # ---- Calculate MAE indices AFTER potential GUI update ----
    # Calculate the indices within the keypoints_ids list (and thus within people_kps rows)
    # that correspond to the FINAL keypoints_to_consider for MAE calculation.
    mae_kpt_indices_in_people_kps = [
        idx for idx, kpt_id in enumerate(keypoints_ids)
        if model_nodes.get(kpt_id) and model_nodes[kpt_id].name in keypoints_to_consider
    ]
    if not mae_kpt_indices_in_people_kps and isinstance(keypoints_to_consider, list): # Check if it was a specific list
         logging.warning(f"Could not find any specified 'keypoints_to_consider' ({keypoints_to_consider}) within the extracted 'keypoints_ids'. MAE tracking will use all extracted keypoints.")
         mae_kpt_indices_in_people_kps = None # Ensure it's None to trigger default behavior
    # ---------------------------------------------------------

    padlen = 3 * (max(len(a), len(b)) - 1)

    for i in range(cam_nb):
        # Pass the calculated indices and other params to convert_json2pandas
        df_coords.append(convert_json2pandas(
            json_files_range[i],
            likelihood_threshold=likelihood_threshold,
            keypoints_ids=keypoints_ids,
            synchronization_gui=synchronization_gui,
            selected_id=selected_id_list[i],
            mae_threshold=300, # Pass mae_threshold from config
            animation=display_tracking_animation, # Pass animation flag from config if needed, currently hardcoded in definition
            mae_keypoint_indices=mae_kpt_indices_in_people_kps # NEW ARGUMENT
            ))
        # Filter DataFrame columns based on FINAL keypoints_to_consider
        if kpt_id_in_df.size > 0: # Check if any keypoints are left after potential GUI update
            df_coords[i] = drop_col(df_coords[i],3) # drop likelihood
            df_coords[i] = df_coords[i][kpt_id_in_df]
        else:
            logging.error(f"Camera {i}: No keypoints selected or found based on 'keypoints_to_consider'. Cannot proceed with synchronization for this camera.")
            # Handle this case, e.g., skip camera or raise error
            # For now, let's create an empty DataFrame to avoid crashing later, but sync will fail.
            df_coords[i] = pd.DataFrame()
            continue # Skip filtering and processing for this camera if no keypoints

        df_coords[i] = df_coords[i].apply(interpolate_zeros_nans, axis=0, args=['linear'])
        df_coords[i] = df_coords[i].bfill().ffill()
        if df_coords[i].shape[0] > padlen:
            df_coords[i] = pd.DataFrame(signal.filtfilt(b, a, df_coords[i], axis=0))
        elif df_coords[i].shape[0] > 0: # Only warn if there was data to filter
            logging.warning(
                f"Camera {i}: insufficient number of samples ({df_coords[i].shape[0]} < {padlen + 1}) to apply the Butterworth filter. "
                "Data will remain unfiltered."
            )

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

        if sum_speeds[i].shape[0] > padlen:
            sum_speeds[i] = pd.DataFrame(signal.filtfilt(b, a, sum_speeds[i], axis=0)).squeeze()
        else:
            logging.warning(
                f"Camera {i}: insufficient number of samples ({sum_speeds[i].shape[0]} < {padlen + 1}) to apply the Butterworth filter. "
                "Data will remain unfiltered."
            )

    # Compute offset for best synchronization:
    # Highest correlation of sum of absolute speeds for each cam compared to reference cam
    ref_cam_id = nb_frames_per_cam.index(min(nb_frames_per_cam)) # ref cam: least amount of frames
    ref_cam_name = cam_names[ref_cam_id]
    ref_frame_nb = len(df_coords[ref_cam_id])
    lag_range = int(ref_frame_nb/2)
    cam_list.pop(ref_cam_id)
    cam_names.pop(ref_cam_id)
    offset = []
    logging.info('')
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
