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


# UI FUNCTIONS

# Global matplotlib settings - remove toolbar
plt.rcParams['toolbar'] = 'none'

def reset_styles(rect, annotation):
    '''
    Resets the styles of a rectangle and its annotation to default.
    '''

    rect.set_linewidth(1)
    rect.set_edgecolor('white')
    rect.set_facecolor((1, 1, 1, 0.1))
    annotation.set_fontsize(7)
    annotation.set_fontweight('normal')


def create_textbox(ax_pos, label, initial, UI_PARAMS):
    '''
    Helper function to create a textbox with consistent styling
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
def handle_ok_button():
    '''
    Handle OK button click. Closes the window and confirms the selection.
    '''
    plt.close()


def handle_person_change(text, selected_idx_container, person_textbox):
    '''
    Handle changes to the person selection text box.
    '''
    try:
        selected_idx_container[0] = int(text)
    except ValueError:
        person_textbox.set_val('0')
        selected_idx_container[0] = 0


def handle_frame_change(text, frame_number, frame_textbox, cap, ax_video, frame_to_json, 
                        pose_dir, json_dir_name, rects, annotations, bounding_boxes_list, 
                        fig, search_around_frames, i, time_range_around_maxspeed, fps, ui):
    '''
    Handle changes to the frame number text box.
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
        
        # Update slider and highlight using ui
        ui['controls']['frame_slider'].set_val(frame_number)
        range_start = max(frame_number - time_range_around_maxspeed * fps, search_around_frames[i][0])
        range_end = min(frame_number + time_range_around_maxspeed * fps, search_around_frames[i][1])
        vertices = np.array([[range_start, 0.20], [range_start, 0.80], 
                             [range_end, 0.80], [range_end, 0.20], [range_start, 0.20]])
        ui['controls']['range_highlight'].xy = vertices
        fig.canvas.draw_idle()


def handle_prev_frame(frame_textbox, search_around_frames, i, cap, ax_video, frame_to_json,
                      pose_dir, json_dir_name, rects, annotations, bounding_boxes_list, fig,
                      time_range_around_maxspeed, fps, ui):
    time_val = float(frame_textbox.text.split(' ±')[0])
    current = round(time_val * fps)
    if current > search_around_frames[i][0]:
        prev_frame = current - 1
        handle_frame_change(None, prev_frame, frame_textbox, cap, ax_video, frame_to_json,
                            pose_dir, json_dir_name, rects, annotations, bounding_boxes_list,
                            fig, search_around_frames, i, time_range_around_maxspeed, fps, ui)


def handle_next_frame(frame_textbox, search_around_frames, i, cap, ax_video, frame_to_json,
                      pose_dir, json_dir_name, rects, annotations, bounding_boxes_list, fig,
                      time_range_around_maxspeed, fps, ui):
    time_val = float(frame_textbox.text.split(' ±')[0])
    current = round(time_val * fps)
    if current < search_around_frames[i][1]:
        next_frame = current + 1
        handle_frame_change(None, next_frame, frame_textbox, cap, ax_video, frame_to_json,
                            pose_dir, json_dir_name, rects, annotations, bounding_boxes_list,
                            fig, search_around_frames, i, time_range_around_maxspeed, fps, ui)


def handle_key_press(event, frame_textbox, search_around_frames, i, cap, ax_video, frame_to_json,
                     pose_dir, json_dir_name, rects, annotations, bounding_boxes_list, fig,
                     time_range_around_maxspeed, fps, ui):
    if event.key == 'left':
        handle_prev_frame(frame_textbox, search_around_frames, i, cap, ax_video, frame_to_json,
                          pose_dir, json_dir_name, rects, annotations, bounding_boxes_list, fig,
                          time_range_around_maxspeed, fps, ui)
    elif event.key == 'right':
        handle_next_frame(frame_textbox, search_around_frames, i, cap, ax_video, frame_to_json,
                          pose_dir, json_dir_name, rects, annotations, bounding_boxes_list, fig,
                          time_range_around_maxspeed, fps, ui)


def handle_toggle_labels(event, keypoint_texts, containers, btn_toggle):
    '''
    Handle toggle labels button click.
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
    Highlights a selected rectangle and its annotation with bold white style.
    '''

    rect.set_linewidth(2)
    rect.set_edgecolor('darkorange')
    rect.set_facecolor((1, 1, 1, 0.1))
    annotation.set_fontsize(8)
    annotation.set_fontweight('bold')


def highlight_hover_box(rect, annotation):
    '''
    Highlights a hovered rectangle and its annotation with yellow style.
    '''

    rect.set_linewidth(2)
    rect.set_edgecolor('darkorange')
    rect.set_facecolor((1, 1, 0, 0.2))
    annotation.set_fontsize(8)
    annotation.set_fontweight('bold')

## on_ family
def on_hover(event, fig, rects, annotations, bounding_boxes_list, selected_idx_container=None):
    '''
    Highlights the bounding box and annotation when the mouse hovers over a person in the plot.
    Maintains highlight on the selected person.
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


def on_pick_keypoint(event, keypoints_names, selected_keypoints, scatter, keypoints_to_consider_container, fig, selected_text):
    '''
    Handle keypoint selection when a point is clicked in the keypoint visualization.
    
    INPUTS:
    - event: The pick event from matplotlib
    - keypoints_names: list of str. Names of all keypoints
    - selected_keypoints: list of str. Currently selected keypoints
    - scatter: matplotlib scatter plot object
    - keypoints_to_consider_container: list containing the selected keypoints
    - fig: matplotlib figure object
    - selected_text: Text object to update selected keypoints display
    
    OUTPUTS:
    - None. Updates the scatter plot and keypoints container in place.
    '''
    ind = event.ind[0]
    keypoint = keypoints_names[ind]
    
    if keypoint in selected_keypoints:
        selected_keypoints.remove(keypoint)
    else:
        selected_keypoints.append(keypoint)
        
    # Update scatter plot colors
    scatter.set_facecolors([('darkorange' if n in selected_keypoints else 'blue') 
                          for n in keypoints_names])
    
    # Update container
    keypoints_to_consider_container[0] = selected_keypoints
    
    # Update selected keypoints display
    if selected_keypoints:
        # Create text with bold selected keypoints
        text_parts = []
        text_parts.append('Selected: ')
        for i, kp in enumerate(selected_keypoints):
            if i > 0:
                text_parts.append(', ')
            text_parts.append(f'$\\bf{{{kp}}}$')  # Use LaTeX bold formatting
        selected_text.set_text(''.join(text_parts))
    else:
        selected_text.set_text('Selected: None\nClick on keypoints to select them')
    
    # Redraw
    fig.canvas.draw_idle()


def update_highlight(central_frame, delta_time, fps, search_around_frames, cam_index, ax_slider, controls):
    """슬라이더의 하이라이트 범위를 업데이트합니다."""
    if 'range_highlight' in controls:
        controls['range_highlight'].remove()
    range_start = max(central_frame - delta_time * fps, search_around_frames[cam_index][0])
    range_end = min(central_frame + delta_time * fps, search_around_frames[cam_index][1])
    controls['range_highlight'] = ax_slider.axvspan(range_start, range_end, ymin=0.2, ymax=0.8, color='darkorange', alpha=0.5, zorder=4)


def on_slider_change(val, fps, controls, fig, search_around_frames, cam_index, ax_slider):
    """슬라이더 값이 변경될 때 호출됩니다."""
    frame_number = int(val)
    central_time = frame_number / fps
    controls['central_time_textbox'].set_val(f"{central_time:.2f}")
    try:
        delta_time = float(controls['delta_time_textbox'].text)
    except ValueError:
        delta_time = 0
    update_highlight(frame_number, delta_time, fps, search_around_frames, cam_index, ax_slider, controls)
    fig.canvas.draw_idle()


def on_central_time_submit(text, fps, controls, fig, search_around_frames, cam_index, ax_slider, frame_slider):
    """중앙 시간 텍스트박스에 값이 제출될 때 호출됩니다."""
    try:
        central_time = float(text)
        frame_number = int(round(central_time * fps))
        frame_number = max(search_around_frames[cam_index][0], min(frame_number, search_around_frames[cam_index][1]))
        frame_slider.set_val(frame_number)
        try:
            delta_time = float(controls['delta_time_textbox'].text)
        except ValueError:
            delta_time = 0
        update_highlight(frame_number, delta_time, fps, search_around_frames, cam_index, ax_slider, controls)
        fig.canvas.draw_idle()
    except ValueError:
        pass  # 유효하지 않은 입력 무시


def on_delta_time_submit(text, fps, search_around_frames, cam_index, controls, fig, ax_slider, frame_slider):
    """델타 시간 텍스트박스에 값이 제출될 때 호출됩니다."""
    try:
        delta_time = float(text)
        if delta_time < 0:
            delta_time = 0  # 음수가 되지 않도록 보장
        frame_number = int(frame_slider.val)
        update_highlight(frame_number, delta_time, fps, search_around_frames, cam_index, ax_slider, controls)
        fig.canvas.draw_idle()
    except ValueError:
        pass  # 유효하지 않은 입력 무시


def on_highlight_press(event, controls, fps, search_around_frames, cam_index, ax_slider, fig):
    """하이라이트 영역을 클릭했을 때 호출되는 함수"""
    if event.inaxes != ax_slider or event.button != 1:  # 왼쪽 버튼만 처리
        return
    
    highlight = controls.get('range_highlight')
    if not highlight:
        return
        
    # 클릭한 x 좌표
    click_x = event.xdata
    
    # 하이라이트의 현재 범위
    highlight_start = highlight.get_xy()[0][0]
    highlight_width = highlight.get_xy()[2][0] - highlight_start
    highlight_end = highlight_start + highlight_width
    
    # 클릭 위치가 하이라이트 영역의 가장자리(±5프레임) 안에 있는지 확인
    edge_tolerance = 5
    if abs(click_x - highlight_start) <= edge_tolerance:
        controls['dragging'] = 'left'
    elif abs(click_x - highlight_end) <= edge_tolerance:
        controls['dragging'] = 'right'
    else:
        controls['dragging'] = None
    
    if controls['dragging']:
        controls['drag_start_x'] = click_x

def on_highlight_motion(event, controls, fps, search_around_frames, cam_index, ax_slider, fig):
    """하이라이트 영역을 드래그할 때 호출되는 함수"""
    if not controls.get('dragging') or event.inaxes != ax_slider:
        return
        
    highlight = controls.get('range_highlight')
    if not highlight:
        return
    
    # 현재 중앙 프레임
    current_frame = int(float(controls['central_time_textbox'].text) * fps)
    
    # 드래그 거리 계산
    drag_distance = event.xdata - controls['drag_start_x']
    
    # 현재 델타 시간
    try:
        current_delta = float(controls['delta_time_textbox'].text)
    except ValueError:
        current_delta = 0
    
    # 드래그에 따른 새로운 델타 시간 계산
    if controls['dragging'] == 'left':
        new_delta = current_delta - drag_distance / (2 * fps)  # 왼쪽을 드래그하면 범위가 줄어듦
    else:  # right
        new_delta = current_delta + drag_distance / (2 * fps)  # 오른쪽을 드래그하면 범위가 늘어남
    
    # 최소 범위 제한 (1초)
    new_delta = max(0.5, new_delta)
    
    # 텍스트박스 업데이트 및 하이라이트 다시 그리기
    controls['delta_time_textbox'].set_val(f"{new_delta:.2f}")
    update_highlight(current_frame, new_delta, fps, search_around_frames, cam_index, ax_slider, controls)
    fig.canvas.draw_idle()
    
    # 드래그 시작 위치 업데이트
    controls['drag_start_x'] = event.xdata

def on_highlight_release(event, controls, fps, search_around_frames, cam_index, ax_slider, fig):
    """마우스 버튼을 놓았을 때 호출되는 함수"""
    controls['dragging'] = None
    controls['drag_start_x'] = None


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


def select_keypoints(keypoints_names):
    # 텍스트 크기 정의
    TITLE_SIZE = 12
    LABEL_SIZE = 8
    BUTTON_SIZE = 10
    
    # figure 생성
    fig = plt.figure(figsize=(6, 8))
    fig.patch.set_facecolor('white')

    # 키포인트 선택 영역
    ax_keypoints = plt.axes([0.1, 0.2, 0.8, 0.7])
    ax_keypoints.set_facecolor('white')
    ax_keypoints.set_title('Select keypoints to synchronize on', fontsize=TITLE_SIZE, pad=10, color='black')
    
    # 모든 키포인트와 위치 정의
    all_keypoints = [
        'Hip', 'RHip', 'LHip', 'RShoulder', 'RElbow', 'RWrist', 'RKnee', 'RAnkle',
        'RSmallToe', 'RBigToe', 'RHeel', 'LShoulder', 'LElbow', 'LWrist', 'LKnee',
        'LAnkle', 'LSmallToe', 'LBigToe', 'LHeel'
    ]
    keypoints_positions = {
        'Hip': (0.50, 0.42), 'RHip': (0.42, 0.42), 'LHip': (0.58, 0.42),
        'RShoulder': (0.40, 0.75), 'RElbow': (0.35, 0.65), 'RWrist': (0.25, 0.50),
        'RKnee': (0.40, 0.25), 'RAnkle': (0.40, 0.05),
        'RSmallToe': (0.35, 0.0), 'RBigToe': (0.42, 0.0), 'RHeel': (0.40, 0.02),
        'LShoulder': (0.60, 0.75), 'LElbow': (0.65, 0.65), 'LWrist': (0.75, 0.50),
        'LKnee': (0.60, 0.25), 'LAnkle': (0.60, 0.05),
        'LSmallToe': (0.65, 0.0), 'LBigToe': (0.58, 0.0), 'LHeel': (0.60, 0.02)
    }
    
    # 키포인트 좌표 생성
    keypoints_x, keypoints_y = zip(*[keypoints_positions[name] for name in all_keypoints])
    
    # 초기 색상 설정
    initial_colors = ['blue' if kp in keypoints_names else 'silver' for kp in all_keypoints]
    
    # scatter 플롯 생성
    selected_keypoints = []
    scatter = ax_keypoints.scatter(keypoints_x, keypoints_y, c=initial_colors, picker=True)
    
    # 키포인트 라벨 추가
    keypoint_texts = [ax_keypoints.text(x + 0.02, y, name, va='center', fontsize=LABEL_SIZE, color='black', visible=False)
                      for x, y, name in zip(keypoints_x, keypoints_y, all_keypoints)]
    
    ax_keypoints.set_xlim(0, 1)
    ax_keypoints.set_ylim(-0.1, 1)
    ax_keypoints.axis('off')
    
    # 선택된 키포인트 표시 영역
    ax_selected = plt.axes([0.1, 0.08, 0.8, 0.04])
    ax_selected.axis('off')
    ax_selected.set_facecolor('black')
    selected_text = ax_selected.text(0.0, 0.5, 'Selected: None\nClick on keypoints to select them', 
                                    va='center', fontsize=BUTTON_SIZE, wrap=True, color='black')
    
    # 버튼 추가
    btn_width, btn_height, btn_y = 0.16, 0.04, 0.02
    center_x = 0.5
    btn_all_none = plt.Button(plt.axes([center_x - 1.5*btn_width - 0.01, btn_y, btn_width, btn_height]), 'Select All')
    btn_toggle = plt.Button(plt.axes([center_x - btn_width/2, btn_y, btn_width, btn_height]), 'Show names')
    btn_ok = plt.Button(plt.axes([center_x + 0.5*btn_width + 0.01, btn_y, btn_width, btn_height]), 'OK')
    
    def toggle_labels(event):
        show_labels = not keypoint_texts[0].get_visible()
        for text, name in zip(keypoint_texts, all_keypoints):
            text.set_visible(show_labels)
            text.set_fontweight('bold' if name in selected_keypoints else 'normal')
        btn_toggle.label.set_text('Hide names' if show_labels else 'Show names')
        plt.draw()
    
    btn_toggle.on_clicked(toggle_labels)
    btn_ok.on_clicked(lambda event: plt.close())
    
    def select_all_none(event):
        if selected_keypoints:
            selected_keypoints.clear()
        else:
            selected_keypoints.extend(keypoints_names)
        
        colors = [
            'darkorange' if kp in selected_keypoints else 'blue' if kp in keypoints_names else 'silver'
            for kp in all_keypoints
        ]
        scatter.set_facecolors(colors)
        
        for text, kp in zip(keypoint_texts, all_keypoints):
            text.set_fontweight('bold' if kp in selected_keypoints else 'normal')
        
        if selected_keypoints:
            text_parts = ['Selected: '] + [f'$\\bf{{{kp}}}$' if i == 0 else f', $\\bf{{{kp}}}$' for i, kp in enumerate(selected_keypoints)]
            selected_text.set_text(''.join(text_parts))
            btn_all_none.label.set_text('Select None')
        else:
            selected_text.set_text('Selected: None\nClick on keypoints to select them')
            btn_all_none.label.set_text('Select All')
        
        plt.draw()
    
    btn_all_none.on_clicked(select_all_none)
    
    for btn in [btn_all_none, btn_toggle, btn_ok]:
        btn.color = '#D3D3D3'  # Light gray
        btn.hovercolor = '#A9A9A9'  # Darker gray on hover
    
    def on_pick(event):
        ind = event.ind[0]
        keypoint = all_keypoints[ind]
        if keypoint in keypoints_names:
            if keypoint in selected_keypoints:
                selected_keypoints.remove(keypoint)
            else:
                selected_keypoints.append(keypoint)
            
            colors = [
                'darkorange' if kp in selected_keypoints else 'blue' if kp in keypoints_names else 'silver'
                for kp in all_keypoints
            ]
            scatter.set_facecolors(colors)
            
            for text, kp in zip(keypoint_texts, all_keypoints):
                text.set_fontweight('bold' if kp in selected_keypoints else 'normal')
            
            if selected_keypoints:
                text_parts = ['Selected: '] + [f'$\\bf{{{kp}}}$' if i == 0 else f', $\\bf{{{kp}}}$' for i, kp in enumerate(selected_keypoints)]
                selected_text.set_text(''.join(text_parts))
                btn_all_none.label.set_text('Select None')
            else:
                selected_text.set_text('Selected: None\nClick on keypoints to select them')
                btn_all_none.label.set_text('Select All')
            
            plt.draw()
    
    fig.canvas.mpl_connect('pick_event', on_pick)
    
    plt.show()
    
    return selected_keypoints


def init_person_selection_ui_step2(frame_rgb, cam_name, frame_number, search_around_frames, time_range_around_maxspeed, fps, cam_index):
    '''
    Step 2: 사람과 프레임 선택을 위한 UI 초기화 (키포인트 선택 제외)
    '''
    # UI 파라미터 정의
    UI_PARAMS = {
        'colors': {
            'background': 'white',
            'text': 'black',
            'control': '#D3D3D3',
            'control_hover': '#A9A9A9',
            'slider': '#4682B4',
            'slider_edge': (0.5, 0.5, 0.5, 0.5)
        },
        'sizes': {
            'label': 10,
            'text': 9.5,
            'button': 10
        },
        'layout': {
            'total_width': 0.6,
            'textbox_width': 0.09,
            'btn_width': 0.04,
            'btn_spacing': 0.01,
            'control_height': 0.04,
            'y_position': 0.1
        }
    }

    # 프레임 크기와 방향을 기반으로 UI 설정
    frame_height, frame_width = frame_rgb.shape[:2]
    is_vertical = frame_height > frame_width
    
    # 비디오 방향에 따라 적절한 피규어 높이 계산
    if is_vertical:
        fig_height = frame_height / 250  # 세로 비디오용
    else:
        fig_height = max(frame_height / 300, 6)  # 가로 비디오용
    
    fig = plt.figure(figsize=(8, fig_height))
    fig.patch.set_facecolor(UI_PARAMS['colors']['background'])

    # 비디오 방향에 따라 UI 레이아웃 조정
    video_axes_height = 0.7 if is_vertical else 0.6
    slider_y = 0.15 if is_vertical else 0.2
    controls_y = UI_PARAMS['layout']['y_position'] if is_vertical else 0.1
    lower_controls_y = controls_y - 0.05  # 아래쪽 컨트롤을 위한 y 좌표
    
    ax_video = plt.axes([0.1, 0.2, 0.8, video_axes_height])
    ax_video.imshow(frame_rgb)
    ax_video.axis('off')
    ax_video.set_facecolor(UI_PARAMS['colors']['background'])

    # 프레임 슬라이더 생성
    ax_slider = plt.axes([ax_video.get_position().x0, slider_y, ax_video.get_position().width, 0.04])
    ax_slider.set_facecolor(UI_PARAMS['colors']['background'])
    frame_slider = Slider(
        ax=ax_slider,
        label='',
        valmin=search_around_frames[cam_index][0],
        valmax=search_around_frames[cam_index][1],
        valinit=frame_number,
        valstep=1,
        valfmt=None 
    )

    frame_slider.poly.set_edgecolor(UI_PARAMS['colors']['slider_edge'])
    frame_slider.poly.set_facecolor(UI_PARAMS['colors']['slider'])
    frame_slider.poly.set_linewidth(1)
    frame_slider.valtext.set_visible(False)

    # 최대 속도 주변 시간 범위에 대한 하이라이트 추가
    range_start = max(frame_number - time_range_around_maxspeed * fps, search_around_frames[cam_index][0])
    range_end = min(frame_number + time_range_around_maxspeed * fps, search_around_frames[cam_index][1])
    highlight = ax_slider.axvspan(range_start, range_end, 
                                  ymin=0.20, ymax=0.80,
                                  color='darkorange', alpha=0.5, zorder=4)

    # 하이라이트를 나중에 업데이트할 수 있도록 저장
    controls = {'range_highlight': highlight}
    controls['frame_slider'] = frame_slider
    
    # UI 요소들의 위치 계산
    controls_y = UI_PARAMS['layout']['y_position']
    lower_controls_y = controls_y - 0.05  # 아래쪽 컨트롤을 위한 y 좌표
    
    # 사람 텍스트박스 생성 (중앙에 배치)
    controls['person_textbox'] = create_textbox(
        [0.5 - UI_PARAMS['layout']['textbox_width']/2 + 0.17, controls_y, UI_PARAMS['layout']['textbox_width'], UI_PARAMS['layout']['control_height']],
        f"{cam_name}: Synchronize on person number",
        '0',
        UI_PARAMS
    )

    # 중앙 시간 텍스트박스 생성 (아래줄 왼쪽)
    controls['central_time_textbox'] = create_textbox(
        [0.5 - UI_PARAMS['layout']['textbox_width']/2 - 0.05, lower_controls_y, UI_PARAMS['layout']['textbox_width'], UI_PARAMS['layout']['control_height']],
        'around time',
        f"{frame_number / fps:.2f}",
        UI_PARAMS
    )

    # 델타 시간 텍스트박스 생성 (아래줄 중앙)
    controls['delta_time_textbox'] = create_textbox(
        [0.5 - UI_PARAMS['layout']['textbox_width']/2 + 0.07, lower_controls_y, UI_PARAMS['layout']['textbox_width'], UI_PARAMS['layout']['control_height']],
        '±',
        f"{time_range_around_maxspeed:.2f}",
        UI_PARAMS
    )
    
    # OK 버튼 생성 (아래줄 오른쪽)
    ok_ax = plt.axes([0.5 - UI_PARAMS['layout']['textbox_width']/2 + 0.17, lower_controls_y, UI_PARAMS['layout']['btn_width'] * 1.5, UI_PARAMS['layout']['control_height']])
    ok_ax.set_facecolor(UI_PARAMS['colors']['control'])
    controls['btn_ok'] = Button(
        ok_ax, 
        'OK',
        color=UI_PARAMS['colors']['control'],
        hovercolor=UI_PARAMS['colors']['control_hover']
    )
    controls['btn_ok'].label.set_color(UI_PARAMS['colors']['text'])
    controls['btn_ok'].label.set_fontsize(UI_PARAMS['sizes']['button'])

    # 동적 요소를 위한 컨테이너 초기화
    containers = {
        'rects': [],
        'annotations': [],
        'bounding_boxes_list': [],
        'selected_idx': [0]
    }

    # 호버 이벤트 연결
    fig.canvas.mpl_connect('motion_notify_event', 
        lambda event: on_hover(event, fig, containers['rects'], 
                             containers['annotations'], 
                             containers['bounding_boxes_list'],
                             containers['selected_idx']))

    # 이벤트 핸들러를 lambda를 사용하여 연결
    frame_slider.on_changed(lambda val: on_slider_change(val, fps, controls, fig, search_around_frames, cam_index, ax_slider))
    controls['central_time_textbox'].on_submit(lambda text: on_central_time_submit(text, fps, controls, fig, search_around_frames, cam_index, ax_slider, frame_slider))
    controls['delta_time_textbox'].on_submit(lambda text: on_delta_time_submit(text, fps, search_around_frames, cam_index, controls, fig, ax_slider, frame_slider))

    # 하이라이트 드래그 이벤트 연결
    controls['dragging'] = None
    controls['drag_start_x'] = None
    fig.canvas.mpl_connect('button_press_event', 
        lambda event: on_highlight_press(event, controls, fps, search_around_frames, cam_index, ax_slider, fig))
    fig.canvas.mpl_connect('motion_notify_event', 
        lambda event: on_highlight_motion(event, controls, fps, search_around_frames, cam_index, ax_slider, fig))
    fig.canvas.mpl_connect('button_release_event', 
        lambda event: on_highlight_release(event, controls, fps, search_around_frames, cam_index, ax_slider, fig))

    # UI 구성 요소 반환
    return {
        'fig': fig,
        'ax_video': ax_video,
        'controls': controls,
        'containers': containers
    }


def select_person(vid_or_img_files, cam_names, json_files_names_range, search_around_frames, pose_dir, json_dirs_names, keypoints_names, time_range_around_maxspeed, fps):
    '''
    Step 1: Select keypoints to consider for all cameras
    Step 2: Select person ID and frame for each camera
    '''
    logging.info('Manual mode: selecting keypoints and then person/frame for each camera.')
    
    # Step 1
    selected_keypoints = select_keypoints(keypoints_names)
    logging.info(f'Selected keypoints for all cameras: {selected_keypoints}')
    
    # Step 2
    selected_id_list = []
    approx_time_maxspeed = []
    keypoints_to_consider = selected_keypoints
    
    try: # video files
        video_files_dict = {cam_name: file for cam_name in cam_names for file in vid_or_img_files if cam_name in os.path.basename(file)}
    except: # image directories
        video_files_dict = {cam_name: files for cam_name in cam_names for files in vid_or_img_files if cam_name in os.path.basename(files[0])}

    for i, cam_name in enumerate(cam_names):
        selected_idx_container = [0]
        
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
        
        # Initialize UI for person/frame selection only (no keypoint selection)
        ui = init_person_selection_ui_step2(frame_rgb, cam_name, frame_number, search_around_frames, time_range_around_maxspeed, fps, i)
        
        # Draw initial bounding boxes
        draw_bounding_boxes_and_annotations(ui['ax_video'], bounding_boxes_list, 
                                          ui['containers']['rects'], 
                                          ui['containers']['annotations'])
        ui['containers']['bounding_boxes_list'] = bounding_boxes_list
        
        # Add slider update handler
        def update_frame(val):
            frame_num = int(val)
            central_time = frame_num / fps
            ui['controls']['central_time_textbox'].set_val(f"{central_time:.2f}")
            
            # Update yellow highlight position
            try:
                delta_time = float(ui['controls']['delta_time_textbox'].text)
            except ValueError:
                delta_time = 0
            range_start = max(frame_num - delta_time * fps, search_around_frames[i][0])
            range_end = min(frame_num + delta_time * fps, search_around_frames[i][1])
            
            # Update highlight vertices
            vertices = np.array([
                [range_start, 0.20],  # Left bottom
                [range_start, 0.80],  # Left top
                [range_end, 0.80],    # Right top
                [range_end, 0.20],    # Right bottom
                [range_start, 0.20]   # Back to left bottom (closed polygon)
            ])
            ui['controls']['range_highlight'].xy = vertices
            
            # Update the video frame and bounding boxes
            update_play(cap, ui['ax_video'].images[0], frame_num, frame_to_json, 
                        pose_dir, json_dirs_names[i], ui['containers']['rects'], 
                        ui['containers']['annotations'], bounding_boxes_list, 
                        ui['ax_video'], ui['fig'])
            
            ui['fig'].canvas.draw_idle()
        
        ui['controls']['frame_slider'].on_changed(update_frame)
        
        # Update central time textbox to also update slider
        def update_central_time(text):
            try:
                central_time = float(text)
                frame_num = int(round(central_time * fps))
                frame_num = max(search_around_frames[i][0], min(frame_num, search_around_frames[i][1]))
                ui['controls']['frame_slider'].set_val(frame_num)
                update_frame(frame_num)
            except ValueError:
                pass
        
        ui['controls']['central_time_textbox'].on_submit(update_central_time)
        
        # Update delta time textbox to update highlight
        def update_delta_time(text):
            try:
                delta_time = float(text)
                if delta_time < 0:
                    delta_time = 0
                frame_num = int(ui['controls']['frame_slider'].val)
                range_start = max(frame_num - delta_time * fps, search_around_frames[i][0])
                range_end = min(frame_num + delta_time * fps, search_around_frames[i][1])
                vertices = np.array([
                    [range_start, 0.20],  # Left bottom
                    [range_start, 0.80],  # Left top
                    [range_end, 0.80],    # Right top
                    [range_end, 0.20],    # Right bottom
                    [range_start, 0.20]   # Back to left bottom (closed polygon)
                ])
                ui['controls']['range_highlight'].xy = vertices
                ui['fig'].canvas.draw_idle()
            except ValueError:
                pass
        
        ui['controls']['delta_time_textbox'].on_submit(update_delta_time)
        
        # Add click event handler
        ui['fig'].canvas.mpl_connect('button_press_event', 
            lambda event: on_click(event, ui['ax_video'], bounding_boxes_list, 
                                 ui['containers']['selected_idx'], ui['controls']['person_textbox']))

        # Event handlers connection
        ui['controls']['person_textbox'].on_submit(
            lambda text: handle_person_change(text, ui['containers']['selected_idx'], ui['controls']['person_textbox']))

        # OK button
        def handle_ok_button():
            try:
                central_time = float(ui['controls']['central_time_textbox'].text)
                delta_time = float(ui['controls']['delta_time_textbox'].text)
                current_frame = int(round(central_time * fps))
                selected_id_list.append(selected_idx_container[0])
                approx_time_maxspeed.append(current_frame / fps)
                logging.info(f"cam{i}: Synchronize on person number {selected_idx_container[0]}, around time {central_time:.2f} ± {delta_time:.2f}")
                plt.close(ui['fig'])
            except ValueError:
                logging.warning('Invalid input in textboxes.')

        btn_ok = ui['controls']['btn_ok']
        btn_ok.on_clicked(lambda event: handle_ok_button())

        # Keyboard navigation
        ui['fig'].canvas.mpl_connect('key_press_event', lambda event: handle_key_press(event, ui['controls']['central_time_textbox'],
                              search_around_frames, i, cap, ui['ax_video'], frame_to_json, pose_dir,
                              json_dirs_names[i], ui['containers']['rects'], ui['containers']['annotations'], bounding_boxes_list, ui['fig'],
                              time_range_around_maxspeed, fps, ui))

        # Show plot and wait for user input
        plt.show()
        cap.release()

        # Store selected values after OK button is clicked
        selected_id_list.append(selected_idx_container[0])
        current_frame = int(round(float(ui['controls']['central_time_textbox'].text) * fps))
        approx_time_maxspeed.append(current_frame / fps)
        logging.info(f'--> Camera #{i}: selected person #{selected_idx_container[0]} at time #{current_frame / fps:.2f}')
    
    return selected_id_list, keypoints_to_consider, approx_time_maxspeed


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
    synchronization_gui = config_dict.get('synchronization').get('synchronization_gui')

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
    
    # Handle manual selection if synchronization_gui is True
    if synchronization_gui:
        selected_id_list, keypoints_to_consider, approx_time_maxspeed = select_person(
            vid_or_img_files, cam_names, json_files_names_range, search_around_frames, 
            pose_dir, json_dirs_names, keypoints_names, time_range_around_maxspeed, fps)
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