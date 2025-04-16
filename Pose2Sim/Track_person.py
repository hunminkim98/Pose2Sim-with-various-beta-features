import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import ConvexHull
import re
from tqdm import tqdm

def find_json_folders(base_path):
    json_folders = []
    for root, dirs, files in os.walk(base_path):
        if any(file.endswith('.json') for file in files):
            json_folders.append(root)
    return json_folders

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def calculate_hull_area(points):
    if len(points) < 3:
        return 0
    try:
        hull = ConvexHull(points)
        return hull.area
    except Exception:
        return 0

def calculate_distance_from_center(points, image_size):
    center_x, center_y = image_size[0] / 2, image_size[1] / 2
    person_center = np.mean(points, axis=0)
    return np.sqrt((person_center[0] - center_x)**2 + (person_center[1] - center_y)**2)

def select_person_automatically(people, image_size):
    max_score = float('-inf')
    selected_person_idx = None
    
    for i, person in enumerate(people):
        keypoints = np.array(person['pose_keypoints_2d']).reshape(-1, 3)
        valid_points = keypoints[np.all(keypoints[:, :2] != 0, axis=1)][:, :2]
        
        if len(valid_points) < 3:
            continue
        
        hull_area = calculate_hull_area(valid_points)
        center_distance = calculate_distance_from_center(valid_points, image_size)
        
        # Normalize scores
        normalized_area = hull_area / (image_size[0] * image_size[1])
        normalized_distance = 1 - (center_distance / (np.sqrt(image_size[0]**2 + image_size[1]**2) / 2))
        
        # Calculate combined score (you can adjust weights if needed)
        score = normalized_area * 0.7 + normalized_distance * 0.3
        
        if score > max_score:
            max_score = score
            selected_person_idx = i
    
    return selected_person_idx

def select_person_manually(people):
    fig, ax = plt.subplots()
    person_patches = []

    for i, person in enumerate(people):
        keypoints = np.array(person['pose_keypoints_2d']).reshape(-1, 3)
        x_data = keypoints[:, 0]
        y_data = keypoints[:, 1]
        valid = (x_data != 0) & (y_data != 0)
        x_data = x_data[valid]
        y_data = y_data[valid]
        scat = ax.scatter(x_data, -y_data, label=f'Person {i+1}')
        ax.annotate(f'{i+1}', xy=(np.mean(x_data), -np.mean(y_data)), color='red', fontsize=12)
        person_patches.append((scat, i))

    ax.set_title('Click on the person you want to track or close the window to enter ID manually')
    ax.set_xlim([0, 4000])
    ax.set_ylim([-3000, 0])

    selected_person_idx = []

    def onclick(event):
        if event.inaxes == ax:
            x_click = event.xdata
            y_click = event.ydata
            min_dist = float('inf')
            selected_idx = None
            for scat, idx in person_patches:
                x_data = scat.get_offsets()[:, 0]
                y_data = scat.get_offsets()[:, 1]
                distances = np.sqrt((x_data - x_click)**2 + (y_data - y_click)**2)
                if len(distances) > 0:
                    dist = np.min(distances)
                    if dist < min_dist:
                        min_dist = dist
                        selected_idx = idx
            if selected_idx is not None:
                selected_person_idx.append(selected_idx)
                plt.close(fig)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if not selected_person_idx:  # If no person was selected by clicking
        while True:
            print(f"\nAvailable person IDs: {list(range(1, len(people) + 1))}")
            try:
                selected_id = int(input("Enter the ID of the person you want to track (or 0 to cancel): "))
                if selected_id == 0:
                    return None
                if 1 <= selected_id <= len(people):
                    return selected_id - 1
                else:
                    print("Invalid ID. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    else:
        return selected_person_idx[0]

def track_person(folder_path, mode='auto'):
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')], key=natural_sort_key)
    if not json_files:
        print(f"No files found in the specified directory: {folder_path}")
        return None, None, None, None

    detected = False
    right_person = False
    data_to_track = None
    pos1 = []
    pre_tracking_data = []
    total_min_avg = 0
    count_min_avg = 0
    keypoint_count = 0
    image_size = (3840, 2160)  # Assuming this is the image size, adjust if needed

    for i, file_name in tqdm(enumerate(json_files), total=len(json_files), desc="Processing files", ncols=100):
        data = read_json_file(os.path.join(folder_path, file_name))
        people = data.get('people', [])
        pre_tracking_data.append(people)

        if i == 0:
            if people:
                keypoint_count = len(people[0]['pose_keypoints_2d'])
                print(f"Detected {keypoint_count} keypoints in the first frame\n")
            else:
                print("No people detected in the first frame")
                keypoint_count = 75  # Default to 25 keypoints * 3 (x, y, confidence)

        current_pos = np.zeros(keypoint_count)

        if not detected and not right_person:
            if not people:
                pos1.append(current_pos)
            else:
                if mode == 'auto':
                    selected_person_idx = select_person_automatically(people, image_size)
                    if selected_person_idx is not None:
                        current_pos = np.array(people[selected_person_idx]['pose_keypoints_2d'])
                        data_to_track = current_pos
                        detected = True
                        right_person = True
                        print(f"Automatically selected person {selected_person_idx + 1}")
                    else:
                        print("No suitable person found for tracking")
                elif mode == 'manual':
                    selected_person_idx = select_person_manually(people)
                    if selected_person_idx is not None:
                        current_pos = np.array(people[selected_person_idx]['pose_keypoints_2d'])
                        data_to_track = current_pos
                        detected = True
                        right_person = True
                        print(f"Manually selected person {selected_person_idx + 1}")
                    else:
                        print("No person selected for tracking")
                        return None, None, None, None
                else:
                    print("Invalid mode selected")
                    return None, None, None, None
        elif detected and right_person:
            if people:
                mae = []
                for k, person in enumerate(people):
                    p1 = np.array(person['pose_keypoints_2d'])
                    x0, y0 = np.array(data_to_track[::3]), np.array(data_to_track[1::3])
                    x1, y1 = p1[::3], p1[1::3]
                    valid = np.where((x0 != 0) & (y0 != 0) & (x1 != 0) & (y1 != 0))[0]
                    if valid.size == 0:
                        x_mae, y_mae = float('inf'), float('inf')
                    else:
                        x_mae = np.mean(np.abs(x0[valid] - x1[valid]))
                        y_mae = np.mean(np.abs(y0[valid] - y1[valid]))
                    mae.append(np.mean([x_mae, y_mae]))
                min_avg, I1 = min((val, idx) for (idx, val) in enumerate(mae))
                if min_avg > 100:
                    detected = False
                    right_person = False
                else:
                    current_pos = np.array(people[I1]['pose_keypoints_2d'])
                    data_to_track = current_pos
                    total_min_avg += min_avg
                    count_min_avg += 1
            else:
                detected = False
                right_person = False

        # Adjust current_pos length if keypoint_count changes
        if len(current_pos) < keypoint_count:
            current_pos = np.pad(current_pos, (0, keypoint_count - len(current_pos)), 'constant')
        elif len(current_pos) > keypoint_count:
            keypoint_count = len(current_pos)
            # Pad previous frames' data to match new keypoint_count
            pos1 = [np.pad(p, (0, keypoint_count - len(p)), 'constant') for p in pos1]

        pos1.append(current_pos)

    avg_min_avg = total_min_avg / count_min_avg if count_min_avg > 0 else 0
    print(f"\nAverage min_avg: {avg_min_avg:.2f}")

    return np.array(pos1), json_files, pre_tracking_data, avg_min_avg

def animate_pre_post_tracking(pre_tracking_data, post_tracking_data, folder_name=None, frame_step=10, interval=100):
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

def save_data(data, json_files, folder_path):
    base_folder = os.path.dirname(folder_path)
    folder_name = os.path.basename(folder_path)
    tracked_folder_name = folder_name + '_tracked'
    save_folder = os.path.join(base_folder, tracked_folder_name)
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i, file_name in enumerate(json_files):
        frame_data = {
            "version": 1.3,
            "people": [{
                "person_id": [-1],
                "pose_keypoints_2d": data[i].tolist(), 
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": []
            }]
        }
        with open(os.path.join(save_folder, file_name), 'w') as f:
            json.dump(frame_data, f)
    return save_folder

if __name__ == "__main__":
    base_path = r'D:\석사\석사4차\pose2sim\Pose2Sim\Demo_SinglePerson\pose'
    json_folders = find_json_folders(base_path)

    print(f"Found {len(json_folders)} folders with JSON files")

    mode = input("Select tracking mode ('auto' for automatic, 'manual' for manual selection via mouse click): ")

    if mode not in ['auto', 'manual']:
        print("Wrong mode selected. Exiting...")
        exit()

    all_avg_min_avg = []
    save_folders = []

    for folder in json_folders:
        print(f"\nProcessing folder: {folder}")
        pos1, json_files, pre_tracking_data, avg_min_avg = track_person(folder, mode=mode)
        if pos1 is None:
            print(f"Skipping folder {folder} due to tracking failure")
            continue
        
        all_avg_min_avg.append(avg_min_avg)
        save_folder = save_data(pos1, json_files, folder)
        save_folders.append(save_folder)

        print(f"\nStarting animation for folder: {folder}")
        print("The animation window will close automatically when finished.")
        animate_pre_post_tracking(pre_tracking_data, pos1, os.path.basename(folder), frame_step=50, interval=30)
        print("Animation completed for this folder.")
    
    print("\nAll processing completed.")
    print(f"Average min_avg across all folders: {np.mean(all_avg_min_avg):.2f}")
    print(f"Processed data saved to: {save_folders}")
    print(f"Processed data saved to: {save_folders}")
