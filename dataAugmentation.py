import numpy as np
import os
import itertools
import random
import matplotlib.pyplot as plt
import math
import uuid


#--------------------------Data Augmentation Functions--------------------------

def random_dropout(coords, dropout_rate=0.1):
    """
    Randomly drops frames in the coordinates array based on the dropout rate.
    """
    while True:
        mask = np.random.rand(coords.shape[0]) > dropout_rate
        if np.any(mask):
            break
    return coords[mask]

def random_noise(coords, noise_strength=0.01):
    noisy_coords = coords.copy()
    total_points = coords.size
    num_noisy = int(total_points * 0.15)
    flat_indices = np.random.choice(total_points, num_noisy, replace=False)
    noise = np.random.normal(0, noise_strength, num_noisy)
    # Flatten, add noise, then reshape
    flat = noisy_coords.flatten()
    flat[flat_indices] += noise
    noisy_coords = flat.reshape(coords.shape)
    return noisy_coords

def scaling_coords(coords, scale_factor=1.1):
    return coords * scale_factor

def rotation_coords(coords, angle_degrees=15):
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])
    return coords @ rotation_matrix.T

def remove_landmarks(coords, indices_to_remove):
    # Set landmarks (rows) at the specified indices to zero instead of deleting
    coords_zeroed = coords.copy()
    coords_zeroed[indices_to_remove, :] = 0
    return coords_zeroed

# def elastic_deformation(coords, alpha=1.0, sigma=0.5):
#     deformed_coords = coords.copy()
#     dx = np.random.normal(0, sigma, coords.shape[0:2])
#     dy = np.random.normal(0, sigma, coords.shape[0:2])
#     deformed_coords[..., 0] += dx * alpha
#     deformed_coords[..., 1] += dy * alpha
#     print(deformed_coords)
#     return deformed_coords

def plot_coords(coords, title="Augmented Data"):
    """
    Plot each frame as a subplot, visualizing arms and hands
    Assumes coords is (frames, landmarks, 2).
    """
    # HIGHLIGHTED: Visualization code disabled for batch processing
    # if coords.ndim == 2:
    #     coords = coords[np.newaxis, ...]
    # num_frames = coords.shape[0]
    # cols = 5
    # rows = math.ceil(num_frames / cols)
    # fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    # axes = axes.flatten()
    # for i in range(num_frames):
    #     ax = axes[i]
    #     frame = coords[i]
    #     left_arm = frame[0:3]
    #     right_arm = frame[3:6]
    #     left_hand = frame[6:27]
    #     right_hand = frame[27:48]
    #     ax.scatter(frame[:, 0], frame[:, 1], color='blue', s=8)
    #     ax.plot(left_arm[:, 0], left_arm[:, 1], color='green', linewidth=1.5)
    #     ax.plot(right_arm[:, 0], right_arm[:, 1], color='orange', linewidth=1.5)
    #     ax.plot(left_hand[:, 0], left_hand[:, 1], color='green', alpha=0.5)
    #     ax.plot(right_hand[:, 0], right_hand[:, 1], color='orange', alpha=0.5)
    #     ax.set_title(f"Frame {i}", fontsize=8)
    #     ax.invert_yaxis()
    #     ax.axis('off')
    # for j in range(num_frames, len(axes)):
    #     axes[j].axis('off')
    # plt.suptitle(title)
    # plt.tight_layout()
    # plt.show()

def ask_and_confirm(aug_func, coords, **kwargs):
    """
    Apply aug_func to coords, plot, and ask user for confirmation.
    """
    # HIGHLIGHTED: User confirmation disabled for batch processing
    # aug_coords = aug_func(coords, **kwargs)
    # plot_coords(aug_coords, title=f"{aug_func.__name__} preview")
    # resp = input(f"Is the result of {aug_func.__name__} okay? (y/n): ")
    # return resp.strip().lower() == 'y'
    return True

def process_removed_landmarks(coords, all_combinations, base_filename, file_path):
    """
    For up to 200 combinations, set the specified landmarks to zero and save the result.
    """
    selected_combinations = all_combinations
    if len(all_combinations) > 17:
        selected_combinations = random.sample(all_combinations, 15)
    for indices in selected_combinations:
        coords_removed = remove_landmarks(coords, indices_to_remove=indices)
        unique_id = uuid.uuid4()
        save_name = os.path.join(os.path.dirname(file_path), f"{base_filename}_removed_landmarks_{unique_id}.npy")
        np.save(save_name, coords_removed)


#--------------------------File Fetching And Sending--------------------------


def fetch_file(main_folder):
    # Generator to yield (filename, data) for all .npy files in the given folder and subfolders
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)

                # Skip and delete any type of augmented file
                if 'aug' in file or 'removed_landmarks' in file:
                    os.remove(file_path)
                    continue

                yield file_path, np.load(file_path)


#--------------------------Main Augmentation Loop--------------------------


AUGMENTATIONS = [
    ('random_dropout', random_dropout, {'dropout_rate': (0.05, 0.15)}, 12),
    ('random_noise', random_noise, {'noise_strength': (0.01, 0.15)}, 12),
    ('scaling_coords', scaling_coords, {'scale_factor': (0.85, 1.15)}, 12),
    ('rotation_coords', rotation_coords, {'angle_degrees': (-15, 15)}, 12),
    # ('elastic_deformation', elastic_deformation, {'alpha': (0.5, 1.5), 'sigma': (0.1, 0.5)}, 0),
]

confirmed = {}

for file_path, coords in fetch_file('Dataset_processed'):
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\nAugmenting file: {file_path}")
    print("Coordinates:\n", coords)
    # HIGHLIGHTED: Visualization disabled
    # plot_coords(coords, title=f"Original Data: {os.path.basename(file_path)}")
    # For each augmentation, confirm with user the first time, then run N times as specified
    for aug_name, aug_func, param_ranges, num_augs in AUGMENTATIONS:
        if aug_name not in confirmed:
            # Use mid-range values for preview
            preview_kwargs = {k: (v[0]+v[1])/2 for k, v in param_ranges.items()}
            if ask_and_confirm(aug_func, coords, **preview_kwargs):
                confirmed[aug_name] = True
            else:
                confirmed[aug_name] = False
                continue  # Skip this augmentation if not confirmed
        if confirmed[aug_name]:
            for i in range(num_augs):
                aug_kwargs = {k: random.uniform(*v) for k, v in param_ranges.items()}
                aug_coords = aug_func(coords, **aug_kwargs)
                unique_id = uuid.uuid4()
                save_name = os.path.join(os.path.dirname(file_path), f"{base_filename}_{aug_name}_{unique_id}.npy")

                np.save(save_name, aug_coords)

    # Removed landmarks: generate all combinations of 1 to 6 indices, save up to 200
    first_six = list(range(6))
    all_combinations = [indices for r in range(1, 7) for indices in itertools.combinations(first_six, r)]
    process_removed_landmarks(coords, all_combinations, base_filename, file_path)
