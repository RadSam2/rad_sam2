import traceback
import numpy as np
import os
import torch  
from PIL import Image
import csv
import matplotlib.pyplot as plt
from monai.metrics import DiceHelper, compute_surface_dice  
from multiprocessing import Pool

# Function to load the video segments saved as npz files
def load_video_segments(npz_file_path):
    """Load video segments from npz file and return the expected dictionary structure."""
    video_segments = {}
    
    # Load the npz file
    npz_data = np.load(npz_file_path, allow_pickle=True)
    
    # Reconstruct the dictionary format: {frame_idx: {obj_id: mask}}
    for frame_idx in npz_data.files:
        # The value is a dictionary-like structure with object IDs and masks
        video_segments[frame_idx] = npz_data[frame_idx].item()  # Reconstruct the dictionary
        
    return video_segments


# Function to calculate IoU
def calculate_iou(pred_mask, gt_mask):
    """Calculate Intersection over Union (IoU) between predicted and ground truth masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0
    iou = intersection / union
    return iou

# Function to load npy ground truth masks
def load_gt_masks(video_dir, frame_names):
    """Load ground truth masks for all frames in the video directory."""
    gt_masks = []
    for frame_name in frame_names:
        gt_path = os.path.join(video_dir, frame_name.replace(".jpg", ".npy"))
        gt_mask = np.load(gt_path)
        gt_masks.append(gt_mask)
    return np.stack(gt_masks)

def put_per_obj_mask(per_obj_mask, height, width):
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask

# Save masks to PNG files
def save_masks_to_dir(output_mask_dir, video_name, frame_name, per_obj_output_mask, height, width, per_obj_png_file, output_palette):
    """Save masks to a directory as PNG files."""
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    base_name, _ = os.path.splitext(frame_name)
    if not per_obj_png_file:
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        output_mask_path = os.path.join(output_mask_dir, video_name, f"{base_name}.png")
        save_ann_png(output_mask_path, output_mask, output_palette)
    else:
        for object_id, object_mask in per_obj_output_mask.items():
            object_name = f"{object_id:03d}"
            os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
            output_mask = object_mask.reshape(height, width).astype(np.uint8)
            output_mask_path = os.path.join(output_mask_dir, video_name, f"{base_name}.png")
            save_ann_png(output_mask_path, output_mask, output_palette)

# Save PNG file with palette
def save_ann_png(path, mask, palette):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2

    # Check unique values in the mask
    unique_values = np.unique(mask)
    print(f"Unique values in mask for object {path}: {unique_values}")  # Debugging statement

    # # Scale the mask values if necessary
    # if np.max(mask) > 1:
    #     mask = mask / np.max(mask)  # Normalize to 0-1 if values are greater than 1
    # mask = (mask * 255).astype(np.uint8)  # Scale to 0-255 for image representation

    # # Check mask values after scaling
    # print(f"Unique values after scaling for object {path}: {np.unique(mask)}")

    output_mask = Image.fromarray(mask)

    if palette is not None:
        output_mask.putpalette(palette)
    else:
        pass
        # print("Warning: No palette provided. Saving mask without palette.")


        # Display the mask to verify (optional)
    # plt.imshow(output_mask, cmap='gray')
    # plt.title(f"Mask for Object {path}")
    # plt.show()
    output_mask.save(path)

# Load a PNG file as mask
def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette

def inspect_npz_file(npz_file_path):
    """Inspect the contents of the npz file."""
    npz_data = np.load(npz_file_path, allow_pickle=True)
    for frame_idx in npz_data.files:
        print(f"Frame: {frame_idx}, Data: {npz_data[frame_idx]}")

# def process_npz_and_save_masks(npz_file_path, ground_truth_dir, frame_names):
#     """Process segmentation masks from npz file, calculate IoU, and save masks as PNG."""
#     # Load video segments from npz file
#     video_segments = load_video_segments(npz_file_path)
    
#     # Load ground truth masks
#     gt_masks = [np.load(os.path.join(ground_truth_dir, f"{os.path.splitext(frame_name)[0]}.npy")) for frame_name in frame_names]
    
#     # Stack results into 3D arrays
#     video_seg_3d = np.stack([video_segments[str(k)][1] for k in range(len(frame_names))])
#     gt_3d = np.stack([gt_masks[k][None] for k in range(len(frame_names))])

#     print(f"video_seg_3d shape: {video_seg_3d.shape}, gt_3d shape: {gt_3d.shape}")

#     # Define tensor shapes
#     n_classes, batch_size = 1, 1
#     spatial_shape = (video_seg_3d.shape[0], video_seg_3d.shape[2], video_seg_3d.shape[3])

#     # Convert to torch tensors
#     y_pred = torch.tensor(video_seg_3d).float().reshape(batch_size, n_classes, *spatial_shape)  # prediction
#     y = torch.tensor(gt_3d).float().reshape(batch_size, n_classes, *spatial_shape)  # ground truth

#     # Calculate Dice and NSD per video (not per frame)
#     score, not_nans = DiceHelper(include_background=False, sigmoid=True, softmax=True)(y_pred, y)
#     dice = score.item()
#     nsd = compute_surface_dice(y_pred, y, class_thresholds=[1]).item()

#     # Calculate IoU per video
#     iou_scores = [calculate_iou(video_segments[str(k)][1], gt_masks[k]) for k in range(len(frame_names))]
#     mean_iou = np.mean(iou_scores)

#     return dice, nsd, mean_iou

# # Process all npz video segments and perform IoU, save PNG masks
# def process_npz_and_save_masks(npz_file_path, ground_truth_dir, frame_names):
#     """Process segmentation masks from npz file, calculate IoU, and save masks as PNG."""
#     # inspect_npz_file(npz_file_path)
#     video_segments = load_video_segments(npz_file_path)
    
#     # Load the ground truth masks
#     gt_masks = load_gt_masks(ground_truth_dir, frame_names)
    
#     iou_scores = []
#     for frame_name in frame_names:
#         frame_idx = int(os.path.splitext(frame_name)[0])
        
#         if str(frame_idx) in video_segments:
#             # Load predicted mask for the frame
#             predicted_masks = video_segments[str(frame_idx)]  # dict {obj_id: mask}
            
#             # Load corresponding ground truth mask for the frame
#             gt_mask = gt_masks[frame_idx]
            
#             for obj_id, pred_mask in predicted_masks.items():
#                 pred_mask = pred_mask.astype(np.uint8)
                
#                 # Print the shape of pred_mask to understand its dimensions
#                 # print(f"Shape of pred_mask for frame {frame_name}, object {obj_id}: {pred_mask.shape}")
                
#                 # Calculate IoU
#                 iou = calculate_iou(pred_mask, gt_mask)
#                 iou_scores.append(iou)
#                 print(f"IoU for frame {frame_name}, object {obj_id}: {iou:.4f}")
                
#                 # Save predicted mask as PNG
#                 # new_output_mask = os.path.join(output_mask_dir, "mast_pics")
                
#                 # # Check if pred_mask has more than 2 dimensions
#                 # if len(pred_mask.shape) == 2:
#                 #     height, width = pred_mask.shape
#                 # else:
#                 #     # print(f"Unexpected shape: {pred_mask.shape}. Adjusting to 2D.")
#                 #     # Handle the case of extra dimensions, e.g., take the first channel if necessary
#                 #     pred_mask = pred_mask.squeeze()  # Remove any singleton dimensions
#                 #     height, width = pred_mask.shape
                
#                 # save_masks_to_dir(
#                 #     new_output_mask,
#                 #     video_name,
#                 #     frame_name,
#                 #     {obj_id: pred_mask},
#                 #     height,
#                 #     width,
#                 #     per_obj_png_file=True,  # Save individual object mask as PNG
#                 #     output_palette=None  # Provide your palette if necessary
#                 # )

    
#     # Return the mean IoU for the entire video
#     if len(iou_scores) > 0:
#         mean_iou = np.mean(iou_scores)
#     else:
#         mean_iou = 0.0  # If there are no valid IoU scores
#     return mean_iou


# # Main loop to process multiple videos
# def process_all_videos(base_video_dir, output_mask_dir, video_prefix):
#     """Process all videos in the base directory and calculate metrics."""
#     # Collect all video directories
#     video_dirs = [d for d in os.listdir(base_video_dir) if os.path.isdir(os.path.join(base_video_dir, d))] # and d.startswith(video_prefix)
    
#     iou_scores = []
#     dice_scores = []  
#     nsd_scores = []
#     video_names = []
    
#     for video_dir in video_dirs:
#         video_path = os.path.join(base_video_dir, video_dir)
#         frame_names = [p for p in os.listdir(video_path) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]]
#         frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        
#         # Check if there are frames to process
#         if len(frame_names) == 0:
#             print(f"No frames found in {video_dir}, skipping...")
#             continue
        
#         # Locate the saved npz file for this video
#         video_name = os.path.basename(video_dir)
#         npz_file_path = os.path.join(output_mask_dir, f"vid_segments/{video_name}.npz")
#         os.makedirs(os.path.dirname(npz_file_path), exist_ok=True)
        
#         if os.path.exists(npz_file_path):
#             print(f"Processing video {video_dir}")
#             dice, nsd, mean_iou = process_npz_and_save_masks(npz_file_path, video_path, frame_names)
#             dice_scores.append(dice)
#             nsd_scores.append(nsd)
#             iou_scores.append(mean_iou)
#             video_names.append(video_name)
#         else:
#             print(f"Segmentation npz file not found for {video_dir}, skipping...")

#     # Save mean IoU scores per video to CSV
#     scores_file = os.path.join(output_mask_dir, "iou_scores_per_video.csv")
#     with open(scores_file, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Video", "Mean IoU Score"])
#         for video_name, iou, dice, nsd in zip(video_names, iou_scores, dice_scores, nsd_scores):
#             writer.writerow([video_name, iou, dice, nsd])
#     print(f"Mean scores saved to {scores_file}")

# def process_video(video_dir):
#     video_path = os.path.join(base_video_dir, video_dir)
#     frame_names = [p for p in os.listdir(video_path) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]]
#     frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

#     # Check if there are frames to process
#     if len(frame_names) == 0:
#         print(f"No frames found in {video_dir}, skipping...")
#         return None

#     # Locate the saved npz file for this video
#     video_name = os.path.basename(video_dir)
#     npz_file_path = os.path.join(output_mask_dir, f"vid_segments/{video_name}.npz")
#     os.makedirs(os.path.dirname(npz_file_path), exist_ok=True)

#     if os.path.exists(npz_file_path):
#         print(f"Processing video {video_dir}")
#         dice, nsd, mean_iou = process_npz_and_save_masks(npz_file_path, output_mask_dir, video_name, frame_names)
#         return video_name, dice, nsd, mean_iou
#     else:
#         print(f"Segmentation npz file not found for {video_dir}, skipping...")
#         return None

# def process_all_videos(base_video_dir, output_mask_dir, video_prefix):
#     """Process all videos in the base directory and calculate metrics."""
#     # Collect all video directories
#     video_dirs = [d for d in os.listdir(base_video_dir) if os.path.isdir(os.path.join(base_video_dir, d))]  # and d.startswith(video_prefix)

#     with Pool() as pool:
#         results = pool.map(process_video, video_dirs)

#     # Initialize metrics lists
#     dice_scores = {}
#     nsd_scores = {}
#     iou_scores = {}
#     video_names = []

#     for result in results:
#         if result is not None:
#             video_name, dice, nsd, mean_iou = result
#             dice_scores[video_name] = dice
#             nsd_scores[video_name] = nsd
#             iou_scores[video_name] = mean_iou
#             video_names.append(video_name)

#     # Save mean IoU scores per video to CSV
#     scores_file = os.path.join(output_mask_dir, "iou_scores_per_video.csv")
#     with open(scores_file, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Video", "Mean IoU Score", "Dice Score", "NSD Score"])
#         for video_name in video_names:
#             iou = iou_scores[video_name]
#             dice = dice_scores[video_name]
#             nsd = nsd_scores[video_name]
#             writer.writerow([video_name, iou, dice, nsd])
#     print(f"Mean scores saved to {scores_file}")

# # Run the full process for all videos
# base_video_dir = "/home/azureuser/cloudfiles/code/Users/admin/sam2_seg/processed_dataset/val_dataset"
# output_mask_dir = "/home/azureuser/cloudfiles/code/Users/admin/sam2_seg/output_dir_TEST_pre1"
# video_prefix = "BRATS_"
# process_all_videos(base_video_dir, output_mask_dir, video_prefix)



# def process_npz_and_save_masks(npz_file_path, ground_truth_dir, video_name, frame_names, save_per_slide=False):
#     """Process segmentation masks from npz file, calculate IoU, and save masks as PNG."""
#     try:
#         # Load video segments from npz file
#         video_segments = load_video_segments(npz_file_path)
        
#         # Load ground truth masks
#         gt_masks = [np.load(os.path.join(ground_truth_dir, f"{os.path.splitext(frame_name)[0]}.npy")) for frame_name in frame_names]
        
#         # Stack results into 3D arrays
#         video_seg_3d = np.stack([video_segments[str(k)][1] for k in range(len(frame_names))])
#         gt_3d = np.stack([gt_masks[k][None] for k in range(len(frame_names))])

#         print(f"video_seg_3d shape: {video_seg_3d.shape}, gt_3d shape: {gt_3d.shape}")

#         # Define tensor shapes
#         n_classes, batch_size = 1, 1
#         spatial_shape = (video_seg_3d.shape[0], video_seg_3d.shape[2], video_seg_3d.shape[3])

#         # Convert to torch tensors
#         y_pred = torch.tensor(video_seg_3d).float().reshape(batch_size, n_classes, *spatial_shape)  # prediction
#         y = torch.tensor(gt_3d).float().reshape(batch_size, n_classes, *spatial_shape)  # ground truth

#         # Calculate Dice and NSD per video (not per frame)
#         score, not_nans = DiceHelper(include_background=False, sigmoid=True, softmax=True)(y_pred, y)
#         dice = score.item()
#         nsd = compute_surface_dice(y_pred, y, class_thresholds=[1]).item()

#         # Calculate IoU per video
#         iou_scores = [calculate_iou(video_segments[str(k)][1], gt_masks[k]) for k in range(len(frame_names))]
#         mean_iou = np.mean(iou_scores)

#         if save_per_slide:
#             return dice, nsd, mean_iou, iou_scores  # Return per-slide scores as well

#         return dice, nsd, mean_iou
#     except Exception as e:
#         print(f"Error processing {video_name}: {e}")
#         return None

# def process_video(args):
#     video_dir, base_video_dir, output_mask_dir, save_per_slide = args
#     video_path = os.path.join(base_video_dir, video_dir)
#     frame_names = [p for p in os.listdir(video_path) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]]
#     frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

#     # Check if there are frames to process
#     if len(frame_names) == 0:
#         print(f"No frames found in {video_dir}, skipping...")
#         return None

#     # Locate the saved npz file for this video
#     video_name = os.path.basename(video_dir)
#     npz_file_path = os.path.join(output_mask_dir, f"vid_segments/{video_name}.npz")
#     os.makedirs(os.path.dirname(npz_file_path), exist_ok=True)

#     if os.path.exists(npz_file_path):
#         try:
#             print(f"Processing video {video_dir}")
#             result = process_npz_and_save_masks(npz_file_path, video_path, video_name, frame_names, save_per_slide)
#             return video_name, result, frame_names
#         except Exception as e:
#             print(f"Error processing {video_dir}: {e}")
#             return None
#     else:
#         print(f"Segmentation npz file not found for {video_dir}, skipping...")
#         return None

# def process_all_videos(base_video_dir, output_mask_dir, video_prefix, save_per_slide=False):
#     """Process all videos in the base directory and calculate metrics."""
#     # Collect all video directories
#     video_dirs = [d for d in os.listdir(base_video_dir) if os.path.isdir(os.path.join(base_video_dir, d))]  # and d.startswith(video_prefix)

#     args = [(video_dir, base_video_dir, output_mask_dir, save_per_slide) for video_dir in video_dirs]
    
#     with Pool() as pool:
#         results = pool.map(process_video, args)

#     # Initialize metrics lists
#     dice_scores = {}
#     nsd_scores = {}
#     iou_scores = {}
#     per_slide_scores = {}
#     video_names = []
#     frame_names_dict = {}

#     for result in results:
#         if result is not None:
#             video_name, metrics, frame_names = result
#             if save_per_slide:
#                 dice, nsd, mean_iou, iou_scores_list = metrics
#                 per_slide_scores[video_name] = iou_scores_list
#                 frame_names_dict[video_name] = frame_names
#             else:
#                 dice, nsd, mean_iou = metrics

#             dice_scores[video_name] = dice
#             nsd_scores[video_name] = nsd
#             iou_scores[video_name] = mean_iou
#             video_names.append(video_name)

#     # Save mean IoU scores per video and per slide scores to CSV
#     scores_file = os.path.join(output_mask_dir, "iou_scores_per_video.csv")
#     with open(scores_file, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         if save_per_slide:
#             writer.writerow(["Video", "Frame", "IoU Score", "Mean IoU Score", "Dice Score", "NSD Score"])
#             for video_name in video_names:
#                 iou = iou_scores[video_name]
#                 dice = dice_scores[video_name]
#                 nsd = nsd_scores[video_name]
#                 for frame_name, slide_iou in zip(frame_names_dict[video_name], per_slide_scores[video_name]):
#                     writer.writerow([video_name, frame_name, slide_iou, iou, dice, nsd])
#         else:
#             writer.writerow(["Video", "Mean IoU Score", "Dice Score", "NSD Score"])
#             for video_name in video_names:
#                 iou = iou_scores[video_name]
#                 dice = dice_scores[video_name]
#                 nsd = nsd_scores[video_name]
#                 writer.writerow([video_name, iou, dice, nsd])
#     print(f"Scores saved to {scores_file}")

# # Run the full process for all videos
# base_video_dir = "/home/azureuser/cloudfiles/code/Users/admin/sam2_seg/processed_dataset/val_dataset_sig"
# output_mask_dir = "/home/azureuser/cloudfiles/code/Users/admin/sam2_seg/output_dir_TEST_post_sig"
# video_prefix = "BRATS_"
# process_all_videos(base_video_dir, output_mask_dir, video_prefix, save_per_slide=True)



def to_one_hot(tensor, num_classes=2):
    """Convert binary tensor to one-hot encoded format."""
    # Ensure input is binary
    tensor = (tensor > 0.5).float()
    # Initialize output tensor with zeros
    shape = tensor.shape
    one_hot = torch.zeros((shape[0], num_classes, *shape[2:]), device=tensor.device)
    # Set values
    one_hot[:, 0] = 1 - tensor[:, 0]  # background
    one_hot[:, 1] = tensor[:, 0]      # foreground
    return one_hot

def process_npz_and_save_masks(npz_file_path, ground_truth_dir, video_name, frame_names, save_per_slide=False):
    """Process segmentation masks from npz file, calculate IoU, Dice, and NSD per slide."""
    try:
        # Load video segments from npz file
        video_segments = load_video_segments(npz_file_path)
        
        # Load ground truth masks
        
        gt_masks = [np.load(os.path.join(ground_truth_dir, f"{os.path.splitext(frame_name)[0]}.npy")) for frame_name in frame_names]
        
        # Initialize per-slide score lists
        iou_scores = []
        dice_scores = []
        nsd_scores = []

        # Check if there are frames to process
        if len(video_segments) == 0 or len(video_segments) != len(frame_names):
            print(f"No pred slide found in vid segment: {video_name}, skipping...")
            return None
        # Calculate metrics for each slide
        for k in range(len(frame_names)):

            if str(k) not in video_segments:
                print(f"Key {k} not found in video segments for video {video_name}")
                print(f" lenght of {video_name}: {len(frame_names)}___{len(video_segments)}")
                continue

            # Get current slide prediction and ground truth
            pred = video_segments[str(k)][1]
            gt = gt_masks[k]

            # Calculate IoU for current slide
            iou = calculate_iou(pred, gt)
            iou_scores.append(iou)

            # Prepare tensors for Dice and NSD calculation
            y_pred = torch.tensor(pred[None, None]).float()  # Shape: [1, 1, H, W]
            y = torch.tensor(gt[None, None]).float()        # Shape: [1, 1, H, W]

            # Calculate Dice per slide
            score, _ = DiceHelper(include_background=False, sigmoid=True, softmax=True)(y_pred, y)
            dice_scores.append(score.item())

            # # Prepare tensors specifically for NSD calculation
            # # Add extra time dimension for NSD
            # y_pred_nsd = torch.tensor(pred[None, None, None]).float()  # Shape: [1, 1, 1, H, W]
            # y_nsd = torch.tensor(gt[None, None, None]).float()        # Shape: [1, 1, 1, H, W]
            # Convert to one-hot format for NSD
            y_pred_one_hot = to_one_hot(y_pred)  # Shape: [1, 2, H, W]
            y_one_hot = to_one_hot(y)            # Shape: [1, 2, H, W]

            # Calculate NSD per slide
            try:
                # We use class_index=1 since after one-hot encoding, index 1 is our foreground class
                y = y_one_hot.unsqueeze(2)
                nsd = compute_surface_dice(y_pred_one_hot, y, class_thresholds=[1]).item()
                nsd_scores.append(nsd)
            except Exception as e:
                print(f"Error calculating NSD for slide {k} in video {video_name}: {e}")
                print(f"Shapes - pred: {y_pred_one_hot.shape}, gt: {y.shape}")
                nsd_scores.append(None)

        # Calculate mean scores (excluding None values for NSD)
        mean_iou = np.mean(iou_scores)
        mean_dice = np.mean(dice_scores)
        mean_nsd = np.mean([s for s in nsd_scores if s is not None]) if any(s is not None for s in nsd_scores) else None

        if save_per_slide:
            return mean_dice, mean_nsd, mean_iou, iou_scores, dice_scores, nsd_scores

        return mean_dice, mean_nsd, mean_iou
    except Exception as e:
        print(f"Error processing {video_name}: {e}")
        traceback.print_exc()  # This will print the full error traceback
        return None
    
    
def process_video(args):
    video_dir, base_video_dir, output_mask_dir, save_per_slide = args
    video_path = os.path.join(base_video_dir, video_dir)
    frame_names = [p for p in os.listdir(video_path) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # Check if there are frames to process
    if len(frame_names) == 0:
        print(f"No frames found in {video_dir}, skipping...")
        return None

    # Locate the saved npz file for this video
    video_name = os.path.basename(video_dir)
    npz_file_path = os.path.join(output_mask_dir, f"vid_segments/{video_name}.npz")
    os.makedirs(os.path.dirname(npz_file_path), exist_ok=True)

    if os.path.exists(npz_file_path):
        try:
            print(f"Processing video {video_dir}")
            result = process_npz_and_save_masks(npz_file_path, video_path, video_name, frame_names, save_per_slide)
            return video_name, result, frame_names
        except Exception as e:
            print(f"Error processing {video_dir}: {e}")
            return None
    else:
        print(f"Segmentation npz file not found for {video_dir}, skipping...")
        return None

def process_all_videos(base_video_dir, output_mask_dir, save_per_slide=False):
    """Process all videos in the base directory and calculate metrics."""
    # Collect all video directories
    video_dirs = [d for d in os.listdir(base_video_dir) if os.path.isdir(os.path.join(base_video_dir, d))]

    args = [(video_dir, base_video_dir, output_mask_dir, save_per_slide) for video_dir in video_dirs]
    
    with Pool() as pool:
        results = pool.map(process_video, args)

    # Initialize metrics dictionaries
    dice_scores = {}
    nsd_scores = {}
    iou_scores = {}
    per_slide_scores = {
        'iou': {},
        'dice': {},
        'nsd': {}
    }
    video_names = []
    frame_names_dict = {}

    for result in results:
        if result is not None:
            video_name, metrics, frame_names = result
            if metrics is None:
                print(f"Metrics not found for video {video_name}, skipping...")
                continue
            if save_per_slide:
                mean_dice, mean_nsd, mean_iou, iou_scores_list, dice_scores_list, nsd_scores_list = metrics
                per_slide_scores['iou'][video_name] = iou_scores_list
                per_slide_scores['dice'][video_name] = dice_scores_list
                per_slide_scores['nsd'][video_name] = nsd_scores_list
                frame_names_dict[video_name] = frame_names
            else:
                mean_dice, mean_nsd, mean_iou = metrics

            dice_scores[video_name] = mean_dice
            nsd_scores[video_name] = mean_nsd
            iou_scores[video_name] = mean_iou
            video_names.append(video_name)

    # Save scores to CSV
    scores_file = os.path.join(output_mask_dir, "scores_per_video_and_slide_.csv")
    with open(scores_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        if save_per_slide:
            writer.writerow(["Video", "Frame", "IoU Score", "Dice Score", "NSD Score", 
                           "Mean IoU Score", "Mean Dice Score", "Mean NSD Score"])
            for video_name in video_names:
                mean_iou = iou_scores[video_name]
                mean_dice = dice_scores[video_name]
                mean_nsd = nsd_scores[video_name]
                
                for i, frame_name in enumerate(frame_names_dict[video_name]):
                    writer.writerow([
                        video_name, 
                        frame_name,
                        per_slide_scores['iou'][video_name][i],
                        per_slide_scores['dice'][video_name][i],
                        per_slide_scores['nsd'][video_name][i] if per_slide_scores['nsd'][video_name][i] is not None else "N/A",
                        mean_iou,
                        mean_dice,
                        mean_nsd if mean_nsd is not None else "N/A"
                    ])
        else:
            writer.writerow(["Video", "Mean IoU Score", "Mean Dice Score", "Mean NSD Score"])
            for video_name in video_names:
                writer.writerow([
                    video_name,
                    iou_scores[video_name],
                    dice_scores[video_name],
                    nsd_scores[video_name] if nsd_scores[video_name] is not None else "N/A"
                ])
    print(f"Scores saved to {scores_file}")

# # Run the full process for all videos
# base_video_dir = "/home/azureuser/cloudfiles/code/Users/admin/sam2_seg/_eval_common_sig"
# output_mask_dir = "/home/azureuser/cloudfiles/code/Users/admin/sam2_seg/output_eval_common_sig"
# video_prefix = "BRATS_"
# process_all_videos(base_video_dir, output_mask_dir, video_prefix, save_per_slide=True)

# Run the full process for all videos
base_video_dir = "C:/Users/ezeki/Downloads/eval_sig/eval_common_sig"
output_mask_dir = "C:/Users/ezeki/OneDrive/Documents/Khunsa_Project_SAM2/sam2_workspace_azure/output_eval_common_sig_medsam2_complete"
process_all_videos(base_video_dir, output_mask_dir, save_per_slide=True)
