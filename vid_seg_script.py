import os  
import torch  
import csv
import numpy as np  
import argparse  
import glob  
from monai.metrics import DiceHelper, compute_surface_dice  
from scipy import ndimage  
from skimage.measure import label, regionprops


# the PNG palette for DAVIS 2017 dataset
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


# Argument parser  
def parse_args():  
    parser = argparse.ArgumentParser(description="SAM2 Video Predictor")  
    parser.add_argument('--base_video_dir', type=str, help="Base directory containing all video directories", default=".")
    parser.add_argument('--output_dir', type=str, help="Output dir for all the scores, predicted masks and metrics", default="./output")  
    parser.add_argument('--prefix', type=str, help="Prefix for video directories", default="BRATS_")  
    parser.add_argument('--steps', type=int, help="Number of steps to run the algorithm", default=5)  
    parser.add_argument('--method', type=str, choices=['first_above_threshold', 'largest_enclosed', 'max_intensity'], help="Method to determine the output mask", default='largest_enclosed')  
    return parser.parse_args()  

def load_masks_from_dir(
    input_mask_dir, video_name, frame_name, per_obj_png_file, allow_missing=False
):
    """Load masks from a directory as a dict of per-object masks."""
    if not per_obj_png_file:
        input_mask_path = os.path.join(input_mask_dir, video_name, f"{frame_name}.png")
        if allow_missing and not os.path.exists(input_mask_path):
            return {}, None
        input_mask, input_palette = load_ann_png(input_mask_path)
        per_obj_input_mask = get_per_obj_mask(input_mask)
    else:
        per_obj_input_mask = {}
        input_palette = None
        # each object is a directory in "{object_id:%03d}" format
        for object_name in os.listdir(os.path.join(input_mask_dir, video_name)):
            object_id = int(object_name)
            input_mask_path = os.path.join(
                input_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            if allow_missing and not os.path.exists(input_mask_path):
                continue
            input_mask, input_palette = load_ann_png(input_mask_path)
            per_obj_input_mask[object_id] = input_mask > 0

    return per_obj_input_mask, input_palette

def get_per_obj_mask(mask):
    """Split a mask into per-object masks."""
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask


def put_per_obj_mask(per_obj_mask, height, width):
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask

def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette


def save_ann_png(path, mask, palette):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)


def save_masks_to_dir(
    output_mask_dir,
    video_name,
    frame_name,
    per_obj_output_mask,
    height,
    width,
    per_obj_png_file,
    output_palette,
):
    """Save masks to a directory as PNG files."""
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    if not per_obj_png_file:
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        output_mask_path = os.path.join(
            output_mask_dir, video_name, f"{frame_name}.png"
        )
        save_ann_png(output_mask_path, output_mask, output_palette)
    else:
        for object_id, object_mask in per_obj_output_mask.items():
            object_name = f"{object_id:03d}"
            os.makedirs(
                os.path.join(output_mask_dir, video_name, object_name),
                exist_ok=True,
            )
            output_mask = object_mask.reshape(height, width).astype(np.uint8)
            output_mask_path = os.path.join(
                output_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            save_ann_png(output_mask_path, output_mask, output_palette)

# Function to calculate IoU
def calculate_iou(pred_mask, gt_mask):
    """Calculate Intersection over Union (IoU) between predicted and ground truth masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0
    iou = intersection / union
    return iou

def find_largest_enclosed_point(img, disturb=False):  
    img = np.pad(img, 1, mode='constant')  
    label, num_labels = ndimage.label(img)  
    max_distance = 1.42  
    max_coordinate = (0, 0)  
  
    for i in range(1, num_labels + 1):  
        region = (label == i).astype(int)  
        distance_transform = ndimage.distance_transform_edt(region)  
        dist_max = np.max(distance_transform)  
  
        if dist_max > max_distance:  
            max_distance = dist_max  
            max_coordinate = np.unravel_index(np.argmax(distance_transform), distance_transform.shape)  
  
    if max_distance == 1.42:  
        return max_distance, (None, None), None  # Added None for label  
  
    if not disturb:  
        return max_distance, (max_coordinate[1] - 1, max_coordinate[0] - 1), 1  # Default input label for foreground  
    else:  
        retry = 5  
        while retry > 0:  
            retry -= 1  
            disturb_x = np.random.randint(-5, 5)  
            disturb_y = np.random.randint(-5, 5)  
            if region[max_coordinate[0] + disturb_y, max_coordinate[1] + disturb_x] == 1:  
                return max_distance, (max_coordinate[1] + disturb_x - 1, max_coordinate[0] + disturb_y - 1), 1  # Default input label  
        return max_distance, (max_coordinate[1] - 1, max_coordinate[0] - 1), 1  # Default input label  
  
def find_largest_lesion_slide(gt, intensity_threshold=0):
    max_area = 0
    max_slide = None
    max_idx = None

    for idx, slide in enumerate(gt):
        # Create a binary mask for the lesion areas above the intensity threshold
        binary_mask = slide > intensity_threshold
        
        # Label connected components in the binary mask
        labeled_mask = label(binary_mask)
        
        # Calculate the area of each region and find the largest one
        for region in regionprops(labeled_mask):
            area = region.area
            if area > max_area:
                max_area = area
                max_slide = slide
                max_idx = idx
        
        # print(f"Slide {idx} largest lesion area: {max_area}")

    if max_slide is not None:
        print(f"Slide with largest lesion area found at index {max_idx}, area: {max_area}")
    else:
        print("No lesion found above intensity threshold.")
        
    return max_slide, 1, max_idx  # Return slide, input label, and frame index


def find_first_large_lesion_slide(gt, area_ratio_threshold=0.01, intensity_threshold=0):
    img_area = gt[0].shape[0] * gt[0].shape[1]  # Calculate total area of a single slide
    required_area = area_ratio_threshold * img_area  # Calculate required lesion area based on ratio
    
    for idx, slide in enumerate(gt):
        # Create a binary mask for lesion areas above the intensity threshold
        binary_mask = slide > intensity_threshold
        
        # Label connected components
        labeled_mask = label(binary_mask)
        
        # Check if any region meets the required area threshold
        for region in regionprops(labeled_mask):
            if region.area >= required_area:
                print(f"Slide {idx} meets area ratio threshold with lesion area: {region.area}")
                return slide, 1, idx  # Return slide, input label, and frame index
        
        # print(f"Slide {idx} does not meet area ratio threshold.")

    print("No slide found with a lesion area above the specified ratio.")
    return None, None, None

def generate_bounding_box_points(mask):  
    # Find the indices of the positive pixels  
    positive_indices = np.argwhere(mask > 0)  
      
    if len(positive_indices) == 0:  
        return None  # No positive pixels found  
      
    # Calculate the bounding box  
    min_coords = positive_indices.min(axis=0)  
    max_coords = positive_indices.max(axis=0)  
      
    # Define points at the corners of the bounding box  
    bounding_box_points = [  
        (min_coords[0], min_coords[1]),  
        (min_coords[0], max_coords[1]),  
        (max_coords[0], min_coords[1]),  
        (max_coords[0], max_coords[1]),  
        ((min_coords[0] + max_coords[0]) // 2, (min_coords[1] + max_coords[1]) // 2)  # Center point  
    ]  
      
    return bounding_box_points  
  
def generate_multiple_points(mask, num_points=5):  
    # Find the indices of the positive pixels  
    positive_indices = np.argwhere(mask > 0)  
      
    if len(positive_indices) < num_points:  
        return positive_indices.tolist()  # Return all if fewer than required  
      
    # Randomly select points  
    selected_indices = positive_indices[np.random.choice(positive_indices.shape[0], num_points, replace=False)]  
      
    return selected_indices.tolist()  
  
def generate_point_from_mask(mask):  
    # Find the indices of the positive pixels  
    positive_indices = np.argwhere(mask > 0)  
      
    if len(positive_indices) == 0:  
        return None  # No positive pixels found  
  
    # Calculate the centroid of the positive pixels  
    centroid = np.mean(positive_indices, axis=0).astype(int)  
  
    return tuple(centroid) 


# Main code execution  
def main():  
    args = parse_args()  
    base_video_dir = os.path.abspath(args.base_video_dir)  
    output_dir = os.path.abspath(args.output_dir)  
    prefix = args.prefix  
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)  
  
    # # Collect all video directories  
    # search_pattern = os.path.join(base_video_dir, f"{prefix}*")  
    # video_dirs = [d for d in glob.glob(search_pattern) if os.path.isdir(d)]  

    # Collect all video directories
    search_pattern = os.path.join(base_video_dir, "*")
    video_dirs = [d for d in glob.glob(search_pattern) if os.path.isdir(d)]

    # Initialize metrics lists  
    dice_scores = {}  
    nsd_scores = {}  
    iou_scores_dict = {}
  
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):  
        if torch.cuda.get_device_properties(0).major >= 8:  
            torch.backends.cuda.matmul.allow_tf32 = True  
            torch.backends.cudnn.allow_tf32 = True  
  
        from sam2.build_sam import build_sam2_video_predictor  
        # checkpoint = "./checkpoints/sam2.1_hiera_large.pt"  
        # model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml" 

        # checkpoint = "./checkpoints/checkpoint-med1.pt"
        # model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

        checkpoint = "./checkpoints/MedSAM2_pretrain.pth"
        model_cfg = "configs/sam2/sam2_hiera_t.yaml"

        # checkpoint = "./sam2_logs_newg4_36epoch/checkpoints/checkpoint.pt"
        # model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

        # checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt" 
        # model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

        predictor = build_sam2_video_predictor(model_cfg, checkpoint)  
  
        for video_dir in video_dirs:  
            ann_obj_id = 1  
            gt_files = [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1].lower() == ".npy"]  
            gt_files.sort(key=lambda p: int(os.path.splitext(p)[0]))  
  
            print(f"Processing {video_dir}, number of GT files: {len(gt_files)}")  
            if len(gt_files) == 0:  
                print(f"No ground truth files found in {video_dir}, skipping...")  
                continue  

            # from the beginning to the end
            inference_state = predictor.init_state(video_path=video_dir)
            
            gt_slides = [np.load(os.path.join(video_dir, f)) for f in gt_files]  
  
            # Determine output mask based on user's choice of method  
            threshold = 100  # Set your threshold here  
            gt_mask = None  
            input_label = None  
            frame_idx = None  
            
            if args.method == 'first_above_threshold':  
                gt_mask, input_label, frame_idx = find_first_large_lesion_slide(gt_slides, threshold)  
                if gt_mask is None:  
                    print("No slide above threshold, skipping...")  
                    continue  
            elif args.method in ['max_intensity', 'largest_enclosed']:  
                gt_mask, input_label, frame_idx = find_largest_lesion_slide(gt_slides)  
                if gt_mask is None:  
                    print("No valid slide found, skipping...")  
                    continue  
  
            # Initialize points and labels for segmentation  
            points = np.array([], dtype=np.float32)  
            labels = np.array([], np.int32)  
            gt_zeros = np.zeros_like(gt_mask)
  
            # Perform iteration  
            def perform_iteration(point_strategy=None):  
                nonlocal points, labels, gt_zeros  
            
                if args.method == 'largest_enclosed':  
                    FN = np.logical_and(gt_mask, np.logical_not(gt_zeros))  
                    FP = np.logical_and(np.logical_not(gt_mask), gt_zeros)  
                    assert len(FN.shape) == 2 and len(FP.shape) == 2  
            
                    max_dis_1, col_1, _ = find_largest_enclosed_point(FN, disturb=False)  
                    max_dis_2, col_2, _ = find_largest_enclosed_point(FP, disturb=False)  
            
                    if max_dis_1 >= max_dis_2:
                        col = col_1
                        input_label = 1
                    else:
                        col = col_2
                        input_label = 0
                    assert len(col) == 2

                    if col[0] is None:
                        return
                    else:
                        if points.size == 0:
                            points = np.array([col], dtype=np.int32)
                        else:
                            points = np.append(points, [col], axis=0).astype(np.int32)
                        labels = np.append(labels, input_label).astype(np.int32)
            
                elif args.method in ['first_above_threshold', 'max_intensity']:
                    if points.size == 0:  
                        if point_strategy == 'BB':  
                            point_list = generate_bounding_box_points(gt_mask)  
                        elif point_strategy == 'multiple':  
                            point_list = generate_multiple_points(gt_mask)  
                        else:  
                            # Default to centroid if no strategy is provided  
                            point_list = [generate_point_from_mask(gt_mask)]  
            
                        if point_list is not None:  
                            points = np.array(point_list, dtype=np.int32)  
                            labels = np.array([input_label] * len(point_list), dtype=np.int32)  
            
                if points.size > 0 and labels.size > 0:  
                    _, _, out_mask_logits = predictor.add_new_points(  
                        inference_state=inference_state,  
                        frame_idx=frame_idx,  # Use the frame_idx here  
                        obj_id=ann_obj_id,  
                        points=points,  
                        labels=labels,  
                    )  
                    gt_zeros = (out_mask_logits.cpu() > 0.0)[0][0]  
  
            # Loop for specified steps  
            for _ in range(args.steps):  
                perform_iteration()  
  
            video_segments = {}  # Video segments contains the per-frame segmentation results  
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):  
                video_segments[out_frame_idx] = {  
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()  
                    for i, out_obj_id in enumerate(out_obj_ids)  
                }  
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=frame_idx - 1, reverse=True):  
                video_segments[out_frame_idx] = {  
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()  
                    for i, out_obj_id in enumerate(out_obj_ids)  
                }  

            # Save the video_segments dictionary to a NumPy archive for later use
            video_name = os.path.basename(video_dir)
            video_segments_file = os.path.join(output_dir, f"vid_segments/{video_name}.npz")

            # Ensure the directory exists before saving
            os.makedirs(os.path.dirname(video_segments_file), exist_ok=True)
            np.savez_compressed(video_segments_file, **{
                str(out_frame_idx): video_segments[out_frame_idx] for out_frame_idx in video_segments
            })
            print(f"Video segments saved to {video_segments_file}")

            # Stack results into 3D arrays
            video_seg_3d = np.stack([video_segments[k][1] for k in video_segments])
            gt_3d = np.stack([gt_slides[k][None] for k in video_segments])


            print(f"video_seg_3d shape: {video_seg_3d.shape}, gt_3d shape: {gt_3d.shape}")

            # Define tensor shapes
            n_classes, batch_size = 1, 1
            spatial_shape = (video_seg_3d.shape[0], video_seg_3d.shape[2], video_seg_3d.shape[3]) 

            # Convert to torch tensors
            y_pred = torch.tensor(video_seg_3d).float().reshape(batch_size, n_classes, *spatial_shape)  # prediction
            y = torch.tensor(gt_3d).float().reshape(batch_size, n_classes, *spatial_shape)  # ground truth

            # Calculate Dice and NSD per video (not per frame)
            score, not_nans = DiceHelper(include_background=False, sigmoid=True, softmax=True)(y_pred, y)
            dice = score.item()
            nsd = compute_surface_dice(y_pred, y, class_thresholds=[1]).item()
            # Calculate IoU per video 
            iou_scores = [calculate_iou(video_segments[k][1], gt_slides[k]) for k in video_segments] 
            mean_iou = np.mean(iou_scores)

            # Append scores for this video
            dice_scores[video_name] = dice
            nsd_scores[video_name] = nsd
            iou_scores_dict[video_name] = mean_iou

        # Save per-video Dice and NSD scores to CSV file
        scores_file = os.path.join(output_dir, f"scores.csv")
        with open(scores_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["video", "Dice Score", "NSD Score"])
            for video_name in dice_scores:
                dice = dice_scores[video_name]
                nsd = nsd_scores[video_name]
                iou = iou_scores_dict[video_name]
                writer.writerow([video_name, dice, nsd])
        print(f"Per-video scores saved to {scores_file}")

        # After processing all videos, calculate final averages for all videos
        final_average_dice = np.mean(list(dice_scores.values()))
        final_average_nsd = np.mean(list(nsd_scores.values()))
        
        # Save final averages across all videos to a CSV summary file
        summary_file = os.path.join(output_dir, "final_metrics.csv")
        with open(summary_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Average Dice", "Average NSD"])
            writer.writerow([f"Final", final_average_dice, final_average_nsd])

        print(f"Final Average Dice across all videos: {final_average_dice:.4f}")
        print(f"Final Average NSD across all videos: {final_average_nsd:.4f}")


if __name__ == "__main__":  
    main()  

### predictor = https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
