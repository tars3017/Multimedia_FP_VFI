import os
import glob
import random
from pathlib import Path
import argparse
import shutil

def get_sequence_folders(root_dir):
    """Find all sequence folders containing images."""
    sequence_paths = []
    
    # Find all directories that contain image files
    for path in glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True):
        # Get the directory containing the images
        seq_dir = os.path.dirname(path)
        # Verify this folder has the expected image pattern
        if all(os.path.exists(os.path.join(seq_dir, f"im{i}.png")) for i in range(1, 8)):
            # Get the relative path from root_dir (e.g., "00001/0266")
            rel_path = os.path.relpath(seq_dir, root_dir)
            sequence_paths.append(rel_path)

    return sequence_paths

def create_annotation_files(root_dir, sequences, train_ratio=0.9, restructure_data=False):
    """Create annotation files for train and test sets."""
    # Shuffle sequences for random split
    # random.shuffle(sequences)
    
    # Split into train and test
    train_sequences = sequences
    
    # Create output directories if needed
    if restructure_data:
        os.makedirs(os.path.join(root_dir, "sequences"), exist_ok=True)
    
    # Create annotation for training set
    with open(os.path.join(root_dir, "tri_trainlist_complete.txt"), "w") as f:
        for seq_path in train_sequences:
            process_sequence(f, root_dir, seq_path, restructure_data)
    
    print(f"Created annotation files with {len(train_sequences)} training")

def process_sequence(file, root_dir, seq_path, restructure_data):
    """Process one sequence, extracting triplets and writing to annotation file."""
    # For each sequence, we can extract multiple triplets: [1,2,3], [3,4,5], [5,6,7]
    triplets = [(1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6), (5, 6, 7), (1, 3, 5), (2, 4, 6), (3, 5, 7), (1, 4, 7)]
    
    for i, (img1_idx, img2_idx, img3_idx) in enumerate(triplets):
        # Create a new sequence ID for this triplet
        triplet_id = f"{seq_path.replace('/', '_')}_{i}"
        
        if restructure_data:
            # Create directory for this triplet
            new_seq_dir = os.path.join(root_dir, "sequences", triplet_id)
            os.makedirs(new_seq_dir, exist_ok=True)
            
            # Copy the images with correct names
            src_dir = os.path.join(root_dir, seq_path)
            shutil.copy(os.path.join(src_dir, f"img{img1_idx}.png"), os.path.join(new_seq_dir, "im1.png"))
            shutil.copy(os.path.join(src_dir, f"img{img2_idx}.png"), os.path.join(new_seq_dir, "im2.png"))
            shutil.copy(os.path.join(src_dir, f"img{img3_idx}.png"), os.path.join(new_seq_dir, "im3.png"))
            
            # Write to annotation file
            file.write(f"{triplet_id}\n")
        else:
            # For non-restructure mode, you'll need to modify the dataset.py to handle your image naming convention
            file.write(f"{seq_path},{img1_idx},{img2_idx},{img3_idx}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate annotation files for Vimeo-like dataset")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset root")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Ratio of training data")
    parser.add_argument("--restructure", action="store_true", help="Restructure data to match Vimeo90K format")
    args = parser.parse_args()
    
    
    # Find all sequence folders
    sequences = get_sequence_folders(args.data_path)
    print(f"Found {len(sequences)} sequences")
    
    # Create annotation files
    create_annotation_files(args.data_path, sequences, args.train_ratio, args.restructure)