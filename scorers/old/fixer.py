import os
import json
import glob
from pathlib import Path

def process_train_file(file_path):
    """
    Process a single train JSONL file and create detector-specific files.
    """
    print(f"Processing: {file_path}")
    try:
        # Read the original file
        with open(file_path, 'r') as f:
            content = f.read().strip()
            if not content:
                print(f"  Warning: Empty file: {file_path}")
                return
            
            # Try to parse as JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Try parsing as JSONL (multiple lines)
                lines = content.split('\n')
                if len(lines) == 1:
                    print(f"  Error: Could not parse JSON in {file_path}")
                    return
                # Take the first line for now
                data = json.loads(lines[0])
                if isinstance(data, list):
                    data = data[0]
    except Exception as e:
        print(f"  Error reading file {file_path}: {e}")
        return
    
    # Extract detectors from the data
    detectors = []
    
    # Handle trained detectors
    train_detectors = ['mdeberta', 'roberta']
    
    # Extract detector-specific data for each trained detector
    for detector in train_detectors:
        if detector in data:
            # Directly use the metrics dictionary without the detector key wrapper
            detectors.append((detector, data[detector]))
    
    if not detectors:
        print(f"  Warning: No train detectors found in {file_path}")
        return
    
    # Create output files for each detector
    file_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    
    # Determine the base name to use for the new files
    base_name = file_name.replace('_train.jsonl', '')
    
    # Create detector-specific files
    for detector_name, detector_data in detectors:
        output_file = os.path.join(file_dir, f"{base_name}_train_{detector_name}.jsonl")
        
        with open(output_file, 'w') as f:
            json.dump(detector_data, f)
        
        print(f"  Created: {output_file}")

def main():
    """
    Process all train files in the specified directories.
    """
    # Define base directories
    base_dirs = [
        "/scratch/users/k21157437/paras/data",
        "/scratch/users/k21157437/sums/data",
        "/scratch/users/k21157437/neutral_new/data"
    ]
    
    languages = ['en', 'pt', 'vi']
    
    # Process each base directory
    for base_dir in base_dirs:
        # Skip if the directory doesn't exist
        if not os.path.exists(base_dir):
            print(f"Directory does not exist: {base_dir}")
            continue
        
        # Process each language
        for lang in languages:
            detect_dir = os.path.join(base_dir, lang, "detect")
            
            # Skip if the detect directory doesn't exist
            if not os.path.exists(detect_dir):
                print(f"Detect directory does not exist: {detect_dir}")
                continue
            
            # Find all train files that match the pattern
            file_pattern = os.path.join(detect_dir, "*_train.jsonl")
            files = glob.glob(file_pattern)
            
            print(f"Found {len(files)} train files in {detect_dir}")
            
            # Process each file
            for file_path in files:
                process_train_file(file_path)

if __name__ == "__main__":
    main()
    print("Processing complete!")