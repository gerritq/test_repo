import os
import json
import glob
import numpy as np
from collections import defaultdict

def main():
    # Hardcoded parameters
    base_dirs = {
        'sums': "/scratch/users/k21157437/sums/data",
        'neutral_new': "/scratch/users/k21157437/neutral_new/data",
        'paras': "/scratch/users/k21157437/paras/data"
    }
    languages = ['en', 'pt', 'vi']
    generator_models = ['GPT-4o mini', 'Gemini 2.0', 'Qwen 2.5', 'Mistral']
    output_file = 'detection_results_table.tex'
    
    language_display_names = {
        'en': 'English',
        'pt': 'Portuguese',
        'vi': 'Vietnamese'
    }
    
    # Define detector mappings
    detector_display_names = {
        'zero_binoculars': ('White-box', 'Binoculars'),
        'zero_llr': ('White-box', 'LLR'),
        'zero_fastdetectgpt_white': ('White-box', 'FDGPT (WB)'),
        'zero_revisedetect': ('Black-box', 'Revise'),
        'zero_gecscore': ('Black-box', 'GECScore'),
        'zero_fastdetectgpt_black': ('Black-box', 'FDGPT (BB)'),
        'train_roberta': ('Supervised', 'xlm-RoBERTa'),
        'train_mdeberta': ('Supervised', 'mDeBERTa')
    }
    
    # Group detectors by family for easier access
    detector_families = defaultdict(list)
    for detector_key, (family, _) in detector_display_names.items():
        detector_families[family].append(detector_key)
    
    # Define task names with proper display names and subtasks
    task_info = [
        ('paras', 'Introductory Paragraph', [('first', 'rag')]),
        ('paras', 'Paragraph Continuation', [('extend', 'rag')]),
        ('sums', 'Summarisation', [('sums', 'few1')]),
        ('neutral_new', 'Text Style Transfer', [('default', 'few5')])
    ]

    # Model name mapping for file paths
    model_id_mapping = {
        'GPT-4o mini': 'gpt',
        'Gemini 2.0': 'gemini',
        'Qwen 2.5': 'qwen',
        'Mistral': 'mistral'
    }
    
    # Define detector file mappings (the suffix to use in filenames)
    zero_detector_file_mappings = {
        'zero_binoculars': 'binoculars',
        'zero_llr': 'llr',
        'zero_fastdetectgpt_white': 'fastdetectgpt_white',
        'zero_revisedetect': 'revise',
        'zero_gecscore': 'gecscore',
        'zero_fastdetectgpt_black': 'fastdetectgpt_black'
    }
    
    train_detector_file_mappings = {
        'train_roberta': 'roberta',
        'train_mdeberta': 'mdeberta'
    }
    
    # Results files dictionary
    results_files = {}
    
    for dir_name, task_display, subtasks in task_info:
        base_dir = base_dirs[dir_name]
        
        for task_prefix, modifier in subtasks:
            # Create a task_key for this specific task/subtask
            if dir_name == 'paras':
                task_key = f"{dir_name}_{task_prefix}"
            else:
                task_key = dir_name
            
            for lang in languages:
                # Construct path to detect directory for this language
                detect_dir = os.path.join(base_dir, lang, "detect")
                
                for model in generator_models:
                    model_id = model_id_mapping[model]
                    
                    # Handle zero-shot detector files - now with specific detector suffixes
                    for detector_key, detector_suffix in zero_detector_file_mappings.items():
                        if dir_name == 'sums':
                            # Special case for sums
                            if lang in ['pt', 'vi']:
                                file_path = os.path.join(detect_dir, f"{lang}_sums_mgt_{modifier}_{model_id}_zero_{detector_suffix}.jsonl")
                            else:
                                file_path = os.path.join(detect_dir, f"{lang}_sums_mgt_{modifier}_{model_id}_zero_{detector_suffix}.jsonl")
                        elif dir_name == 'neutral_new':
                            if lang in ['pt', 'vi']:
                                # No "default" in path for PT and VI in Text Style Transfer
                                file_path = os.path.join(detect_dir, f"{lang}_mgt_{modifier}_{model_id}_zero_{detector_suffix}.jsonl")
                            else:
                                file_path = os.path.join(detect_dir, f"{lang}_{task_prefix}_mgt_{modifier}_{model_id}_zero_{detector_suffix}.jsonl")
                        elif dir_name == 'paras':
                            file_path = os.path.join(detect_dir, f"{lang}_paras_{modifier}_{task_prefix}_{model_id}_zero_{detector_suffix}.jsonl")
                        
                        if os.path.exists(file_path):
                            key = (task_key, lang, model, detector_key)
                            results_files[key] = file_path
                        else:
                            print(f"File not found: {file_path}")
                    
                    # Handle trained detector files - now with specific detector suffixes
                    for detector_key, detector_suffix in train_detector_file_mappings.items():
                        if dir_name == 'sums':
                            # Special case for sums
                            if lang in ['pt', 'vi']:
                                file_path = os.path.join(detect_dir, f"{lang}_sums_mgt_{modifier}_{model_id}_train_hp_{detector_suffix}.jsonl")
                            else:
                                file_path = os.path.join(detect_dir, f"{lang}_sums_mgt_{modifier}_{model_id}_train_hp_{detector_suffix}.jsonl")
                        elif dir_name == 'neutral_new':
                            if lang in ['pt', 'vi']:
                                # No "default" in path for PT and VI in Text Style Transfer
                                file_path = os.path.join(detect_dir, f"{lang}_mgt_{modifier}_{model_id}_train_hp_{detector_suffix}.jsonl")
                            else:
                                file_path = os.path.join(detect_dir, f"{lang}_{task_prefix}_mgt_{modifier}_{model_id}_train_hp_{detector_suffix}.jsonl")
                        elif dir_name == 'paras':
                            file_path = os.path.join(detect_dir, f"{lang}_paras_{modifier}_{task_prefix}_{model_id}_train_hp_{detector_suffix}.jsonl")
                        
                        if os.path.exists(file_path):
                            key = (task_key, lang, model, detector_key)
                            results_files[key] = file_path
                        else:
                            print(f"File not found: {file_path}")
    
    # Extract metrics for each combination
    results_data = {}
    
    for key, file_path in results_files.items():
        task, lang, model, detector = key
        
        try:
            #detector_key = detector[5:]  # Remove 'zero_' prefix
            if detector.startswith('zero'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    accuracy = data.get('accuracy')
                    f1 = data.get('f1')
                    results_data[key] = (accuracy, f1)
            else:
                with open(file_path, 'r') as f:
                    data = [json.loads(line) for line in f]
                    # find the most recent run
                    max_date = max(item["start_date"] for item in data)
                    #print(max_date)
                    most_recent_data = [item for item in data if item["start_date"] == max_date]
                    if len(most_recent_data) != 12:
                        print(f'\n[WARNING]    Incomplete HP run for {file_path}\n')

                    highest_val_acc = max(most_recent_data, key=lambda x: x["val_accuracy"])
                    results_data[key] = (highest_val_acc.get('test_accuracy'), highest_val_acc.get('test_f1'))

        except Exception as e:
            print(f"Exception {e} for: {file_path}")
        # except json.JSONDecodeError:
        #     print(f"Error parsing JSON in file: {file_path}")
        # except Exception as e:
        #     print(f"Error extracting metrics from {file_path}: {e}")
    
    # Calculate averages for each task/lang/detector combination
    avg_results = {}
    family_avg_results = {}
    
    for task_key in set(k[0] for k in results_data.keys()):
        for lang in languages:
            # Calculate averages for individual detectors
            for detector_key in detector_display_names.keys():
                # Get all scores for this task/lang/detector across models
                acc_scores = []
                f1_scores = []
                for model in generator_models:
                    key = (task_key, lang, model, detector_key)
                    if key in results_data:
                        acc, f1 = results_data[key]
                        if acc is not None:
                            acc_scores.append(acc)
                        if f1 is not None:
                            f1_scores.append(f1)
                
                # Calculate and store averages if we have scores
                if acc_scores:
                    avg_acc = sum(acc_scores) / len(acc_scores)
                    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else None
                    avg_results[(task_key, lang, detector_key)] = (avg_acc, avg_f1)
            
            # Calculate family averages
            for family, detector_list in detector_families.items():
                # For each model
                for model in generator_models:
                    # Collect scores for all detectors in this family
                    family_acc_scores = []
                    family_f1_scores = []
                    
                    for detector_key in detector_list:
                        key = (task_key, lang, model, detector_key)
                        if key in results_data:
                            acc, f1 = results_data[key]
                            if acc is not None:
                                family_acc_scores.append(acc)
                            if f1 is not None:
                                family_f1_scores.append(f1)
                    
                    # Calculate and store family averages
                    if family_acc_scores:
                        family_avg_acc = sum(family_acc_scores) / len(family_acc_scores)
                        family_avg_f1 = sum(family_f1_scores) / len(family_f1_scores) if family_f1_scores else None
                        family_avg_results[(task_key, lang, model, family)] = (family_avg_acc, family_avg_f1)
                
                # Calculate family average across all models (for the Avg column)
                all_family_acc = []
                all_family_f1 = []
                
                for detector_key in detector_list:
                    avg_key = (task_key, lang, detector_key)
                    if avg_key in avg_results:
                        avg_acc, avg_f1 = avg_results[avg_key]
                        if avg_acc is not None:
                            all_family_acc.append(avg_acc)
                        if avg_f1 is not None:
                            all_family_f1.append(avg_f1)
                
                if all_family_acc:
                    overall_family_avg_acc = sum(all_family_acc) / len(all_family_acc)
                    overall_family_avg_f1 = sum(all_family_f1) / len(all_family_f1) if all_family_f1 else None
                    family_avg_results[(task_key, lang, "avg", family)] = (overall_family_avg_acc, overall_family_avg_f1)
    
    # Generate LaTeX table
    latex_code = []
    
    # Calculate column specification
    num_languages = len(languages)
    num_models = len(generator_models)
    # Each model needs 2 columns for ACC and F1, plus 2 more for the average
    cols_per_model = 2
    cols_per_avg = 2
    cols_per_language = (num_models * cols_per_model) + cols_per_avg
    
    # Use the provided column specification format with p{1in} as dummy columns
    # Format: l l + (language columns) + |p{1in}| + (language columns) + |p{1in}| + (language columns)
    col_spec = "ll"
    
    # Add first language columns
    col_spec += "c" * cols_per_language
    
    # Add dummy column and second language columns
    col_spec += " p{.1in} " + "c" * cols_per_language
    
    # Add dummy column and third language columns
    col_spec += " p{.1in} " + "c" * cols_per_language
    
    # Use the exact specified tabular environment
    latex_code.append("\\begin{tabular}{llcccccccccc p{.1in} cccccccccc p{.1in} cccccccccc}")
    
    # Header rows
    latex_code.append("\\toprule")
    
    # First header row with language groups
    # Using the exact format from the example
    header_row = ["\\textbf{Task}", "\\textbf{Detector}"]
    
    # First language (English)
    header_row.append(f"\\multicolumn{{10}}{{c}}{{\\textbf{{{language_display_names['en']}}}}}")
    
    # Dummy column
    header_row.append("")
    
    # Second language (Portuguese)
    header_row.append(f"\\multicolumn{{10}}{{c}}{{\\textbf{{{language_display_names['pt']}}}}}")
    
    # Dummy column
    header_row.append("")
    
    # Third language (Vietnamese)
    header_row.append(f"\\multicolumn{{10}}{{c}}{{\\textbf{{{language_display_names['vi']}}}}}")
    
    latex_code.append(" & ".join(header_row) + " \\\\")
    
    # Calculate positions for cmidrules, accounting for dummy separator columns
    cmidrule_parts = []
    
    # First language group (columns 3-12)
    cmidrule_parts.append("\\cmidrule(lr){3-12}")
    
    # Second language group (columns 14-23)
    cmidrule_parts.append("\\cmidrule(lr){14-23}")
    
    # Third language group (columns 25-34)
    cmidrule_parts.append("\\cmidrule(lr){25-34}")
    
    latex_code.append(" ".join(cmidrule_parts))
    
    # Second header row with model names (with ACC/F1 subcolumns) and Avg
    second_header_row = ["", ""]  # Empty cells for Task and Detector columns
    
    # For each language
    for lang_idx in range(num_languages):
        # Add 4 models with 2 columns each (ACC/F1)
        for model in generator_models:
            second_header_row.append(f"\\multicolumn{{2}}{{c}}{{{model}}}")
        
        # Add avg column at the end of each language section
        second_header_row.append(f"\\multicolumn{{2}}{{c}}{{\\textbf{{Avg}}}}")
        
        # Add empty cell for dummy column after each language (except the last)
        if lang_idx < num_languages - 1:
            second_header_row.append("")
    
    latex_code.append(" & ".join(second_header_row) + " \\\\")
    
    # Add cmidrule separators for each model's ACC/F1 columns
    cmidrule_parts = []
    
    # English columns (3-12)
    col_idx = 3
    for model_idx in range(num_models + 1):  # +1 for the Avg column
        cmidrule_parts.append(f"\\cmidrule(lr){{{col_idx}-{col_idx+1}}}")
        col_idx += 2
    
    # Portuguese columns (14-23)
    col_idx = 14
    for model_idx in range(num_models + 1):  # +1 for the Avg column
        cmidrule_parts.append(f"\\cmidrule(lr){{{col_idx}-{col_idx+1}}}")
        col_idx += 2
    
    # Vietnamese columns (25-34)
    col_idx = 25
    for model_idx in range(num_models + 1):  # +1 for the Avg column
        cmidrule_parts.append(f"\\cmidrule(lr){{{col_idx}-{col_idx+1}}}")
        col_idx += 2
    
    latex_code.append(" ".join(cmidrule_parts))
    
    # Third header row with ACC/F1 labels
    third_header_row = ["", ""]  # Empty cells for Task and Detector columns
    
    for lang_idx in range(num_languages):
        # Add ACC/F1 labels for each model and average
        for _ in range(num_models + 1):
            third_header_row.extend(["ACC", "F1"])
        
        # Add empty cell for dummy column after each language (except the last)
        if lang_idx < num_languages - 1:
            third_header_row.append("")
    
    latex_code.append(" & ".join(third_header_row) + " \\\\")
    latex_code.append("\\midrule")
    
    # Calculate total number of columns in the table (for cdashline)
    # Total includes: 2 (Task, Detector) + 10 (English) + 1 (dummy) + 10 (Portuguese) + 1 (dummy) + 10 (Vietnamese)
    total_cols = 34
    
    # Data rows for each task
    for dir_name, task_display, subtasks in task_info:
        for task_prefix, modifier in subtasks:
            # Create a task_key for this specific task/subtask
            if dir_name == 'paras':
                task_key = f"{dir_name}_{task_prefix}"
            else:
                task_key = dir_name
            
            # Group detectors by family
            detectors_by_family = defaultdict(list)
            for detector_key, (family, name) in detector_display_names.items():
                detectors_by_family[family].append((detector_key, name))
            
            # First row includes task name with multirow
            # Count all detectors plus family average rows
            num_detectors = sum(len(detectors) + 1 for detectors in detectors_by_family.values())
            first_row = True
            
            for family, detectors in detectors_by_family.items():
                # Process individual detectors
                for detector_key, detector_name in detectors:
                    row = []
                    
                    # Add task name with multirow on first row of the task
                    if first_row:
                        row.append(f"\\multirow{{{num_detectors}}}{{*}}{{{task_display}}}")
                        first_row = False
                    else:
                        row.append("")
                    
                    # Add detector name
                    row.append(detector_name)
                    
                    # Add metrics for each language and model
                    for lang_idx, lang in enumerate(languages):
                        # Add individual model metrics
                        for model in generator_models:
                            key = (task_key, lang, model, detector_key)
                            values = results_data.get(key, (None, None))
                            
                            # Add accuracy
                            if values and values[0] is not None:
                                row.append(f"{values[0]:.2f}")
                            else:
                                row.append("")
                            
                            # Add F1 score
                            if values and values[1] is not None:
                                row.append(f"{values[1]:.2f}")
                            else:
                                row.append("")
                        
                        # Add average column for this language
                        avg_key = (task_key, lang, detector_key)
                        avg_values = avg_results.get(avg_key, (None, None))
                        
                        # Add average accuracy - make it bold
                        if avg_values and avg_values[0] is not None:
                            row.append(f"\\textbf{{\\greygra{{{avg_values[0]:.2f}}}}}")
                        else:
                            row.append("")
                        
                        # Add average F1 score - make it bold
                        if avg_values and avg_values[1] is not None:
                            row.append(f"\\textbf{{\\greygra{{{avg_values[1]:.2f}}}}}")
                        else:
                            row.append("")
                        
                        # Add empty cell for dummy column after each language (except the last)
                        if lang_idx < num_languages - 1:
                            row.append("")
                    
                    latex_code.append(" & ".join(row) + " \\\\")
                
                # Add dotted line that spans all columns (including dummy columns)
                latex_code.append("\\cdashline{2-34} \\addlinespace[1pt]")
                
                # Add family average row (no italics for values as requested)
                avg_row = ["", f"Avg. {family}"]
                
                for lang_idx, lang in enumerate(languages):
                    # Add model-specific average values
                    for model in generator_models:
                        key = (task_key, lang, model, family)
                        values = family_avg_results.get(key, (None, None))
                        
                        # Add family average accuracy (without italics)
                        if values and values[0] is not None:
                            avg_row.append(f"\\greygra{{{values[0]:.2f}}}")
                        else:
                            avg_row.append("")
                        
                        # Add family average F1 (without italics)
                        if values and values[1] is not None:
                            avg_row.append(f"\\greygra{{{values[1]:.2f}}}")
                        else:
                            avg_row.append("")
                    
                    # Add overall average for this family across all models
                    key = (task_key, lang, "avg", family)
                    values = family_avg_results.get(key, (None, None))
                    
                    # Add overall family average accuracy (only bold, no italics)
                    if values and values[0] is not None:
                        avg_row.append(f"\\textbf{{\\greygra{{{values[0]:.2f}}}}}")
                    else:
                        avg_row.append("")
                    
                    # Add overall family average F1 (only bold, no italics)
                    if values and values[1] is not None:
                        avg_row.append(f"\\textbf{{\\greygra{{{values[1]:.2f}}}}}")
                    else:
                        avg_row.append("")
                    
                    # Add empty cell for dummy column after each language (except the last)
                    if lang_idx < num_languages - 1:
                        avg_row.append("")
                
                latex_code.append(" & ".join(avg_row) + " \\\\")
                
                # Use addlinespace as before
                latex_code.append("\\addlinespace[3pt]")
            
            # Add midrule after each task
            latex_code.append("\\midrule")
    
    # Table footer
    latex_code.append("\\bottomrule")
    latex_code.append("\\end{tabular}")
    
    # Write to file
    latex_output = "\n".join(latex_code)
    with open(output_file, 'w') as f:
        f.write(latex_output)
    
    
    # Also print the table to console
    print("\n\n\n")
    print(latex_output)
    print("\n\n\n")

if __name__ == "__main__":
    main()