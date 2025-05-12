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
    output_file = 'detection_results_avg_table.tex'
    
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
                    
                    # Handle zero-shot detector files
                    for detector_key, detector_suffix in zero_detector_file_mappings.items():
                        if dir_name == 'sums':
                            if lang in ['pt', 'vi']:
                                file_path = os.path.join(detect_dir, f"{lang}_sums_mgt_{modifier}_{model_id}_zero_{detector_suffix}.jsonl")
                            else:
                                file_path = os.path.join(detect_dir, f"{lang}_sums_mgt_{modifier}_{model_id}_zero_{detector_suffix}.jsonl")
                        elif dir_name == 'neutral_new':
                            if lang in ['pt', 'vi']:
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
                    
                    # Handle trained detector files
                    for detector_key, detector_suffix in train_detector_file_mappings.items():
                        if dir_name == 'sums':
                            if lang in ['pt', 'vi']:
                                file_path = os.path.join(detect_dir, f"{lang}_sums_mgt_{modifier}_{model_id}_train_hp_{detector_suffix}.jsonl")
                            else:
                                file_path = os.path.join(detect_dir, f"{lang}_sums_mgt_{modifier}_{model_id}_train_hp_{detector_suffix}.jsonl")
                        elif dir_name == 'neutral_new':
                            if lang in ['pt', 'vi']:
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
            if detector.startswith('zero'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    accuracy = data.get('accuracy')
                    f1 = data.get('f1')
                    results_data[key] = (accuracy, f1)
            else:
                with open(file_path, 'r') as f:
                    data = [json.loads(line) for line in f]
                    max_date = max(item["start_date"] for item in data)
                    most_recent_data = [item for item in data if item["start_date"] == max_date]
                    if len(most_recent_data) != 12:
                        print(f'\n[WARNING]    Incomplete HP run for {file_path}\n')

                    highest_val_acc = max(most_recent_data, key=lambda x: x["val_accuracy"])
                    results_data[key] = (highest_val_acc.get('test_accuracy'), highest_val_acc.get('test_f1'))

        except Exception as e:
            print(f"Exception {e} for: {file_path}")
    
    # Calculate averages for each task/lang/detector combination
    avg_results = {}
    family_avg_results = {}
    row_avg_results = {}
    family_row_avg_results = {}
    
    for task_key in set(k[0] for k in results_data.keys()):
        for lang in languages:
            # Calculate averages for individual detectors
            for detector_key in detector_display_names.keys():
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
                
                if acc_scores:
                    avg_acc = sum(acc_scores) / len(acc_scores)
                    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else None
                    avg_results[(task_key, lang, detector_key)] = (avg_acc, avg_f1)
            
            # Calculate family averages
            for family, detector_list in detector_families.items():
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
                    family_avg_results[(task_key, lang, family)] = (overall_family_avg_acc, overall_family_avg_f1)
    
    # Calculate row averages (across languages) for each detector
    for task_key in set(k[0] for k in results_data.keys()):
        for detector_key in detector_display_names.keys():
            acc_scores = []
            f1_scores = []
            for lang in languages:
                avg_key = (task_key, lang, detector_key)
                if avg_key in avg_results:
                    avg_acc, avg_f1 = avg_results[avg_key]
                    if avg_acc is not None:
                        acc_scores.append(avg_acc)
                    if avg_f1 is not None:
                        f1_scores.append(avg_f1)
            
            if acc_scores:
                row_avg_acc = sum(acc_scores) / len(acc_scores)
                row_avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else None
                row_avg_results[(task_key, detector_key)] = (row_avg_acc, row_avg_f1)
        
        # Calculate family row averages
        for family, detector_list in detector_families.items():
            family_acc_scores = []
            family_f1_scores = []
            
            for lang in languages:
                family_key = (task_key, lang, family)
                if family_key in family_avg_results:
                    family_acc, family_f1 = family_avg_results[family_key]
                    if family_acc is not None:
                        family_acc_scores.append(family_acc)
                    if family_f1 is not None:
                        family_f1_scores.append(family_f1)
            
            if family_acc_scores:
                family_row_avg_acc = sum(family_acc_scores) / len(family_acc_scores)
                family_row_avg_f1 = sum(family_f1_scores) / len(family_f1_scores) if family_f1_scores else None
                family_row_avg_results[(task_key, family)] = (family_row_avg_acc, family_row_avg_f1)
    
    # Generate LaTeX table with task headers within the table
    latex_code = []
    
    # Column specification for the entire table
    col_spec = "l"  # Detector column
    for _ in range(len(languages)):
        col_spec += "cc"
    col_spec += "cc"  # For the row average
    
    latex_code.append(f"\\begin{{tabular}}{{{col_spec}}}")
    
    # Header rows
    latex_code.append("\\toprule")
    
    # First header row with language groups
    header_row = ["\\textbf{Detector}"]
    
    # Add language headers
    for lang in languages:
        header_row.append(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{language_display_names[lang]}}}}}")
    
    # Add row average header
    header_row.append("\\multicolumn{2}{c}{\\textbf{Avg}}")
    
    latex_code.append(" & ".join(header_row) + " \\\\")
    
    # Add cmidrule separators for language groups and row average
    cmidrule_parts = []
    col_start = 2
    for _ in range(len(languages)):
        cmidrule_parts.append(f"\\cmidrule(lr){{{col_start}-{col_start+1}}}")
        col_start += 2
    
    # Add cmidrule for row average
    cmidrule_parts.append(f"\\cmidrule(lr){{{col_start}-{col_start+1}}}")
    
    latex_code.append(" ".join(cmidrule_parts))
    
    # Second header row with ACC/F1 labels
    second_header_row = [""]  # Empty cell for Detector column
    
    for _ in range(len(languages) + 1):  # +1 for row average
        second_header_row.extend(["ACC", "F1"])
    
    latex_code.append(" & ".join(second_header_row) + " \\\\")
    latex_code.append("\\midrule")
    
    # Process each task as a section
    first_task = True
    for dir_name, task_display, subtasks in task_info:
        for task_prefix, modifier in subtasks:
            # Create a task_key for this specific task/subtask
            if dir_name == 'paras':
                task_key = f"{dir_name}_{task_prefix}"
            else:
                task_key = dir_name
            
            # Add task header as a panel header spanning all columns
            total_cols = 1 + (len(languages) * 2) + 2  # +2 for row average
            if not first_task:
                # Add space between task sections
                latex_code.append("\\midrule")
            else:
                first_task = False
                
            latex_code.append(f"\\multicolumn{{{total_cols}}}{{l}}{{\\textbf{{{task_display}}}}} \\\\")
            latex_code.append("\\midrule")
            
            # Group detectors by family
            detectors_by_family = defaultdict(list)
            for detector_key, (family, name) in detector_display_names.items():
                detectors_by_family[family].append((detector_key, name))
            
            # Process each detector family
            for family, detectors in detectors_by_family.items():
                # Process individual detectors
                for detector_key, detector_name in detectors:
                    row = []
                    
                    # Add detector name
                    row.append(detector_name)
                    
                    # Add average metrics for each language
                    for lang in languages:
                        avg_key = (task_key, lang, detector_key)
                        avg_values = avg_results.get(avg_key, (None, None))
                        
                        # Add average accuracy
                        if avg_values and avg_values[0] is not None:
                            row.append(f"{avg_values[0]:.2f}")
                        else:
                            row.append("")
                        
                        # Add average F1 score
                        if avg_values and avg_values[1] is not None:
                            row.append(f"{avg_values[1]:.2f}")
                        else:
                            row.append("")
                    
                    # Add row average (across languages)
                    row_avg_key = (task_key, detector_key)
                    row_avg_values = row_avg_results.get(row_avg_key, (None, None))
                    
                    # Add row average accuracy
                    if row_avg_values and row_avg_values[0] is not None:
                        row.append(f"\\textbf{{{row_avg_values[0]:.2f}}}")
                    else:
                        row.append("")
                    
                    # Add row average F1 score
                    if row_avg_values and row_avg_values[1] is not None:
                        row.append(f"\\textbf{{{row_avg_values[1]:.2f}}}")
                    else:
                        row.append("")
                    
                    latex_code.append(" & ".join(row) + " \\\\")
                
                # Add dotted line
                latex_code.append(f"\\cdashline{{1-{total_cols}}} \\addlinespace[1pt]")
                
                # Add family average row
                avg_row = [f"Avg. {family}"]
                
                for lang in languages:
                    # Add family average values
                    key = (task_key, lang, family)
                    values = family_avg_results.get(key, (None, None))
                    
                    # Add family average accuracy
                    if values and values[0] is not None:
                        avg_row.append(f"\\textbf{{\\greygra{{{values[0]:.2f}}}}}")
                    else:
                        avg_row.append("")
                    
                    # Add family average F1
                    if values and values[1] is not None:
                        avg_row.append(f"\\textbf{{\\greygra{{{values[1]:.2f}}}}}")
                    else:
                        avg_row.append("")
                
                # Add family row average (across languages)
                family_row_avg_key = (task_key, family)
                family_row_avg_values = family_row_avg_results.get(family_row_avg_key, (None, None))
                
                # Add family row average accuracy
                if family_row_avg_values and family_row_avg_values[0] is not None:
                    avg_row.append(f"\\textbf{{\\greygra{{{family_row_avg_values[0]:.2f}}}}}")
                else:
                    avg_row.append("")
                
                # Add family row average F1
                if family_row_avg_values and family_row_avg_values[1] is not None:
                    avg_row.append(f"\\textbf{{\\greygra{{{family_row_avg_values[1]:.2f}}}}}")
                else:
                    avg_row.append("")
                
                latex_code.append(" & ".join(avg_row) + " \\\\")
                
                # Use addlinespace
                latex_code.append("\\addlinespace[3pt]")
    
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