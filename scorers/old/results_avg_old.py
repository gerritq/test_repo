import os
import json
import glob
import numpy as np
from collections import defaultdict


all_results {'first': {'en':}}

def main():
    # Hardcoded parameters
    base_dirs = {
        'sums': "/scratch/users/k21157437/sums/data",
        'neutral_new': "/scratch/users/k21157437/neutral_new/data",
        'paras': "/scratch/users/k21157437/paras/data"
    }
    languages = ['en', 'pt', 'vi']
    generator_models = ['GPT-4o mini', 'Gemini 2.0', 'Qwen 2.5', 'Mistral']
    output_file = 'detection_results_single_table.tex'
    
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
    
    # Generate single large table
    generate_single_large_table(task_info, languages, language_display_names, 
                               detector_display_names, detector_families, 
                               avg_results, family_avg_results, output_file)

def generate_single_large_table(task_info, languages, language_display_names, 
                               detector_display_names, detector_families, 
                               avg_results, family_avg_results, output_file):
    """Generate a single large table with all 4 tasks (2 tasks per row)"""
    latex_code = []
    
    # Process tasks in pairs
    task_pairs = []
    for i in range(0, len(task_info), 2):
        if i + 1 < len(task_info):
            task_pairs.append((task_info[i], task_info[i+1]))
        else:
            # If odd number of tasks, pair the last one with None
            task_pairs.append((task_info[i], None))
    
    # Column specification for the paired table - 14 columns total
    # 1 for detector column + (3*2) for first task languages + 1 dummy column + (3*2) for second task languages
    # Each language has 2 columns (ACC and F1)
    col_spec = "l" + "cc"*3 + "c" + "cc"*3
    
    latex_code.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_code.append("\\toprule")
    
    # Process each pair of tasks
    for pair_idx, task_pair in enumerate(task_pairs):
        if pair_idx > 0:
            # Add a separator between pairs
            latex_code.append("\\midrule[1.5pt]")
        
        # First task in the pair
        task1, task1_display, subtasks1 = task_pair[0]
        
        # Check if there's a second task
        has_second_task = task_pair[1] is not None
        if has_second_task:
            task2, task2_display, subtasks2 = task_pair[1]
        
        # Process each subtask
        for subtask_idx, (task1_prefix, modifier1) in enumerate(subtasks1):
            # Create task_key for first task
            if task1 == 'paras':
                task1_key = f"{task1}_{task1_prefix}"
            else:
                task1_key = task1
            
            # Get task_key for second task if it exists
            task2_key = None
            if has_second_task and subtask_idx < len(subtasks2):
                task2_prefix, modifier2 = subtasks2[subtask_idx]
                if task2 == 'paras':
                    task2_key = f"{task2}_{task2_prefix}"
                else:
                    task2_key = task2
            
            # Add task headers row - only add detector name in first panel
            if pair_idx == 0:
                task_header = [""]
                task_header.extend([f"\\multicolumn{{6}}{{c}}{{\\textbf{{{task1_display}}}}}"]) 
                task_header.append("")  # Dummy column
            else:
                task_header = ["\\multicolumn{1}{c}{}"]  # Empty cell with proper alignment
                task_header.extend([f"\\multicolumn{{6}}{{c}}{{\\textbf{{{task1_display}}}}}"]) 
                task_header.append("")  # Dummy column
            
            if task2_key:
                task_header.extend([f"\\multicolumn{{6}}{{c}}{{\\textbf{{{task2_display}}}}}"]) 
            
            latex_code.append(" & ".join(task_header) + " \\\\")
            latex_code.append("\\cmidrule(lr){2-7}" + (" \\cmidrule(lr){9-14}" if task2_key else ""))
            
            # Add language headers row - only add detector name in first panel
            if pair_idx == 0:
                header_row = ["\\textbf{Detector}"]
            else:
                header_row = ["\\multicolumn{1}{c}{}"]  # Empty cell with proper alignment
            
            # Add language headers for first task
            for lang in languages:
                header_row.append(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{language_display_names[lang]}}}}}")
            
            header_row.append("")  # Dummy column
            
            # Add language headers for second task if it exists
            if task2_key:
                for lang in languages:
                    header_row.append(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{language_display_names[lang]}}}}}")
            
            latex_code.append(" & ".join(header_row) + " \\\\")
            
            # Add cmidrule separators for each language
            cmidrules = []
            col = 2
            for _ in range(len(languages)):
                cmidrules.append(f"\\cmidrule(lr){{{col}-{col+1}}}")
                col += 2
            
            # Skip dummy column
            col += 1
            
            if task2_key:
                for _ in range(len(languages)):
                    cmidrules.append(f"\\cmidrule(lr){{{col}-{col+1}}}")
                    col += 2
                    
            latex_code.append(" ".join(cmidrules))
            
            # Add metric type headers (ACC/F1)
            metrics_row = [""]
            for _ in range(len(languages)):
                metrics_row.extend(["ACC", "F1"])
            
            metrics_row.append("")  # Dummy column
            
            if task2_key:
                for _ in range(len(languages)):
                    metrics_row.extend(["ACC", "F1"])
                    
            latex_code.append(" & ".join(metrics_row) + " \\\\")
            
            latex_code.append("\\midrule")
            
            # Group detectors by family
            detectors_by_family = defaultdict(list)
            for detector_key, (family, name) in detector_display_names.items():
                detectors_by_family[family].append((detector_key, name))
            
            # Process each detector family
            for family_idx, (family, detectors) in enumerate(detectors_by_family.items()):
                # Process individual detectors
                for detector_key, detector_name in detectors:
                    row = []
                    
                    # Add detector name
                    row.append(detector_name)
                    
                    # Add metrics for first task
                    for lang in languages:
                        avg_key = (task1_key, lang, detector_key)
                        avg_values = avg_results.get(avg_key, (None, None))
                        
                        # Add average accuracy
                        if avg_values and avg_values[0] is not None:
                            row.append(f"\\greygra{{{avg_values[0]*100:.1f}}}")
                        else:
                            row.append("")
                        
                        # Add average F1
                        if avg_values and avg_values[1] is not None:
                            row.append(f"\\greygra{{{avg_values[1]*100:.1f}}}")
                        else:
                            row.append("")
                    
                    # Add dummy column
                    row.append("")
                    
                    # Add metrics for second task if it exists
                    if task2_key:
                        for lang in languages:
                            avg_key = (task2_key, lang, detector_key)
                            avg_values = avg_results.get(avg_key, (None, None))
                            
                            # Add average accuracy
                            if avg_values and avg_values[0] is not None:
                                row.append(f"\\greygra{{{avg_values[0]*100:.1f}}}")
                            else:
                                row.append("")
                            
                            # Add average F1
                            if avg_values and avg_values[1] is not None:
                                row.append(f"\\greygra{{{avg_values[1]*100:.1f}}}")
                            else:
                                row.append("")
                    
                    latex_code.append(" & ".join(row) + " \\\\")
                
                # Add dotted lines separately for each side of the table
                latex_code.append(f"\\cdashline{{1-7}}")  # First task
                if task2_key:
                    latex_code.append(f"\\cdashline{{9-14}}")  # Second task
                latex_code.append("\\addlinespace[4pt]")
                
                # Add family average row
                avg_row = [f"\\textbf{{Avg. {family}}}"]
                
                # Add family average for first task
                for lang in languages:
                    key = (task1_key, lang, family)
                    values = family_avg_results.get(key, (None, None))
                    
                    # Add family average accuracy
                    if values and values[0] is not None:
                        avg_row.append(f"\\textbf{{\\greygra{{{values[0]*100:.1f}}}}}") 
                    else:
                        avg_row.append("")
                    
                    # Add family average F1
                    if values and values[1] is not None:
                        avg_row.append(f"\\textbf{{\\greygra{{{values[1]*100:.1f}}}}}") 
                    else:
                        avg_row.append("")
                
                # Add dummy column
                avg_row.append("")
                
                # Add family average for second task if it exists
                if task2_key:
                    for lang in languages:
                        key = (task2_key, lang, family)
                        values = family_avg_results.get(key, (None, None))
                        
                        # Add family average accuracy
                        if values and values[0] is not None:
                            avg_row.append(f"\\textbf{{\\greygra{{{values[0]*100:.1f}}}}}") 
                        else:
                            avg_row.append("")
                        
                        # Add family average F1
                        if values and values[1] is not None:
                            avg_row.append(f"\\textbf{{\\greygra{{{values[1]*100:.1f}}}}}") 
                        else:
                            avg_row.append("")
                
                latex_code.append(" & ".join(avg_row) + " \\\\")
                
                # Add space after each family section except the last one
                if family_idx < len(detectors_by_family) - 1:
                    latex_code.append("\\addlinespace[6pt]")
    
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