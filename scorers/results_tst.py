import os
import json
import glob
import numpy as np
from collections import defaultdict

def main():
    # Hardcoded parameters
    base_dirs = {
        'neutral_new': "/scratch/users/k21157437/neutral_new/data"
    }
    languages = ['en']  # English only
    generator_models = ['GPT-4o mini', 'Gemini 2.0', 'Qwen 2.5', 'Mistral']
    output_file = 'english_tst_table.tex'
    
    language_display_names = {
        'en': 'English Text Style Transfer'
    }
    
    # Define detector mappings
    detector_display_names = {
        'zero_binoculars': ('White-box', 'Binoculars'),
        'zero_llr': ('White-box', 'LLR'),
        'zero_fastdetectgpt_white': ('White-box', 'FDGPT (WB)'),
        'zero_revise': ('Black-box', 'Revise'),
        'zero_gecscore': ('Black-box', 'GECScore'),
        'zero_fastdetectgpt_black': ('Black-box', 'FDGPT (BB)'),
        'train_hp_roberta': ('Supervised', 'xlm-RoBERTa'),
        'train_hp_mdeberta': ('Supervised', 'mDeBERTa')
    }
    
    # Define detector file mappings
    zero_detector_file_mappings = {
        'zero_binoculars': 'binoculars',
        'zero_llr': 'llr',
        'zero_fastdetectgpt_white': 'fastdetectgpt_white',
        'zero_revise': 'revise',
        'zero_gecscore': 'gecscore',
        'zero_fastdetectgpt_black': 'fastdetectgpt_black'
    }
    
    train_detector_file_mappings = {
        'train_hp_roberta': 'roberta',
        'train_hp_mdeberta': 'mdeberta'
    }
    
    # Group detectors by family for easier access
    detector_families = defaultdict(list)
    for detector_key, (family, _) in detector_display_names.items():
        detector_families[family].append(detector_key)
    
    # Define task names with proper display names and subtasks
    task_info = [
        ('neutral_new', 'Text Style Transfer', [('paras', 'few5')])
    ]

    # Model name mapping for file paths
    model_id_mapping = {
        'GPT-4o mini': 'gpt',
        'Gemini 2.0': 'gemini',
        'Qwen 2.5': 'qwen',
        'Mistral': 'mistral'
    }
    
    # Remove the missing_files list since we're printing errors immediately
    results_files = {}
    
    for dir_name, task_display, subtasks in task_info:
        base_dir = base_dirs[dir_name]
        
        for task_prefix, modifier in subtasks:
            # Create a task_key for this specific task/subtask
            task_key = dir_name
            
            for lang in languages:
                # Construct path to detect directory for this language
                detect_dir = os.path.join(base_dir, lang, "detect")
                
                for model in generator_models:
                    model_id = model_id_mapping[model]
                    
                    # Handle zero-shot detector files
                    for detector_key, detector_suffix in zero_detector_file_mappings.items():
                        if dir_name == 'neutral_new':
                            file_path = os.path.join(detect_dir, f"{lang}_{task_prefix}_mgt_{modifier}_{model_id}_{detector_key}.jsonl")
                        
                        if os.path.exists(file_path):
                            key = (task_key, lang, model, detector_key)
                            results_files[key] = file_path
                        else:
                            print(f"File not found: {file_path}")
                    
                    # Handle trained detector files
                    for detector_key, detector_suffix in train_detector_file_mappings.items():
                        if dir_name == 'neutral_new':
                            file_path = os.path.join(detect_dir, f"{lang}_{task_prefix}_mgt_{modifier}_{model_id}_{detector_key}.jsonl")
                        
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
                    print(max_date)
                    most_recent_data = [item for item in data if item["start_date"] == max_date]
                    if len(most_recent_data) != 12:
                        print(f'\n[WARNING]    Incomplete HP run for {file_path}\n')

                    highest_val_acc = max(most_recent_data, key=lambda x: x["val_accuracy"])
                    results_data[key] = (highest_val_acc.get('test_accuracy'), highest_val_acc.get('test_f1'))

        except Exception as e:
            print(f"Exception {e} for: {file_path}")
    
    print(results_data)
    
    # Initialize dictionaries to store average results
    avg_results = {}
    family_avg_results = {}
    
    for task_key in set(k[0] for k in results_data.keys()):
        for lang in languages:
            # Calculate averages for individual detectors across all models
            for detector_key in detector_display_names.keys():
                # Get all scores for this detector across models
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
                
                # Calculate and store average for this detector across models
                if acc_scores:
                    avg_acc = sum(acc_scores) / len(acc_scores)
                    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else None
                    avg_results[(task_key, lang, detector_key)] = (avg_acc, avg_f1)
            
            # Calculate family averages
            for family, detector_list in detector_families.items():
                # For each model column, calculate family average (average of detectors for that model)
                for model in generator_models:
                    model_family_acc_list = []
                    model_family_f1_list = []
                    
                    # Collect all detector scores for this model and family
                    for detector_key in detector_list:
                        key = (task_key, lang, model, detector_key)
                        if key in results_data:
                            acc, f1 = results_data[key]
                            if acc is not None:
                                model_family_acc_list.append(acc)
                            if f1 is not None:
                                model_family_f1_list.append(f1)
                    
                    # Calculate family average for this model column
                    if model_family_acc_list:
                        model_family_avg_acc = sum(model_family_acc_list) / len(model_family_acc_list)
                        model_family_avg_f1 = sum(model_family_f1_list) / len(model_family_f1_list) if model_family_f1_list else None
                        family_avg_results[(task_key, lang, model, family)] = (model_family_avg_acc, model_family_avg_f1)
                
                # Now calculate family average for the "Avg" column (averaging across all detectors in family)
                all_detector_avgs_acc = []
                all_detector_avgs_f1 = []
                
                # Get the average of each detector in this family
                for detector_key in detector_list:
                    avg_key = (task_key, lang, detector_key)
                    if avg_key in avg_results:
                        avg_acc, avg_f1 = avg_results[avg_key]
                        if avg_acc is not None:
                            all_detector_avgs_acc.append(avg_acc)
                        if avg_f1 is not None:
                            all_detector_avgs_f1.append(avg_f1)
                
                # Calculate overall family average for the "Avg" column
                if all_detector_avgs_acc:
                    overall_family_avg_acc = sum(all_detector_avgs_acc) / len(all_detector_avgs_acc)
                    overall_family_avg_f1 = sum(all_detector_avgs_f1) / len(all_detector_avgs_f1) if all_detector_avgs_f1 else None
                    family_avg_results[(task_key, lang, "avg", family)] = (overall_family_avg_acc, overall_family_avg_f1)
    
    # Generate LaTeX table
    latex_code = []
    
    # Get number of models for column calculations
    num_models = len(generator_models)
    
    # Column specification for English only (no task column)
    # 1 column for Detector + 10 columns for models and average
    latex_code.append("\\begin{tabular}{lcccccccccc}")
    
    # Header rows
    latex_code.append("\\toprule")
    
    # First header row with language group (only English)
    header_row = ["\\textbf{Detector}"]
    header_row.append("\\multicolumn{10}{c}{\\textbf{" + language_display_names['en'] + "}}")
    latex_code.append(" & ".join(header_row) + " \\\\")
    
    # Add cmidrule for English section
    latex_code.append("\\cmidrule(lr){2-11}")
    
    # Second header row with model names (with ACC/F1 subcolumns) and Avg
    second_header_row = [""]  # Empty cell for Detector column
    
    # Add 4 models with 2 columns each (ACC/F1)
    for model in generator_models:
        second_header_row.append(f"\\multicolumn{{2}}{{c}}{{{model}}}")
    
    # Add avg column
    second_header_row.append(f"\\multicolumn{{2}}{{c}}{{\\textbf{{Avg}}}}")
    
    latex_code.append(" & ".join(second_header_row) + " \\\\")
    
    # Add cmidrule separators for each model's ACC/F1 columns
    cmidrule_parts = []
    col_idx = 2
    for model_idx in range(num_models + 1):  # +1 for the Avg column
        cmidrule_parts.append(f"\\cmidrule(lr){{{col_idx}-{col_idx+1}}}")
        col_idx += 2
    
    latex_code.append(" ".join(cmidrule_parts))
    
    # Third header row with ACC/F1 labels
    third_header_row = [""]  # Empty cell for Detector column
    
    # Add ACC/F1 labels for each model and average
    for _ in range(num_models + 1):
        third_header_row.extend(["ACC", "F1"])
    
    latex_code.append(" & ".join(third_header_row) + " \\\\")
    latex_code.append("\\midrule")
    
    # Calculate total number of columns in the table (for cdashline)
    total_cols = 11  # 1 (Detector) + 10 (Models + Avg)
    
    # Data rows for each task
    for dir_name, task_display, subtasks in task_info:
        for task_prefix, modifier in subtasks:
            # Create a task_key for this specific task/subtask
            task_key = dir_name
            
            # Group detectors by family
            detectors_by_family = defaultdict(list)
            for detector_key, (family, name) in detector_display_names.items():
                detectors_by_family[family].append((detector_key, name))
            
            for family, detectors in detectors_by_family.items():
                # Process individual detectors
                for detector_key, detector_name in detectors:
                    row = []
                    
                    # Add detector name
                    row.append(detector_name)
                    
                    # Add metrics for English
                    lang = 'en'
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
                    
                    # Add average column
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
                    
                    latex_code.append(" & ".join(row) + " \\\\")
                
                # Add dotted line
                latex_code.append("\\cdashline{1-11} \\addlinespace[1pt]")
                
                # Add family average row (no italics for values)
                avg_row = [f"Avg ({family})"]
                
                lang = 'en'
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
    
    print('\n\n\n')
    print(latex_output)
    print('\n\n\n')

if __name__ == "__main__":
    main()