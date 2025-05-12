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
            for family in detector_families.keys():
                # Get the average of each detector in this family
                all_detector_avgs_acc = []
                all_detector_avgs_f1 = []
                
                for detector_key in detector_families[family]:
                    avg_key = (task_key, lang, detector_key)
                    if avg_key in avg_results:
                        avg_acc, avg_f1 = avg_results[avg_key]
                        if avg_acc is not None:
                            all_detector_avgs_acc.append(avg_acc)
                        if avg_f1 is not None:
                            all_detector_avgs_f1.append(avg_f1)
                
                # Calculate overall family average
                if all_detector_avgs_acc:
                    overall_family_avg_acc = sum(all_detector_avgs_acc) / len(all_detector_avgs_acc)
                    overall_family_avg_f1 = sum(all_detector_avgs_f1) / len(all_detector_avgs_f1) if all_detector_avgs_f1 else None
                    family_avg_results[(task_key, lang, family)] = (overall_family_avg_acc, overall_family_avg_f1)
    
    # Generate LaTeX table
    latex_code = []
    
    # Column specification - just Detector, ACC, and F1
    latex_code.append("\\begin{tabular}{lcc}")
    
    # Header rows
    latex_code.append("\\toprule")
    
    # First header row with language
    header_row = ["\\textbf{Detector}"]
    header_row.append("\\multicolumn{2}{c}{\\textbf{" + language_display_names['en'] + "}}")
    latex_code.append(" & ".join(header_row) + " \\\\")
    
    # Add cmidrule for section
    latex_code.append("\\cmidrule(lr){2-3}")
    
    # Second header row with ACC/F1 labels
    second_header_row = ["", "ACC", "F1"]
    latex_code.append(" & ".join(second_header_row) + " \\\\")
    latex_code.append("\\midrule")
    
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
                    
                    # Add average acc and f1 (times 100, with one decimal)
                    avg_key = (task_key, lang, detector_key)
                    avg_values = avg_results.get(avg_key, (None, None))
                    
                    # Add average accuracy
                    if avg_values and avg_values[0] is not None:
                        row.append(f"\\greygra{{{avg_values[0] * 100:.1f}}}")
                    else:
                        row.append("")
                    
                    # Add average F1 score
                    if avg_values and avg_values[1] is not None:
                        row.append(f"\\greygra{{{avg_values[1] * 100:.1f}}}")
                    else:
                        row.append("")
                    
                    latex_code.append(" & ".join(row) + " \\\\")
                
                # Add dotted line
                latex_code.append("\\cdashline{1-3} \\addlinespace[1pt]")
                
                # Add family average row
                avg_row = [f"Avg ({family})"]
                
                lang = 'en'
                # Add family average values
                key = (task_key, lang, family)
                values = family_avg_results.get(key, (None, None))
                
                # Add family average accuracy
                if values and values[0] is not None:
                    avg_row.append(f"\\textbf{{\\greygra{{{values[0] * 100:.1f}}}}}")
                else:
                    avg_row.append("")
                
                # Add family average F1
                if values and values[1] is not None:
                    avg_row.append(f"\\textbf{{\\greygra{{{values[1] * 100:.1f}}}}}")
                else:
                    avg_row.append("")
                
                latex_code.append(" & ".join(avg_row) + " \\\\")
                
                # Add space after family
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