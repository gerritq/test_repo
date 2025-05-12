import os
import json
from collections import defaultdict

def main():
    # Hardcoded parameters
    base_dir = "/scratch/users/k21157437/mix/detect/lang_model"
    languages = ['en', 'pt', 'vi']
    generator_models = ['GPT-4o mini', 'Gemini 2.0', 'Qwen 2.5', 'Mistral']
    output_file = 'detection_results_with_models_table.tex'
    
    language_display_names = {
        'en': 'English',
        'pt': 'Portuguese',
        'vi': 'Vietnamese'
    }
    
    # Model name mapping for file paths
    model_id_mapping = {
        'GPT-4o mini': 'gpt',
        'Gemini 2.0': 'gemini',
        'Qwen 2.5': 'qwen',
        'Mistral': 'mistral'
    }
    
    # Define detector mappings
    detector_display_names = {
        'zero_binoculars': ('White-box', 'Binoculars'),
        'zero_llr': ('White-box', 'LLR'),
        'zero_fastdetectgpt_white': ('White-box', 'FDGPT (WB)'),
        'zero_revisedetect': ('Black-box', 'Revise'),
        'zero_gecscore': ('Black-box', 'GECScore'),
        'zero_fastdetectgpt_black': ('Black-box', 'FDGPT (BB)'),
        'train_hp_roberta': ('Supervised', 'xlm-RoBERTa'),
        'train_hp_mdeberta': ('Supervised', 'mDeBERTa')
    }
    
    # Group detectors by family for easier access
    detector_families = defaultdict(list)
    for detector_key, (family, _) in detector_display_names.items():
        detector_families[family].append(detector_key)
    
    # Results files dictionary
    results_files = {}
    
    for lang in languages:
        for model in generator_models:
            model_id = model_id_mapping[model]
            
            for detector_key in detector_display_names.keys():
                # Format: en_gemini_train_mdeberta.jsonl or en_gpt_zero_binoculars.jsonl
                file_path = os.path.join(base_dir, f"{lang}_{model_id}_{detector_key}.jsonl")
                
                if os.path.exists(file_path):
                    key = (lang, model, detector_key)
                    results_files[key] = file_path
                else:
                    print(f"File not found: {file_path}")
    
    # Extract metrics for each combination
    results_data = {}
    
    for key, file_path in results_files.items():
        lang, model, detector = key

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
                    most_recent_data = [item for item in data if item["start_date"] == max_date]
                    if len(most_recent_data) != 12:
                        print(f'\n[WARNING]    Incomplete HP run for {file_path}\n')

                    best_run = max(data, key=lambda x: x.get("accuracy", 0))
                    results_data[key] = (best_run.get('accuracy'), best_run.get('f1'))

        except Exception as e:
            print(f"Exception {e} for: {file_path}")
        
    
    # Calculate averages for each detector across models
    detector_avg_results = {}
    
    for lang in languages:
        for detector_key, (_, detector_name) in detector_display_names.items():
            # Get all scores for this lang/detector across models
            acc_scores = []
            f1_scores = []
            
            for model in generator_models:
                key = (lang, model, detector_key)
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
                detector_avg_results[(lang, detector_key)] = (avg_acc, avg_f1)
    
    # Calculate family averages for each model
    family_model_avg_results = {}
    
    for lang in languages:
        for model in generator_models:
            for family, detector_list in detector_families.items():
                # Collect scores for all detectors in this family for this model
                family_acc_scores = []
                family_f1_scores = []
                
                for detector_key in detector_list:
                    key = (lang, model, detector_key)
                    if key in results_data:
                        acc, f1 = results_data[key]
                        if acc is not None:
                            family_acc_scores.append(acc)
                        if f1 is not None:
                            family_f1_scores.append(f1)
                
                # Calculate and store family averages for this model
                if family_acc_scores:
                    family_avg_acc = sum(family_acc_scores) / len(family_acc_scores)
                    family_avg_f1 = sum(family_f1_scores) / len(family_f1_scores) if family_f1_scores else None
                    family_model_avg_results[(lang, model, family)] = (family_avg_acc, family_avg_f1)
    
    # Calculate overall family averages (across all models)
    family_overall_avg_results = {}
    
    for lang in languages:
        for family, detector_list in detector_families.items():
            # Collect all detector averages for this family
            all_family_acc = []
            all_family_f1 = []
            
            for detector_key in detector_list:
                avg_key = (lang, detector_key)
                if avg_key in detector_avg_results:
                    avg_acc, avg_f1 = detector_avg_results[avg_key]
                    if avg_acc is not None:
                        all_family_acc.append(avg_acc)
                    if avg_f1 is not None:
                        all_family_f1.append(avg_f1)
            
            # Calculate and store overall family averages
            if all_family_acc:
                overall_family_avg_acc = sum(all_family_acc) / len(all_family_acc)
                overall_family_avg_f1 = sum(all_family_f1) / len(all_family_f1) if all_family_f1 else None
                family_overall_avg_results[(lang, family)] = (overall_family_avg_acc, overall_family_avg_f1)
    
    # Generate LaTeX table
    latex_code = []
    
    # Calculate column specification
    num_languages = len(languages)
    num_models = len(generator_models)
    # Each model needs 2 columns for ACC and F1, plus 2 more for the average
    cols_per_model = 2
    cols_per_avg = 2
    cols_per_language = (num_models * cols_per_model) + cols_per_avg
    
    # Total number of columns: 1 for detector + (languages * (models * 2 + 2))
    total_cols = 1 + (num_languages * (num_models * cols_per_model + cols_per_avg))
    
    # Use the provided column specification format
    col_spec = "l" + "c" * (total_cols - 1)
    
    latex_code.append(f"\\begin{{tabular}}{{{col_spec}}}")
    
    # Header rows
    latex_code.append("\\toprule")
    
    # First header row with language groups
    header_row = ["\\textbf{Detector}"]
    
    for lang in languages:
        header_row.append(f"\\multicolumn{{{cols_per_language}}}{{c}}{{\\textbf{{{language_display_names[lang]}}}}}")
    
    latex_code.append(" & ".join(header_row) + " \\\\")
    
    # Calculate positions for cmidrules
    cmidrule_parts = []
    col_start = 2
    
    for _ in range(num_languages):
        col_end = col_start + cols_per_language - 1
        cmidrule_parts.append(f"\\cmidrule(lr){{{col_start}-{col_end}}}")
        col_start = col_end + 1
    
    latex_code.append(" ".join(cmidrule_parts))
    
    # Second header row with model names (with ACC/F1 subcolumns) and Avg
    second_header_row = [""]  # Empty cell for Detector column
    
    # For each language
    for _ in range(num_languages):
        # Add models with 2 columns each (ACC/F1)
        for model in generator_models:
            second_header_row.append(f"\\multicolumn{{2}}{{c}}{{{model}}}")
        
        # Add avg column at the end of each language section
        second_header_row.append(f"\\multicolumn{{2}}{{c}}{{\\textbf{{Avg}}}}")
    
    latex_code.append(" & ".join(second_header_row) + " \\\\")
    
    # Add cmidrule separators for each model's ACC/F1 columns
    cmidrule_parts = []
    col_idx = 2
    
    for _ in range(num_languages):
        for _ in range(num_models + 1):  # +1 for the Avg column
            cmidrule_parts.append(f"\\cmidrule(lr){{{col_idx}-{col_idx+1}}}")
            col_idx += 2
    
    latex_code.append(" ".join(cmidrule_parts))
    
    # Third header row with ACC/F1 labels
    third_header_row = [""]  # Empty cell for Detector column
    
    for _ in range(num_languages):
        # Add ACC/F1 labels for each model and average
        for _ in range(num_models + 1):
            third_header_row.extend(["ACC", "F1"])
    
    latex_code.append(" & ".join(third_header_row) + " \\\\")
    latex_code.append("\\midrule")
    
    # Data rows for each detector family
    for family, detectors in detector_families.items():
        # Process individual detectors
        for detector_key, (_, detector_name) in [(k, v) for k, v in detector_display_names.items() if v[0] == family]:
            row = [detector_name]  # Add detector name
            
            # Add metrics for each language
            for lang in languages:
                # Add individual model metrics
                for model in generator_models:
                    key = (lang, model, detector_key)
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
                
                # Add average column for this detector across models
                avg_key = (lang, detector_key)
                avg_values = detector_avg_results.get(avg_key, (None, None))
                
                # Add average accuracy (in bold)
                if avg_values and avg_values[0] is not None:
                    row.append(f"\\textbf{{{avg_values[0]:.2f}}}")
                else:
                    row.append("")
                
                # Add average F1 (in bold)
                if avg_values and avg_values[1] is not None:
                    row.append(f"\\textbf{{{avg_values[1]:.2f}}}")
                else:
                    row.append("")
            
            latex_code.append(" & ".join(row) + " \\\\")
        
        # Add dotted line
        latex_code.append(f"\\cdashline{{1-{total_cols}}} \\addlinespace[1pt]")
        
        # Add family average row
        avg_row = [f"Avg. {family}"]
        
        for lang in languages:
            # Add model-specific family averages
            for model in generator_models:
                key = (lang, model, family)
                values = family_model_avg_results.get(key, (None, None))
                
                # Add family average accuracy for this model
                if values and values[0] is not None:
                    avg_row.append(f"{values[0]:.2f}")
                else:
                    avg_row.append("")
                
                # Add family average F1 for this model
                if values and values[1] is not None:
                    avg_row.append(f"{values[1]:.2f}")
                else:
                    avg_row.append("")
            
            # Add overall family average (across all models)
            key = (lang, family)
            values = family_overall_avg_results.get(key, (None, None))
            
            # Add overall family average accuracy (in bold)
            if values and values[0] is not None:
                avg_row.append(f"\\textbf{{{values[0]:.2f}}}")
            else:
                avg_row.append("")
            
            # Add overall family average F1 (in bold)
            if values and values[1] is not None:
                avg_row.append(f"\\textbf{{{values[1]:.2f}}}")
            else:
                avg_row.append("")
        
        latex_code.append(" & ".join(avg_row) + " \\\\")
        
        # Add space after each family
        latex_code.append("\\addlinespace[3pt]")
    
    # Table footer
    latex_code.append("\\bottomrule")
    latex_code.append("\\end{tabular}")
    
    # Write to file
    latex_output = "\n".join(latex_code)
    with open(output_file, 'w') as f:
        f.write(latex_output)
    
    print(latex_output)

if __name__ == "__main__":
    main()