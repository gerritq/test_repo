import os
import json
from collections import defaultdict

def main():
    base_dir = "/scratch/users/k21157437/mix/detect/lang"
    languages = ['en', 'pt', 'vi']
    output_file = 'aresults_lang.tex'
    
    language_display_names = {
        'en': 'English',
        'pt': 'Portuguese',
        'vi': 'Vietnamese'
    }
    
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
    
    detector_families = defaultdict(list)
    for detector_key, (family, _) in detector_display_names.items():
        detector_families[family].append(detector_key)
    
    results_files = {}
    
    for lang in languages:
        
        for detector_key, (family, detector_name) in detector_display_names.items():
            if detector_key.startswith('zero_'):
                file_path = os.path.join(base_dir, f"{lang}_{detector_key}.jsonl")
            else:  
                model_name = detector_key.split('_')[1]
                file_path = os.path.join(base_dir, f"{lang}_{detector_key}.jsonl")
            
            if os.path.exists(file_path):
                key = (lang, detector_key)
                results_files[key] = file_path
            else:
                print(f"File not found: {file_path}")
    
    # Extract metrics for each combination
    results_data = {}
    
    for key, file_path in results_files.items():
        lang, detector = key


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
    
    # Calculate family averages
    family_avg_results = {}
    
    for lang in languages:
        # Calculate family averages
        for family, detector_list in detector_families.items():
            # Collect scores for all detectors in this family
            family_acc_scores = []
            family_f1_scores = []
            
            for detector_key in detector_list:
                key = (lang, detector_key)
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
                family_avg_results[(lang, family)] = (family_avg_acc, family_avg_f1)
    
    # Generate LaTeX table
    latex_code = []
    
    # Use a simplified tabular environment with 7 columns:
    # 1 for detector name + (2 columns for each of the 3 languages)
    latex_code.append("\\begin{tabular}{lcccccc}")
    
    # Header rows
    latex_code.append("\\toprule")
    
    # First header row with language groups
    header_row = ["\\textbf{Detector}"]
    
    for lang in languages:
        header_row.append(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{language_display_names[lang]}}}}}")
    
    latex_code.append(" & ".join(header_row) + " \\\\")
    
    # Add cmidrule separators for each language's ACC/F1 columns
    cmidrule_parts = []
    col_idx = 2
    for lang_idx in range(len(languages)):
        cmidrule_parts.append(f"\\cmidrule(lr){{{col_idx}-{col_idx+1}}}")
        col_idx += 2
    
    latex_code.append(" ".join(cmidrule_parts))
    
    # Second header row with ACC/F1 labels
    second_header_row = [""]  # Empty cell for Detector column
    
    for _ in range(len(languages)):
        second_header_row.extend(["ACC", "F1"])
    
    latex_code.append(" & ".join(second_header_row) + " \\\\")
    latex_code.append("\\midrule")
    
    # Data rows for each detector family
    for family, detectors in detector_families.items():
        # Process individual detectors
        for detector_key, (_, detector_name) in [(k, v) for k, v in detector_display_names.items() if v[0] == family]:
            row = [detector_name]  # Add detector name
            
            # Add metrics for each language
            for lang in languages:
                key = (lang, detector_key)
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
            
            latex_code.append(" & ".join(row) + " \\\\")
        
        # Add dotted line
        latex_code.append("\\cdashline{1-7} \\addlinespace[1pt]")
        
        # Add family average row
        avg_row = [f"Avg. {family}"]
        
        for lang in languages:
            key = (lang, family)
            values = family_avg_results.get(key, (None, None))
            
            # Add family average accuracy (in bold)
            if values and values[0] is not None:
                avg_row.append(f"\\textbf{{{values[0]:.2f}}}")
            else:
                avg_row.append("")
            
            # Add family average F1 (in bold)
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