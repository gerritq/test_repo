import os
import json

all_results = {}
langs = ['en', 'pt', 'vi']

paras_prompts = ['minimal', 'cp', 'rag']
sums_prompts = ['minimal', 'instruct', 'few1', 'few2', 'few3']
tst_pronpts = ['minimal', 'instruct', 'few1', 'few2', 'few3', 'few4', 'few5']

metrics_paras_sums = ['bleu', 'rougeL', 'bertscore', 'qafacteval']
metrics_tst = ['bleu', 'rougeL', 'bertscore', 'style_transfer']

# Initialize nested dictionary structure
for lang in langs:
    all_results[lang] = {'first': {},
                        'extend': {},
                        'sums': {},
                        'tst': {}}
    for prompt in paras_prompts:
        all_results[lang]['first'][prompt] = {}
        all_results[lang]['extend'][prompt] = {}

    for prompt in sums_prompts:
        all_results[lang]['sums'][prompt] = {}

    for prompt in tst_pronpts:
        all_results[lang]['tst'][prompt] = {}

def first_and_extend():
    tasks = ['first', 'extend']
    metrics = ['brb', 'qa']
    
    for lang in langs:
        metrics_dir = f"/scratch/users/k21157437/paras/data/{lang}/metrics"
        for task in tasks:
            for prompt in paras_prompts:
                for metric in metrics:
                    # Handle the correct filename format
                    file_suffix = '_macro' if metric == 'qa' else ''
                    file_dir = os.path.join(metrics_dir, f"{lang}_paras_{task}_{metric}_{prompt}{file_suffix}.jsonl")

                    try:
                        with open(file_dir, 'r', encoding='utf-8') as f:
                            if metric == 'brb':
                                data = [json.loads(line) for line in f]
                                # Extract metrics from BRB format
                                for metric_name in data[0].keys():
                                    all_results[lang][task][prompt][metric_name] = data[0][metric_name]['overall']
                            elif metric == 'qa':
                                # Handle QA format correctly
                                data = json.loads(f.read())
                                all_results[lang][task][prompt]['qafacteval'] = data['overall']['f1']
                    except FileNotFoundError as e:
                        print(f"File not found: {file_dir}")
                    except Exception as e:
                        print(f"Error processing {file_dir}: {e}")


def sums():
    metrics = ['brb', 'qa']
    task = 'sums'

    for lang in langs:
        metrics_dir = f"/scratch/users/k21157437/sums/data/{lang}/eval/metrics"
        for prompt in sums_prompts:
            for metric in metrics:
                file_suffix = '_macro' if metric == 'qa' else ''
                file_dir = os.path.join(metrics_dir, f"{lang}_{task}_eval_{metric}_{prompt}{file_suffix}.jsonl")

                try:
                    with open(file_dir, 'r', encoding='utf-8') as f:
                        if metric == 'brb':
                            data = [json.loads(line) for line in f]
                            # Extract metrics from BRB format
                            for metric_name in data[0].keys():
                                all_results[lang][task][prompt][metric_name] = data[0][metric_name]['overall']
                        elif metric == 'qa':
                            # Handle QA format correctly
                            data = json.loads(f.read())
                            all_results[lang][task][prompt]['qafacteval'] = data['overall']['f1']
                except FileNotFoundError as e:
                    print(f"File not found: {file_dir}")
                except Exception as e:
                    print(f"Error processing {file_dir}: {e}")


def tst():
    metrics = ['brb', 'qa']
    prompts = ['minimal', 'instruct', 'few1', 'few2', 'few3', 'few4', 'few5']
    task = 'tst'


    langs_tst = langs
    for lang in langs:
        metrics_dir = f"/scratch/users/k21157437/neutral_new/data/{lang}/eval/metrics"
        for prompt in prompts:
            for metric in metrics:
                file_dir = os.path.join(metrics_dir, f"{lang}_{prompt}_metrics_macro.jsonl")

                try:
                    with open(file_dir, 'r', encoding='utf-8') as f:
                        data = [json.loads(line) for line in f]
                        for metric, score in data[0].items():
                            all_results[lang][task][prompt][metric] = score
                except FileNotFoundError as e:
                    print(f"File not found: {file_dir}")
                except Exception as e:
                    print(f"Error processing {file_dir}: {e}")

    # Inlcude En paras
    # lang = 'en'
    # metrics_dir = f"/scratch/users/k21157437/neutral_new/data/en/eval_paras/metrics"
    # for prompt in prompts:
    #     for metric in metrics:
    #         file_dir = os.path.join(metrics_dir, f"{lang}_{prompt}_metrics_macro.jsonl")

    #         try:
    #             with open(file_dir, 'r', encoding='utf-8') as f:
    #                 data = [json.loads(line) for line in f]
    #                 for metric, score in data[0].items():
    #                     all_results[lang][task][prompt][metric] = score
    #         except FileNotFoundError as e:
    #             print(f"File not found: {file_dir}")
    #         except Exception as e:
    #             print(f"Error processing {file_dir}: {e}")

def select_best_take_differences():
    
    latex_results = {}
    for task in ['first', 'extend', 'sums', 'tst']:
        latex_results[task] = {'en': {},
                            'vi': {},
                            'pt': {}}

        if task in ['first', 'extend']:
            for lang in langs:
                for metric in metrics_paras_sums:
                    rag_first = all_results[lang][task]['rag'][metric]
                    minimal_first = all_results[lang][task]['minimal'][metric]

                    latex_results[task][lang][metric] = (rag_first, rag_first-minimal_first)

        if task == 'sums':
            for lang in langs:
                for metric in metrics_paras_sums:
                    few4_sums = all_results[lang][task]['few1'][metric]
                    minimal_sums = all_results[lang][task]['minimal'][metric]

                    latex_results[task][lang][metric] = (few4_sums, few4_sums-minimal_sums)

        if task == 'tst':
            for lang in langs:
                for metric in metrics_tst:
                    few5_tst = all_results[lang][task]['few5'][metric]
                    minimal_tst = all_results[lang][task]['minimal'][metric]

                    latex_results[task][lang][metric] = (few5_tst, few5_tst-minimal_tst)



    return latex_results

def create_latex_table(latex_results):
    # Start the table
    latex_table = "\\begin{tabular}{lccccc}\n\\toprule\n"
    
    # Metrics header row - standard metrics first
    latex_table += "\\textbf{Language} & \\textbf{BLEU} & \\textbf{RougeL} & \\textbf{BERTScore} & \\textbf{QAFactEval} & \\textbf{Style Transfer} \\\\\n\\midrule\n"
    
    # First two tasks use the same metrics
    for task_idx, task, prompt in [('first', 'Introductory Paragraph', 'RAG'), ('extend', 'Paragraph Continuation', 'RAG'), ('sums', 'Summarisation', 'One-shot')]:
        # Add task header
        latex_table += f"\\multicolumn{{6}}{{l}}{{\\textit{{{task} - {prompt}}}}} \\\\\n\\midrule\n"
        
        # Add language rows
        for lang_code, lang_name in [('en', 'English'), ('pt', 'Portuguese'), ('vi', 'Vietnamese')]:
            metric_values = []
            
            # First four standard metrics
            for metric in ['bleu', 'rougeL', 'bertscore', 'qafacteval']:
                score, diff = latex_results[task_idx][lang_code][metric]
                if diff > 0:
                    metric_values.append(f"{score:.2f} (\\textcolor{{ForestGreen}}{{+{diff:.2f}}})")
                else:
                    metric_values.append(f"{score:.2f} (\\textcolor{{orange}}{{{diff:.2f}}})")
            
            # Empty column for Style Transfer in non-TST tasks
            metric_values.append("-")
            
            latex_table += f"{lang_name} & " + " & ".join(metric_values) + " \\\\\n"
        
        # Add a blank line between task sections
        latex_table += "\\midrule\n"
    
    # Text Style Transfer section
    latex_table += f"\\multicolumn{{6}}{{l}}{{\\textit{{TST - Five-shot}}}} \\\\\n\\midrule\n"
    
    # Add language rows for TST
    for lang_code, lang_name in [('en', 'English'), ('pt', 'Portuguese'), ('vi', 'Vietnamese')]:
        metric_values = []
        
        # All metrics including Style Transfer
        for metric in ['bleu', 'rougeL', 'bertscore', 'gap', 'style_transfer']:
            if metric == 'gap':
                metric_values.append('-')
            else:
                score, diff = latex_results['tst'][lang_code][metric]
                if diff > 0:
                    metric_values.append(f"{score:.2f} (\\textcolor{{ForestGreen}}{{+{diff:.2f}}})")
                else:
                    metric_values.append(f"{score:.2f} (\\textcolor{{orange}}{{{diff:.2f}}})")
        
        latex_table += f"{lang_name} & " + " & ".join(metric_values) + " \\\\\n"
    
    # Close the table
    latex_table += "\\bottomrule\n\\end{tabular}"
    
    print('\n\n')
    print(latex_table)
    print('\n\n')

def main():
    first_and_extend()
    sums()
    tst()
    print(all_results)

    print('\n\n')
    latex_results = select_best_take_differences()
    create_latex_table(latex_results)



if __name__ == "__main__":
    main()