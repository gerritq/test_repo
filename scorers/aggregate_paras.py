import os
import json
import pandas as pd
import argparse

class MetricsAggregator:
    def __init__(self, base_dir, prompt_techs, langs=["en", "pt", "vi"], subsets=["first", "extend"]):
        self.base_dir = base_dir
        self.langs = langs
        self.subsets = subsets
        self.prompt_techs = prompt_techs
        
        # Language mapping for nicer labels
        self.language_mapping = {
            "en": "English",
            "pt": "Portuguese",
            "vi": "Vietnamese"
        }
        
        self.results = []
    
    def load_metrics(self):
        for lang in self.langs:
            for subset in self.subsets:
                for prompt_tech in self.prompt_techs:
                    # Define file paths
                    brb_file = os.path.join(self.base_dir, f"{lang}/metrics/{lang}_paras_{subset}_brb_{prompt_tech}.jsonl")
                    qa_file = os.path.join(self.base_dir, f"{lang}/metrics/{lang}_paras_{subset}_qa_{prompt_tech}_macro.jsonl")
                    
                    # Create entry with level, language, and prompt_tech
                    entry = {
                        "language": self.language_mapping.get(lang, lang),
                        "prompt_tech": prompt_tech,
                        "level": subset
                    }
                    
                    # Load BLEU, ROUGE, and BERTScore
                    if os.path.exists(brb_file):
                        with open(brb_file, 'r', encoding='utf-8') as f:
                            data = [json.loads(line) for line in f]
                            if data:
                                metrics_data = data[0]
                                entry["bleu"] = metrics_data.get("bleu", {}).get("overall", None)
                                entry["rouge1"] = metrics_data.get("rouge1", {}).get("overall", None)
                                entry["rouge2"] = metrics_data.get("rouge2", {}).get("overall", None)
                                entry["rougeL"] = metrics_data.get("rougeL", {}).get("overall", None)
                                entry["bertscore"] = metrics_data.get("bertscore", {}).get("overall", None)
                    else:
                        print(f"Warning: Missing file {brb_file}")
                    
                    # Load QAFactEval
                    if os.path.exists(qa_file):
                        with open(qa_file, 'r', encoding='utf-8') as f:
                            data = [json.loads(line) for line in f]
                            if data:
                                qa_data = data[0]
                                entry["f1"] = qa_data.get("overall", {}).get("f1", None)
                    else:
                        print(f"Warning: Missing file {qa_file}")
                    
                    self.results.append(entry)
    
    def generate_table(self):
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        if len(df) == 0:
            print("Warning: No data loaded!")
            return ""
        
        # Map levels to display names
        level_display = {'first': 'Introductory', 'extend': 'Continuation'}
        df['level'] = df['level'].map(level_display)
        
        # Map prompt techniques to display names
        prompt_tech_mapping = {
            "minimal": "Minimal",
            "cp": "Content Prompts",
            "rag": "RAG"
        }
        df['prompt_tech'] = df['prompt_tech'].map(lambda x: prompt_tech_mapping.get(x, x))
        
        # Define sort orders
        level_order = ['Introductory', 'Continuation']
        language_order = ['English', 'Portuguese', 'Vietnamese']
        prompt_tech_order = ['Minimal', 'Content Prompts', 'RAG']
        
        # Create sort keys
        df['level_sort'] = pd.Categorical(df['level'], categories=level_order, ordered=True)
        df['language_sort'] = pd.Categorical(df['language'], categories=language_order, ordered=True)
        df['prompt_tech_sort'] = pd.Categorical(df['prompt_tech'], categories=prompt_tech_order, ordered=True)
        
        # Sort
        df = df.sort_values(['level_sort', 'language_sort', 'prompt_tech_sort'])
        df = df.drop(['level_sort', 'language_sort', 'prompt_tech_sort'], axis=1)
        
        # Rename columns
        column_mapping = {
            'level': 'Level',
            'language': 'Language',
            'prompt_tech': 'Technique',
            'bleu': 'BLEU',
            'rouge1': 'ROUGE-1',
            'rouge2': 'ROUGE-2',
            'rougeL': 'ROUGE-L',
            'bertscore': 'BERTScore',
            'f1': 'QAFactEval'
        }
        df = df.rename(columns=column_mapping)
        
        # Generate LaTeX
        latex = self._generate_latex(df)
        
        return latex
    
    def _generate_latex(self, df):
        # Count rows per level
        level_counts = {}
        for level in ['Introductory', 'Continuation']:
            level_counts[level] = len(df[df['Level'] == level])
        
        # Count rows per language within each level
        lang_counts = {}
        for level in ['Introductory', 'Continuation']:
            lang_counts[level] = {}
            level_df = df[df['Level'] == level]
            for lang in ['English', 'Portuguese', 'Vietnamese']:
                lang_counts[level][lang] = len(level_df[level_df['Language'] == lang])
        
        # Create the LaTeX table manually
        latex_lines = []
        
        # Just the tabular environment
        latex_lines.append('\\begin{tabular}{lllrrrrrr}')
        latex_lines.append('\\toprule')
        
        # Column headers
        headers = ['\\textbf{Level}', '\\textbf{Language}', '\\textbf{Technique}', 
                  '\\textbf{BLEU}', '\\textbf{ROUGE-1}', '\\textbf{ROUGE-2}', 
                  '\\textbf{ROUGE-L}', '\\textbf{BERTScore}', '\\textbf{QAFactEval}']
        latex_lines.append(' & '.join(headers) + ' \\\\')
        
        latex_lines.append('\\midrule')
        
        # Data rows - track position for adding midrules
        current_level = None
        current_language = None
        lang_count_in_level = 0
        
        languages_in_order = ['English', 'Portuguese', 'Vietnamese']
        
        for level in ['Introductory', 'Continuation']:
            level_rows = []
            level_df = df[df['Level'] == level]
            
            # First language
            for lang_index, language in enumerate(languages_in_order):
                lang_df = level_df[level_df['Language'] == language]
                
                for tech_index, (idx, row) in enumerate(lang_df.iterrows()):
                    line_parts = []
                    
                    # Level column (only first row of the entire level gets the multirow)
                    if lang_index == 0 and tech_index == 0:
                        level_row_count = level_counts[level]
                        line_parts.append(f"\\multirow{{{level_row_count}}}{{*}}{{{level}}}")
                    else:
                        line_parts.append("")
                    
                    # Language column (only first row of each language group gets the multirow)
                    if tech_index == 0:
                        lang_row_count = lang_counts[level][language]
                        line_parts.append(f"\\multirow{{{lang_row_count}}}{{*}}{{{language}}}")
                    else:
                        line_parts.append("")
                    
                    # Add technique and metric values
                    line_parts.append(f"{row['Technique']}")
                    line_parts.append(f"{row['BLEU']:.2f}")
                    line_parts.append(f"{row['ROUGE-1']:.2f}")
                    line_parts.append(f"{row['ROUGE-2']:.2f}")
                    line_parts.append(f"{row['ROUGE-L']:.2f}")
                    line_parts.append(f"{row['BERTScore']:.2f}")
                    line_parts.append(f"{row['QAFactEval']:.2f}")
                    
                    level_rows.append(' & '.join(line_parts) + ' \\\\')
                
                # Add cmidrule after each language except the last one in a level
                if lang_index < len(languages_in_order) - 1:
                    level_rows.append('\\cmidrule{2-9}')
            
            # Add all rows for this level
            latex_lines.extend(level_rows)
            
            # Add midrule between levels (except after the last level)
            if level != 'Continuation':
                latex_lines.append('\\midrule')
        
        # Footer
        latex_lines.append('\\bottomrule')
        latex_lines.append('\\end{tabular}')
        
        latex_table = '\n'.join(latex_lines)
        
        # Save to file
        with open('metrics_table.tex', 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        return latex_table
        
        latex_table = '\n'.join(latex_lines)
        
        # Save to file
        with open('metrics_table.tex', 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        return latex_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory for metrics files')
    parser.add_argument('--prompt_techs', nargs='+', required=True, help='Prompting techniques')
    parser.add_argument('--langs', nargs='+', default=["en", "pt", "vi"], help='Languages to process')
    parser.add_argument('--subsets', nargs='+', default=["first", "extend"], help='Data subsets')
    
    args = parser.parse_args()
    
    aggregator = MetricsAggregator(
        base_dir=args.base_dir,
        prompt_techs=args.prompt_techs,
        langs=args.langs,
        subsets=args.subsets
    )
    aggregator.load_metrics()
    table = aggregator.generate_table()
    print(table)