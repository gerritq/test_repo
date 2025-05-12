import os
import json
import pandas as pd
from collections import defaultdict
import argparse

class MetricsAggregator:
    def __init__(self, 
                 ds,
                 base_dir,
                 prompt_techs,
                 langs=["en", "pt", "vi"],
                 subsets=["main"],
                 selected_metrics=None):

        self.ds = ds
        self.base_dir = base_dir
        self.langs = langs
        self.subsets = subsets
        self.prompt_techs = prompt_techs
        
        # Define all available metrics
        self.all_metrics = {
            "brb": ["bleu", "rouge1", "rouge2", "rougeL", "bertscore"],
            "qa": ["f1"], #["precision", "recall", "em", "f1", "lerc_quip", "is_answered"]
            "sc": ["sc f1"],
            "geval": ["GEval"],
            "mcheck": ["MCheck"]
        }
        
        # Process selected metrics
        self.selected_metrics = selected_metrics
        self.process_selected_metrics()
        
        # Language mapping for nicer labels
        self.language_mapping = {
            "en": "English",
            "pt": "Portuguese",
            "vi": "Vietnamese"
        }
        
        self.results = []
    
    def process_selected_metrics(self):
        """Process the selected metrics to determine which ones to include"""
        # If no specific metrics are selected, include all
        if not self.selected_metrics:
            self.brb = self.all_metrics["brb"]
            self.qa_metrics = self.all_metrics["qa"]
            self.sc_metrics = self.all_metrics["sc"]
            self.geval_metrics = self.all_metrics["geval"]
            self.mcheck_metrics = self.all_metrics["mcheck"]
            return
            
        # Initialize empty lists for each metric category
        self.brb = []
        self.qa_metrics = []
        self.sc_metrics = []
        self.geval_metrics = []
        self.mcheck_metrics = []
        
        # Process each selected metric group
        for metric in self.selected_metrics:
            if metric == "brb" or metric == "all":
                self.brb = self.all_metrics["brb"]
            elif metric == "qa" or metric == "all":
                self.qa_metrics = self.all_metrics["qa"]
            elif metric == "sc" or metric == "all":
                self.sc_metrics = self.all_metrics["sc"]
            elif metric == "geval" or metric == "all":
                self.geval_metrics = self.all_metrics["geval"]
            elif metric == "mcheck" or metric == "all":
                self.mcheck_metrics = self.all_metrics["mcheck"]
            # Process individual metrics
            elif metric in self.all_metrics["brb"]:
                self.brb.append(metric)
            elif metric in self.all_metrics["qa"]:
                self.qa_metrics.append(metric)
            elif metric == "sc f1" or metric == "scf1":
                self.sc_metrics = ["sc f1"]
            elif metric == "GEval":
                self.geval_metrics = ["GEval"]
            elif metric == "MCheck":
                self.mcheck_metrics = ["MCheck"]
    
    def load_metrics(self):
        for lang in self.langs:
            for subset in self.subsets:
                for prompt_tech in self.prompt_techs:

                    if self.ds == 'paras':
                        brb_file = os.path.join(self.base_dir, f"{lang}/metrics/{lang}_paras_{subset}_brb_{prompt_tech}.jsonl")
                        qa_file = os.path.join(self.base_dir, f"{lang}/metrics/{lang}_paras_{subset}_qa_{prompt_tech}_macro.jsonl")
                        sc_file = os.path.join(self.base_dir, f"{lang}/metrics/{lang}_paras_{subset}_sc_macro.jsonl")
                        geval_file = os.path.join(self.base_dir, f"{lang}/metrics/{lang}_{subset}_geval_{prompt_tech}_macro.jsonl")
                        mcheck_file = ''
                    if self.ds == 'sums':
                        brb_file = os.path.join(self.base_dir, f"{lang}/eval/metrics/{lang}_sums_eval_brb_{prompt_tech}.jsonl")
                        qa_file = os.path.join(self.base_dir, f"{lang}/eval/metrics/{lang}_sums_eval_qa_{prompt_tech}_macro.jsonl")
                        sc_file = os.path.join(self.base_dir, f"{lang}/eval/metrics/{lang}_sums_eval_sc_{prompt_tech}_macro.jsonl")
                        mcheck_file = os.path.join(self.base_dir, f"{lang}/eval/metrics/{lang}_sums_eval_mcheck_{prompt_tech}.jsonl")                        
                        # not used here ..
                        geval_file = os.path.join(self.base_dir, f"{lang}/eval/{subset}/metrics/{lang}_paras_eval_{subset}_geval_{prompt_tech}_macro.jsonl")

                    entry = {
                        "language": self.language_mapping.get(lang, lang),  # Use nicer language labels
                        "prompt_tech": prompt_tech
                    }
                    
                    # Only include subset if ds is not 'sums'
                    if self.ds != 'sums':
                        entry["subset"] = subset
                    
                    # Load basic metrics if they're selected
                    if self.brb and os.path.exists(brb_file):
                        with open(brb_file, 'r', encoding='utf-8') as f:
                            data = [json.loads(line) for line in f]
                            if data:
                                metrics_data = data[0]  # Assuming single-line JSONL file
                                for metric in self.brb:
                                    entry[metric] = metrics_data.get(metric, {}).get("overall", None)
                    elif self.brb:
                        print(f"Warning: Missing file {brb_file}")
                    
                    # Load QA-related metrics if they're selected
                    if self.qa_metrics and os.path.exists(qa_file):
                        with open(qa_file, 'r', encoding='utf-8') as f:
                            data = [json.loads(line) for line in f]
                            if data:
                                qa_data = data[0]  # Assuming single-line JSONL file
                                for metric in self.qa_metrics:
                                    entry[metric] = qa_data.get("overall", {}).get(metric, None)
                    elif self.qa_metrics:
                        print(f"Warning: Missing file {qa_file}")

                    # Load overall score from `sc_macro.jsonl` if selected
                    if self.sc_metrics and os.path.exists(sc_file):
                        with open(sc_file, 'r', encoding='utf-8') as f:
                            data = [json.loads(line) for line in f]
                            if data:
                                sc_data = data[0]  # Assuming single-line JSONL file
                                entry["sc f1"] = sc_data.get("overall", None)
                    elif self.sc_metrics:
                        print(f"Warning: Missing file {sc_file}")

                    # Load GEVAL metrics if selected
                    if self.geval_metrics and os.path.exists(geval_file):
                        with open(geval_file, 'r', encoding='utf-8') as f:
                            data = [json.loads(line) for line in f]
                            if data:        
                                entry["GEval"] = data[0]["GEval Mean Score"]
                    elif self.geval_metrics:
                        print(f"Warning: Missing file {geval_file}")

                    # Load MCHECK metrics if selected
                    if self.mcheck_metrics and os.path.exists(mcheck_file):
                        with open(mcheck_file, 'r', encoding='utf-8') as f:
                            data = [json.loads(line) for line in f]
                            if data:        
                                entry["MCheck"] = data[0]["mean_support"]
                    elif self.mcheck_metrics:
                        print(f"Warning: Missing file {mcheck_file}")
                    
                    self.results.append(entry)

    def generate_dataframe(self):
        """Convert the collected results into a Pandas DataFrame."""
        df = pd.DataFrame(self.results)
        
        # Change subset naming for paras dataset
        if self.ds == 'paras' and 'subset' in df.columns:
            df = df.rename(columns={'subset': 'level'})
        
        # Rename the levels for display
        if 'level' in df.columns:
            level_display = {'first': 'Introductory', 'extend': 'Continuation'}
            df['level'] = df['level'].map(lambda x: level_display.get(x, x))
        
            # Create sort order with new names
            level_order = {'Introductory': 0, 'Continuation': 1}
            df['level_sort'] = df['level'].map(level_order)
        
        # Define mapping for prompt tech values
        prompt_tech_mapping = {
            "minimal": "Minimal",
            "instruct": "Instruct",
            "few1": "Few-Shot 1",
            "few2": "Few-Shot 2",
            "few3": "Few-Shot 3",
            "few4": "Few-Shot 4",
            "few5": "Few-Shot 5",
            "rag": "RAG",
            "cp": "Content Prompts",
        }
        
        # Create custom sort order for prompt techniques
        prompt_tech_order = {'Minimal': 0, 'Content Prompts': 1, 'RAG': 2}
        
        # Apply sorting with proper order
        if 'level' in df.columns and 'prompt_tech' in df.columns:
            # Convert prompt_tech first
            df["prompt_tech"] = df["prompt_tech"].apply(
                lambda x: prompt_tech_mapping.get(x, x.capitalize() if isinstance(x, str) else x)
            )
            
            # Add sort column for prompt_tech
            df['prompt_tech_sort'] = df['prompt_tech'].map(lambda x: prompt_tech_order.get(x, 999))
            
            # Sort with all three criteria
            df = df.sort_values(['level_sort', 'language', 'prompt_tech_sort'])
            
            # Remove sort columns
            df = df.drop('prompt_tech_sort', axis=1)
            if 'level_sort' in df.columns:
                df = df.drop('level_sort', axis=1)
        
        # Move level to first column
        if 'level' in df.columns:
            cols = df.columns.tolist()
            cols.remove('level')
            cols.insert(0, 'level')
            df = df[cols]
        
        # Format level column
        if 'level' in df.columns:
            # Group by level
            for lvl in ['Introductory', 'Continuation']:
                indices = df[df['level'] == lvl].index
                if len(indices) > 0:
                    # Keep level value only for the first row in each level group
                    df.loc[indices[1:], 'level'] = ""
        
        # Process the language column to show value only once per group
        if "language" in df.columns:
            # Clear non-first occurrences of languages within prompt_tech groups
            for lvl in df['level'].unique():
                for lang in df["language"].unique():
                    level_lang_indices = df[(df['level'] == lvl) & (df["language"] == lang)].index
                    if len(level_lang_indices) > 0:
                        # Make sure first row for each language in each level has language value
                        df.loc[level_lang_indices[0], "language"] = lang
                        # Clear others in same prompt_tech group
                        df.loc[level_lang_indices[1:], "language"] = ""
        
        # Define column name formatting
        column_name_mapping = {
            "language": "Language",
            "prompt_tech": "Technique",
            "subset": "Subset",
            "level": "Level",
            "bleu": "BLEU",
            "rouge1": "ROUGE-1",
            "rouge2": "ROUGE-2",
            "rougeL": "ROUGE-L",
            "bertscore": "BERTScore",
            "precision": "Precision",
            "recall": "Recall",
            "em": "EM",
            "f1": "QAFactEval",
            "lerc_quip": "LERC-QuiP",
            "is_answered": "Is Answered",
            "sc f1": "SC F1",
            "GEval": "GEval",
            "MCheck": "MCheck"
        }
        
        # Rename columns with proper formatting
        renamed_columns = {}
        for col in df.columns:
            if col in column_name_mapping:
                renamed_columns[col] = column_name_mapping[col]
            else:
                renamed_columns[col] = ' '.join(word.capitalize() for word in col.split('_'))
        
        df = df.rename(columns=renamed_columns)
        
        # Make column headers bold with LaTeX formatting
        df.columns = [f"\\textbf{{{col}}}" for col in df.columns]
        
        return df

    # def generate_dataframe(self):
    #     """Convert the collected results into a Pandas DataFrame."""
    #     df = pd.DataFrame(self.results)
        
    #     # Define a mapping for prompt tech values
    #     prompt_tech_mapping = {
    #         "minimal": "Minimal",
    #         "instruct": "Instruct",
    #         "few1": "Few-Shot 1",
    #         "few2": "Few-Shot 2",
    #         "few3": "Few-Shot 3",
    #         "few4": "Few-Shot 4",
    #         "few5": "Few-Shot 5",
    #         "rag": "RAG",
    #         "cp": "Content Prompts",
    #     }
        
    #     if "prompt_tech" in df.columns:
    #             df["prompt_tech"] = df["prompt_tech"].apply(
    #                 lambda x: prompt_tech_mapping.get(x, x.capitalize() if isinstance(x, str) else x)
    #             )
        
    #     # Process the language column to show value only once per group
    #     if "language" in df.columns:
    #         # Group by language and set the language value to empty string for all but the first occurrence
    #         for lang in df["language"].unique():
    #             lang_indices = df[df["language"] == lang].index
    #             if len(lang_indices) > 0:
    #                 # Keep language value only for the first row in each language group
    #                 df.loc[lang_indices[1:], "language"] = ""
        
    #     # Define a mapping for column name formatting
    #     column_name_mapping = {
    #         "language": "Language",
    #         "prompt_tech": "Technique",
    #         "subset": "Subset",
    #         "bleu": "BLEU",
    #         "rouge1": "ROUGE-1",
    #         "rouge2": "ROUGE-2",
    #         "rougeL": "ROUGE-L",
    #         "bertscore": "BERTScore",
    #         "precision": "Precision",
    #         "recall": "Recall",
    #         "em": "EM",
    #         "f1": "QAFactEval", # F1
    #         "lerc_quip": "LERC-QuiP",
    #         "is_answered": "Is Answered",
    #         "sc f1": "SC F1",
    #         "GEval": "GEval",
    #         "MCheck": "MCheck"
    #     }
        
    #     # Rename columns with proper formatting
    #     renamed_columns = {}
    #     for col in df.columns:
    #         if col in column_name_mapping:
    #             renamed_columns[col] = column_name_mapping[col]
    #         else:
    #             # For any columns not in our mapping, capitalize each word
    #             renamed_columns[col] = ' '.join(word.capitalize() for word in col.split('_'))
        
    #     df = df.rename(columns=renamed_columns)
        
    #     # Make column headers bold with LaTeX formatting
    #     df.columns = [f"\\textbf{{{col}}}" for col in df.columns]
        
    #     return df
    
    def generate_latex(self):
        """Generate a LaTeX table from the DataFrame with midrules after each language group."""
        df = self.generate_dataframe()
        latex_table = df.to_latex(index=False, float_format="%.2f", escape=False)
        
        latex_list = latex_table.splitlines()
        content_start = next(i for i, line in enumerate(latex_list) if "midrule" in line) + 1
        
        rows_per_language = len(self.prompt_techs)
        
        content_rows = len(latex_list) - content_start - 1  # -1 for bottomrule
        
        midrule_positions = []
        for i in range(content_start + rows_per_language, content_start + content_rows, rows_per_language):
            if i < len(latex_list) - 1:
                midrule_positions.append(i)

        for pos in sorted(midrule_positions, reverse=True):
            latex_list.insert(pos, r"\midrule")
        
        latex_final = "\n".join(latex_list)
        
        print(latex_final)
        
        with open('metrics_table.tex', "w", encoding="utf-8") as f:
            f.write(latex_final)
        return latex_final

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory for metrics files')
    parser.add_argument('--ds', type=str, required=True, help='Dataset type (paras or sums)')
    parser.add_argument('--prompt_techs', nargs='+', required=True, help='Prompting techniques')
    parser.add_argument('--langs', nargs='+', default=["en", "pt", "vi"], help='Languages to process')
    parser.add_argument('--subsets', nargs='+', default=["main"], help='Data subsets')
    parser.add_argument('--metrics', nargs='+', default=None, 
                        help='Metrics to include in the table. Options: all, brb, qa, sc, geval, mcheck, or individual metric names. Default includes all metrics.')
    
    args = parser.parse_args()
    
    aggregator = MetricsAggregator(
        base_dir=args.base_dir,
        ds=args.ds,
        prompt_techs=args.prompt_techs,
        langs=args.langs,
        subsets=args.subsets,
        selected_metrics=args.metrics
    )
    aggregator.load_metrics()
    aggregator.generate_latex()