import json
import argparse
import time
import numpy as np
from utils import load_jsonl, merge_jsonl, save_jsonl
from collections import defaultdict
from summac.model_summac import SummaCConv

class SummaCEvaluator:
    def __init__(self, 
                 lang,
                 in_file,
                 out_file,
                 prompt_techs,
                 device="cuda"):
        self.lang = lang
        self.in_file = in_file
        self.out_file = out_file
        self.prompt_techs = prompt_techs
        self.device = device
        
        self.model_conv = SummaCConv(
            models=["vitc"],
            bins="percentile",
            granularity="sentence",
            nli_labels="e",
            device=self.device,
            start_file="default",
            agg="mean"
            )



    def evaluate(self):
        """Runs the SummaC evaluation and saves results."""
        print("Start ... ", flush=True)
        start_time = time.time()

        trans = "_trans" if self.lang != "en" else ""
        
        for prompt_tech in self.prompt_techs:
            print("\n----------------", flush=True)
            print("----------------", flush=True)
            print(f"Running eval for {self.lang} {prompt_tech}", flush=True)

            in_file_mod = self.in_file.replace('.jsonl', f'_{prompt_tech}.jsonl')
            out_file_macro = self.out_file.replace('.jsonl', f'_sc_{prompt_tech}_macro.jsonl')
            out_file_micro = self.out_file.replace('.jsonl', f'_sc_{prompt_tech}_micro.jsonl')
            
            data = load_jsonl(in_file_mod)
            print(f"N: {len(data)}", flush=True)

            mgts, refs = [], []
            for item in data:
                mgts.append(item[f'mgt_{prompt_tech}{trans}'])
                refs.append(item[f'src_inf{trans}'])
            
            # Compute SummaC scores
            score_conv = self.model_conv.score(refs, mgts) # doc, summart
            
            out = []
            scores_by_tertile = defaultdict(list)

            for i in range(len(data)):
                item_new = data[i].copy()
                item_new.update({"prompting": prompt_tech,
                                 "summacconv_score": score_conv["scores"][i]})
                out.append(item_new)
                
                scores_by_tertile[data[i]["word_tertile"]].append(score_conv["scores"][i])
                scores_by_tertile["overall"].append(score_conv["scores"][i])
            
            # Compute mean scores
            mean_scores_by_tertile = {k: np.mean(v) for k, v in scores_by_tertile.items()}

            # Save results
            save_jsonl(out, out_file_micro)
            save_jsonl([mean_scores_by_tertile], out_file_macro)

        print(f"Evaluation completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, required=True, help='Language code (e.g., en, vi, pt)')
    parser.add_argument('--in_file', type=str, required=True, help='Base input file path (without technique suffix)')
    parser.add_argument('--out_file', type=str, required=True, help='Base output file path (without technique suffix)')
    parser.add_argument('--prompt_techs', nargs='+', required=True, help='Prompting techniques to evaluate')    
    args = parser.parse_args()

    evaluator = SummaCEvaluator(
                lang=args.lang,
                in_file=args.in_file,
                out_file=args.out_file,
                prompt_techs=args.prompt_techs
            )
    evaluator.evaluate()