import argparse
from minicheck.minicheck import MiniCheck
import os
from nltk import sent_tokenize
import numpy as np
import torch
import time
from utils import load_jsonl, save_jsonl
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cache_dir = "/scratch_tmp/users/k21157437/.checkcache"

# doc = "A group of students gather in the school library to study for their upcoming final exams."
# claim_1 = "The students are preparing for an examination."
# claim_2 = "The students are on vacation."

# model_name can be one of the followings:
# ['roberta-large', 'deberta-v3-large', 'flan-t5-large', 'Bespoke-MiniCheck-7B']

#  MiniCheck-Flan-T5-Large (770M) is the best fack-checking model 
# with size < 1B and reaches GPT-4 performance.
# scorer = MiniCheck(model_name='flan-t5-large', cache_dir=chache_dir)
# pred_label, raw_prob, _, _ = scorer.score(docs=[doc, doc], claims=[claim_1, claim_2])

# print(pred_label) # [1, 0]
# print(raw_prob)   # [0.9805923700332642, 0.007121330592781305]



class MCheckEvaluator:
    def __init__(self, 
                    lang,
                    in_file,
                    out_file,
                    prompt_techs):
        self.lang = lang
        self.in_file = in_file
        self.out_file = out_file
        self.prompt_techs = prompt_techs
        self.scorer = MiniCheck(model_name='flan-t5-large', cache_dir=cache_dir)


    def run_evaluation(self):
        print('Start ... ', flush=True)
        start = time.time()

        trans = '_trans' if self.lang != 'en' else ''

        for prompt_tech in self.prompt_techs:
            print('\n----------------', flush=True)
            print('----------------', flush=True)
            print(f"Running MiniCheck for {self.lang} {prompt_tech} in {self.prompt_techs}", flush=True)

            in_file_mod = self.in_file.replace('.jsonl', f'_{prompt_tech}.jsonl')
            out_file_macro = self.out_file.replace('.jsonl', f'_mcheck_{prompt_tech}.jsonl')
            #out_file_micro = self.out_file.replace('.jsonl', f'_qa_{prompt_tech}_micro.jsonl')

            data = load_jsonl(in_file_mod)
            #data = data[:2]

            item_scores = []
            for item in data:
                claims = sent_tokenize(item[f'mgt_{prompt_tech}{trans}'])
                docs = [item['src_inf']]*len(claims)
                pred_label, raw_prob, _, _ = self.scorer.score(docs=docs, claims=claims, )
                item_scores.append(float(np.mean(pred_label)))

            eval_score = float(np.mean(item_scores))
            save_jsonl([{'mean_support': eval_score}], out_file_macro)

if __name__ == "__main__":
    print("GPU available", torch.cuda.is_available(), flush=True)
    print("GPU cout", torch.cuda.device_count(), flush=True)
    print("GPU name ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found", flush=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, required=True, help='Language of the texts (e.g., "en")')
    parser.add_argument('--in_file', type=str, required=True, help='Input JSONL file path')
    parser.add_argument('--out_file', type=str, required=True, help='Output file path for metrics')
    parser.add_argument('--prompt_techs', type=str, nargs='+', required=True, help='Prompting techniques to evaluate')
    args = parser.parse_args()
    
    evaluator = MCheckEvaluator(args.lang, args.in_file, args.out_file, args.prompt_techs)
    evaluator.run_evaluation()
