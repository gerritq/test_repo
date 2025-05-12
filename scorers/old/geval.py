import os
import argparse
import time
import numpy as np
from tqdm import tqdm
from deepeval.metrics import GEval, SummarizationMetric
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCase
from utils import load_jsonl, save_jsonl, merge_jsonl
from multiprocessing import Pool, cpu_count

os.environ['OPENAI_API_KEY'] = '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'


def process_item(args):
    """Process a single item for G-Eval measurement"""
    item, lang, prompt_tech = args
    trans = '_trans' if lang != 'en' else ''
    
    metric = GEval(name="Factual Accuracy",
                  criteria="Determine whether the actual output is factually correct based on the expected output.",
                  evaluation_params=[LLMTestCaseParams.INPUT, 
                                    LLMTestCaseParams.ACTUAL_OUTPUT, 
                                    LLMTestCaseParams.EXPECTED_OUTPUT])
    
    test_case = LLMTestCase(
                input=f"Please write a Wikipedia article section about {item['section_title']} for the page on {item['page_title']}.",
                actual_output=item[f'mgt_{prompt_tech}{trans}'],
                expected_output=item[f'trgt{trans}'])
    
    metric.measure(test_case)
    return {
        'id': item['id'], 
        'score': metric.score, 
        'reason': metric.reason
    }


class GEvalFactualityMetric:

    def __init__(self, 
                 lang,
                 ds,
                 in_file,
                 out_file,
                 prompt_techs,
                 total_n,
                 n_workers=6):
        self.lang = lang
        self.ds = ds
        self.total_n = total_n
        self.in_file = in_file
        self.out_file = out_file
        self.prompt_techs = prompt_techs
        self.n_workers = n_workers
        self.metric = GEval(name="Factual Accuracy",
                            criteria="Determine whether the actual output is factually correct based on the expected output.",
                            evaluation_params=[LLMTestCaseParams.INPUT, 
                                                LLMTestCaseParams.ACTUAL_OUTPUT, 
                                                LLMTestCaseParams.EXPECTED_OUTPUT])

    def load_data(self, in_file):
        data = load_jsonl(in_file)
        n_per_tertile = self.total_n // 3

        data_by_tertiles = {'low': [], 'medium': [], 'high': []}

        for item in data:
            data_by_tertiles[item['word_tertile']].append(item)

        return (data_by_tertiles['low'][:n_per_tertile] + 
                data_by_tertiles['medium'][:n_per_tertile] + 
                data_by_tertiles['high'][:n_per_tertile])


    def evaluate(self):
        print(f'Start evaluation for ds {self.ds} using {self.n_workers} workers... ', flush=True)
        start = time.time()
        
        for prompt_tech in self.prompt_techs:
            print('\n----------------', flush=True)
            print('----------------', flush=True)
            print(f"Running eval for {self.lang} {prompt_tech} in {self.prompt_techs}", flush=True)

            in_file_mod = self.in_file.replace('.jsonl', f'_{prompt_tech}.jsonl')
            
            out_file_macro = self.out_file.replace('.jsonl', f'_geval_{prompt_tech}_macro.jsonl')
            out_file_micro = self.out_file.replace('.jsonl', f'_geval_{prompt_tech}_micro.jsonl')

            data = self.load_data(in_file_mod)
            
            # Prepare arguments for parallel processing
            process_args = [(item, self.lang, prompt_tech) for item in data]
            
            # Process items in parallel
            with Pool(processes=self.n_workers) as pool:
                results = list(tqdm(
                    pool.imap(process_item, process_args),
                    total=len(process_args),
                    desc=f"Processing {prompt_tech}"
                ))
            
            # Extract scores and reasons
            scores = [result['score'] for result in results]
            
            mean_score = float(np.mean(scores))
            save_jsonl([{'GEval Mean Score': mean_score}], out_file_macro)
            save_jsonl(results, out_file_micro)
            
            end = time.time()
            print(f"Completed {prompt_tech} evaluation in {end - start:.2f} seconds", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, required=True, help='Language of the texts')
    parser.add_argument('--ds', type=str, required=True, help='Dataset')
    parser.add_argument('--total_n', type=int, help='Size of data')
    parser.add_argument('--in_file', type=str, required=True, help='Input file path')
    parser.add_argument('--out_file', type=str, required=True, help='Output file path')
    parser.add_argument('--prompt_techs', nargs='+', required=True, help='Prompting techniques')
    parser.add_argument('--n_workers', type=int, default=6, help='Number of worker processes (default: CPU count - 1)')
    args = parser.parse_args()

    geval = GEvalFactualityMetric(
        lang=args.lang,
        ds=args.ds,
        total_n=args.total_n,
        in_file=args.in_file,
        out_file=args.out_file,
        prompt_techs=args.prompt_techs,
        n_workers=args.n_workers,
    )
    geval.evaluate()