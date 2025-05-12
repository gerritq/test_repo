import os
import argparse
import time
import numpy as np
from tqdm import tqdm
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCase
from utils import load_jsonl, save_jsonl, merge_jsonl
from multiprocessing import Pool, cpu_count

os.environ['OPENAI_API_KEY'] = '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'


class GEvalFactualityMetric:

    def __init__(self, 
                 lang,
                 ds,
                 in_file,
                 out_file,
                 prompt_techs,
                 total_n,):
        self.lang = lang
        self.ds = ds
        self.total_n = total_n
        self.in_file = in_file
        self.out_file = out_file
        self.prompt_techs = prompt_techs
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
        print(f'Start evaluation for ds {self.ds}... ', flush=True)
        start = time.time()
        trans = '_trans' if self.lang != 'en' else ''
        
        for prompt_tech in self.prompt_techs:
            print('\n----------------', flush=True)
            print('----------------', flush=True)
            print(f"Running eval for {self.lang} {prompt_tech} in {self.prompt_techs}", flush=True)

            in_file_mod = self.in_file.replace('.jsonl', f'_{prompt_tech}.jsonl')
            
            #TO DO: needs to be fixed for paras
            # eval_file = f'../../data/{self.lang}/eval/{self.lang}_eval_{self.subset}_meta.jsonl'
            out_file_macro = self.out_file.replace('.jsonl', f'_geval_{prompt_tech}_macro.jsonl')
            out_file_micro = self.out_file.replace('.jsonl', f'_geval_{prompt_tech}_micro.jsonl')

            data = self.load_data(in_file_mod)
            
            scores = []
            reasons = []
            for item in tqdm(data):
                test_case = LLMTestCase(
                                        input=f"Please write a Wikipedia article section about {item['section_title']} for the page on {item['page_title']}.",
                                        actual_output=item[f'mgt_{prompt_tech}{trans}'],
                                        expected_output=item[f'trgt{trans}'])
                self.metric.measure(test_case)
                scores.append(self.metric.score)
                reasons.append({'id': item['id'], 
                                'score': self.metric.score, 
                                'reason': self.metric.reason})
                # print('\n')
                # print(test_case.input)
                # print(test_case.actual_output)
                # print(test_case.expected_output)
                # print(self.metric.score)
                # print(self.metric.reason)

            mean_score = float(np.mean(scores))
            save_jsonl([{'GEval Mean Score': mean_score}], out_file_macro)
            save_jsonl(reasons, out_file_micro)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, required=True, help='Language of the texts')
    parser.add_argument('--ds', type=str, required=True, help='Dataset')
    parser.add_argument('--total_n', type=int, help='Size of data')
    parser.add_argument('--in_file', type=str, required=True, help='Input file path')
    parser.add_argument('--out_file', type=str, required=True, help='Output file path')
    parser.add_argument('--prompt_techs', nargs='+', required=True, help='Prompting techniques')
    args = parser.parse_args()

    geval = GEvalFactualityMetric(
        lang=args.lang,
        ds=args.ds,
        total_n=args.total_n,
        in_file=args.in_file,
        out_file=args.out_file,
        prompt_techs=args.prompt_techs,
    )
    geval.evaluate()


# minimal="Smith was a popular figure in the entertainment industry, leading to numerous endorsement deals throughout her career. She became a spokesperson for various products and brands, including TrimSpa, Guess, and H&M. These endorsements helped to solidify her status as a prominent celebrity and further increase her fame and fortune. Smith's image was often used to promote a wide range of products."
# cp="During an interview on Late Night with Conan O'Brien, Anna Nicole Smith was asked about her 'Playmate diet', to which she immediately responded, \"I eat like a pig.\" In February 2004, Smith became a spokeswoman for TrimSpa and reportedly lost 69 pounds with their product. However, legal action was taken against TrimSpa and Smith for false advertising of the weight loss pill."
# rag = "During an interview on Late Night with Conan O'Brien, Anna Nicole Smith was asked about her 'Playmate diet,' to which she immediately responded with a joke. In October 2003, Smith became a spokeswoman for TrimSpa, a weight-loss supplement. She reportedly lost around 60 to 70 pounds with the help of TrimSpa. Legal action was taken against TrimSpa and Smith for false and misleading marketing."
# trgt="In an interview on Late Night with Conan O'Brien, Smith was asked what her \"Playmate diet\" consisted of. She instantly replied, \"Fried chicken.\" In October 2003, she became a spokeswoman for TrimSpa, which allegedly helped her lose a reported 69 pounds (31 kg). TrimSpa diet product company and Smith were sued in a class-action lawsuit alleging their marketing of a weight loss pill was false or misleading."

# correctness_metric = GEval(
#     name="Factual Accuracy",
#     criteria="Determine whether the actual output is factually correct based on the expected output.",
#     evaluation_params=[LLMTestCaseParams.INPUT, 
#                        LLMTestCaseParams.ACTUAL_OUTPUT, 
#                        LLMTestCaseParams.EXPECTED_OUTPUT],
#                        )



# mgts = [minimal, cp, rag]
# for mgt in mgts:
#     test_case = LLMTestCase(
#             input="Please write a Wikipedia article about Anna Nicole Smith.",
#             actual_output=mgt,
#             expected_output=trgt
#         )

#     correctness_metric.measure(test_case)
#     print(correctness_metric.score)
#     print(correctness_metric.reason)