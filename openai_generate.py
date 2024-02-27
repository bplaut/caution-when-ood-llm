from openai import OpenAI
from dotenv import load_dotenv
import math
import os
from generate_text import Generator
from string import ascii_uppercase

class OpenAIGenerator(Generator):
    def __init__(self, args):
        print("Setting up OpenAI text generator...")
        load_dotenv()
        self.client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))
        self.args = args

    def generate(self, prompts):
        assert len(prompts) == 1 # for now
        full_name = ('gpt-3.5-turbo-0125' if self.args['model'] == 'gpt-3.5-turbo' else
                     'gpt-4-0613' if self.args['model'] == 'gpt-4' else
                     'gpt-4-0125-preview' if self.args['model'] == 'gpt-4-turbo' else
                     self.args['model'])
        response = self.client.chat.completions.create(
            model=full_name,
            top_logprobs=5,
            seed=2549900867,
            top_p=0,
            max_tokens=self.args['max_new_tokens'],
            logprobs=True,
            messages=[{"role": "user", "content": prompts[0]}]
        )
        print("System fingerprint:", response.system_fingerprint, '\n')
        # The output types are lists to support batching (which we'll hopefully add for these models, and which the non-OpenAI models already support)
        text_output = [response.choices[0].message.content]
        token_output = [token.token for token in response.choices[0].logprobs.content]
        scores = [math.exp(token.logprob) for token in response.choices[0].logprobs.content]
        self.print_output(prompts, text_output, token_output, scores)
        return text_output, token_output, scores

    def print_output(self, prompts, text_outputs, token_outputs, scores):
        print(f'PROMPT: "{prompts[0]}"')
        max_token_idx_len = len(str(len(token_outputs))) # most number of digits for token idx
        for j in range(len(token_outputs)):
            token = token_outputs[j]
            score = scores[j]
            idx_str = str(j).zfill(max_token_idx_len) # pad with 0s for prettiness
            print(f"Token {idx_str} | {round(score, 4)} | {token}")

    def compute_confidence_levels(self, text_outputs, token_outputs, scores, choices, normalize=True):
        if not normalize: # OpenAI api only gives us normalized probabilities
            return [0]
        targets = [c for c in ascii_uppercase][:len(choices[0])] # 0 index because it's a list for batching
        # Find the token idx corresponding to a target (A./B./C. etc)
        token_idx = None
        for i in range(len(token_outputs) - 1):
            if token_outputs[i] in targets and token_outputs[i + 1] == '.':
                token_idx = i
                break
        return [scores[token_idx]] if token_idx is not None else [0]
