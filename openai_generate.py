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
                     'gpt-4-turbo-2024-04-09' if self.args['model'] == 'gpt-4-turbo' else
                     'gpt-4o-2024-11-20' if self.args['model'] == 'gpt-4o' else
                     'o1-mini-2024-09-12' if self.args['model'] == 'o1-mini' else
                     'o1-2024-12-17' if self.args['model'] == 'o1' else
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
        # The output types are lists to support batching (which OpenAI currently doesn't actually support, but hopefully will eventually)
        text_output = [response.choices[0].message.content]
        token_output = [[token.token for token in response.choices[0].logprobs.content]]
        scores = [math.exp(token.logprob) for token in response.choices[0].logprobs.content]
        self.print_output(prompts, text_output, token_output, scores)
        self.verify_greedy_decoding(response.choices[0])
        if text_output[0] != ''.join(token_output[0]):
            print(f"Warning (openai_generate.py): text output doesn't match joined token output: {text_output[0]} vs {''.join(token_output[0])}")
        
        return text_output, token_output, scores

    def print_output(self, prompts, text_outputs, token_outputs, scores):
        print(f'PROMPT: "{prompts[0]}"')
        tokens = token_outputs[0]
        max_token_idx_len = len(str(len(tokens))) # most number of digits for token idx
        for j in range(len(tokens)):
            token = tokens[j]
            score = scores[j]
            idx_str = str(j).zfill(max_token_idx_len) # pad with 0s for prettiness
            print(f"Token {idx_str} | {round(score, 4)} | {token}")
            
    def verify_greedy_decoding(self, output):
        # Check that we're doing greedy decoding
        chosen_token = output.logprobs.content[0]
        top_token = chosen_token.top_logprobs[0]
        if chosen_token.token != top_token.token:
            print (f"Warning (openai_generate.py): Greedy decoding not used. Chosen token: {chosen_token.token}, top token: {top_token.token}")
        if abs(chosen_token.logprob - top_token.logprob) > 0.0001:
            print (f"Warning (openai_generate.py): logprobs don't match up. Chosen token logprob: {chosen_token.logprob}, top token logprob: {top_token.logprob}")

    def compute_confidence_levels(self, text_outputs, token_outputs, scores, choices, normalize=True, product=False):
        # If product, returns the product of all scores. Else Returns the score of the token indicating the answer
        # All the lists/0-indices are because the types are lists to support batching
        if not normalize: # OpenAI api only gives us normalized probabilities, not raw logits
            return [0]
        if product:
            return [math.prod(scores)]
        targets = [c for c in ascii_uppercase][:len(choices[0])]
        tokens = token_outputs[0]
        # First, try to find a token corresponding A./B./C. etc
        for i in range(len(tokens) - 1):
            if tokens[i].strip() in targets and tokens[i + 1] == '.':
                # strip just in case the token is e.g. ' A' instead of 'A'
                return [scores[i]]

        # If we failed, find a token corresponding to A/B/C etc
        for i in range(len(tokens) - 1):
            if tokens[i].strip() in targets:
                return [scores[i]]
        return result
