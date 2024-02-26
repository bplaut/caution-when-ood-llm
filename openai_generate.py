from openai import OpenAI
from dotenv import load_dotenv
import math
import os
from generate_text import Generator

class OpenAIGenerator(Generator):
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))

    def generate(self, prompts):
        assert len(prompts) == 1 # for now
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            top_logprobs=5,
            seed=2549900867,
            top_p=0,
            logprobs=True,
            messages=[{"role": "user", "content": prompts[0]}]
        )
        print("System fingerprint:", response.system_fingerprint, '\n')
        text_output = response.choices[0].message.content
        token_output = [token.token for token in response.choices[0].logprobs.content]
        scores = [math.exp(token.logprob) for token in response.choices[0].logprobs.content]
        return text_output, token_output, scores

    def compute_confidence_levels(self, text_outputs, token_outputs, scores, choices, normalize=True):
        return 0
