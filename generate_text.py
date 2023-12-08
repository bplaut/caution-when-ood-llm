from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import argparse
import torch as t

class Generator(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Use an LLM to generate text via HuggingFace.')
        parser.add_argument('-m', '--model', type=str, help='Which LLM to use. Check this file for currently supported options and/or add your own.',required=True)
        parser.add_argument('-p', '--prompts', type=str, help='List of prompts, separated by |. For example "Hello my name is Ben|What a time to be alive". If not provided, you will be asked for a prompt by command line.', default=None)
        parser.add_argument('-n', '--max_new_tokens', type=int, help='Number of new tokens to generate on top of the prompt', default=10)
        parser.add_argument('-t', '--num_top_tokens', type=int, help='For each token, print out the top candidates considered by the model and their probabilities', default=0)
        parser.add_argument('-c', '--chat_mode', action="store_true", help='Whether to treat the prompt as a chat message and generate a chatbot response, vs just normal text auto-complete', default=False)
        parser.add_argument('-s', '--do_sample', action="store_true", help='Should we sample from the probability distribution, or greedily pick the most likely token?', default=False)
        parser.add_argument('-r', '--num_responses', type=int, help='Number of responses to generate per prompt. This argument is ignored for greedy decoding, since that only generates one answer.', default=1)
        parser.add_argument('-i', '--interactive_mode', action="store_true", help='Run the LLM in interactive mode where you can go back and forth with the LLM indefinitely. Only relevant in chat mode.', default=False)
        args = parser.parse_args()
                            
        if args.model == 'Mistral-7b':
            model_name = 'mistralai/Mistral-7B-v0.1'
        elif args.model == 'Zephyr-7b-beta':
            model_name = 'HuggingFaceH4/zephyr-7b-beta'
        elif args.model == 'Zephyr-7b-alpha':
            model_name = 'HuggingFaceH4/zephyr-7b-alpha'
        elif args.model == 'gpt2':
            model_name = 'gpt2'
        elif args.model == 'Llama-13b':
            model_name = 'meta-llama/Llama-2-13b-hf'
        elif args.model == 'Llama-13b-chat':
            model_name = 'meta-llama/Llama-2-13b-chat-hf'
        elif args.model == 'Llama-7b':
            model_name = 'meta-llama/Llama-2-7b-hf'
        elif args.model == 'Llama-7b-chat':
            model_name = 'meta-llama/Llama-2-7b-chat-hf'
        elif args.model == 'Llama-70b-chat':
            model_name = 'meta-llama/Llama-2-70b-chat-hf'
        else:
            raise Exception("Unrecognized model name. Try python generate_text -h")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.args = args

        if self.args.num_responses > 1 and (self.args.interactive_mode or not self.args.do_sample):
            self.num_responses = 1
        else:
            self.num_responses = self.args.num_responses

        prompts = None if args.prompts == None else args.prompts.split('|')
        if prompts != None:
            self.initial_prompts = args.prompts.split('|')
        else:
            self.initial_prompts = [input("\nEnter an initial prompt:\n")]
            print('\n')            
            
    def prepare_for_chat(self, prompts):
        chats = [[{"role": "user", "content": p}] for p in prompts]
        return [self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True, return_tensors="pt") for c in chats]

    def print_output(self, output, model_inputs, prompts):
        text_outputs = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        print('\n')
        for i in range(len(text_outputs)):
            prompt_idx = i//self.num_responses
            print('PROMPT %d: "%s"\n' % (prompt_idx+1, prompts[prompt_idx]))
            print('OUTPUT %d: "%s"\n' % (i % self.num_responses + 1, text_outputs[i]))
            token_ids = output.sequences[i][len(model_inputs[prompt_idx]):]

            if self.args.num_top_tokens > 0:
                for j in range(len(token_ids)):
                    # This isn't that efficient right now, I should be sorting/exping/etc in batch
                    (sorted_scores, top_token_ids) = t.sort(output.scores[j][i], descending=True)
                    sorted_probs = t.exp(sorted_scores) / t.sum(t.exp(sorted_scores))
                    top_tokens = self.tokenizer.batch_decode(top_token_ids[:self.args.num_top_tokens])
                    if self.args.num_top_tokens == 1:
                        print(t_to_str(sorted_probs[0]), '|', t_to_str(sorted_scores[0]), '|', top_tokens[0])
                    else:
                        print('\nToken %d:' % (j+1), repr(self.tokenizer.decode(token_ids[j])))
                        print("Top tokens:", top_tokens)
                        print("Top probs:", t_to_str(sorted_probs[:self.args.num_top_tokens]))
                        print("Top logits:", t_to_str(sorted_scores[:self.args.num_top_tokens]))
                    
                    if self.tokenizer.decode(token_ids[j]) == self.tokenizer.pad_token:
                        # If we have prompts/responses of different lengths, some will get padded
                        break
                    
            print('\n')
        return text_outputs        

    def generate(self, prompts):        
        model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        output = self.model.generate(**model_inputs, max_new_tokens=self.args.max_new_tokens, do_sample=self.args.do_sample, output_scores=True, num_return_sequences=self.num_responses, return_dict_in_generate=True, renormalize_logits=False)
        return self.print_output(output, model_inputs, prompts)

def t_to_str(T):
    # Get rid of a bunch of stuff in the tensor format that I don't like
    s = str(T).replace(",\n       device='cuda:0')", "")
    s = s.replace("tensor(", "")
    s = s.replace("\n", "")
    s = s.replace("    ", "")
    return s.replace(", device='cuda:0')", "")
    
def main():
    t.set_printoptions(sci_mode=False, precision=3)
    generator = Generator()
    prompts = generator.prepare_for_chat(generator.initial_prompts) if generator.args.chat_mode else generator.initial_prompts
    
    output_text = generator.generate(prompts)
    if generator.args.interactive_mode:
        while True:
            # Careful with typing of lists vs strs here
            user_response = input("User response: ")
            new_prompt = '\n'.join(output_text + generator.prepare_for_chat([user_response])) 
            output_text = generator.generate([new_prompt])
        
main()
