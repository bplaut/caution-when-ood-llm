from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import argparse
import torch as t

# Still need to try beam search at some point
class Generator(object):
    def __init__(self, args):        
        if args['model'] == 'Mistral-raw':
            model_name = 'mistralai/Mistral-7B-v0.1'
        elif args['model'] == 'Mistral':
            model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
        elif args['model'] == 'Zephyr':
            model_name = 'HuggingFaceH4/zephyr-7b-beta'
        elif args['model'] == 'gpt2':
            model_name = 'gpt2'
        elif args['model'] == 'Llama-13b-raw':
            model_name = 'meta-llama/Llama-2-13b-hf'
        elif args['model'] == 'Llama-13b':
            model_name = 'meta-llama/Llama-2-13b-chat-hf'
        elif args['model'] == 'Llama-7b-raw':
            model_name = 'meta-llama/Llama-2-7b-hf'
        elif args['model'] == 'Llama-7b':
            model_name = 'meta-llama/Llama-2-7b-chat-hf'
        elif args['model'] == 'Llama-70b':
            model_name = 'meta-llama/Llama-2-70b-chat-hf'
        else:
            raise Exception("Unrecognized model name. Try python generate_text -h")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.args = args

        if self.args['num_responses'] > 1 and (self.args['interactive'] or not self.args['do_sample']):
            self.num_responses = 1
        else:
            self.num_responses = self.args['num_responses']
            
    def min_max_logit(self, scores, response_idx, normalize=True):
        # scores has shape (response_length, num_responses, vocab_size). It's a tuple of tensors
        scores_tensor = t.stack(list(scores), dim=0)
        if normalize:
            scores_tensor = t.exp(scores_tensor) / t.sum(t.exp(scores_tensor), dim=2, keepdim=True)
        (max_logit_per_token, _) = t.max(scores_tensor, dim=2)
        (min_among_max_logits, indices) = t.min(max_logit_per_token, dim=0)
        # Small bug to be fixed: this should exclude the padding characters at the end
        return (min_among_max_logits[response_idx], indices[response_idx])
            
    def prepare_for_chat(self, prompts):
        chats = [[{"role": "user", "content": p}] for p in prompts]
        return [self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True, return_tensors="pt") for c in chats]

    def print_output(self, output, model_inputs, prompts, text_outputs):
        print('\n')
        for i in range(len(text_outputs)):
            prompt_idx = i//self.num_responses
            print('PROMPT %d: "%s"\n' % (prompt_idx+1, prompts[prompt_idx]))
            print('OUTPUT %d: "%s"\n' % (i % self.num_responses + 1, text_outputs[i]))
            token_ids = output.sequences[i][len(model_inputs[prompt_idx]):]

            if self.args['num_top_tokens'] > 0:
                (mm_logit, mm_logit_idx) = self.min_max_logit(output.scores, i, normalize=False)
                (mm_prob, mm_prob_idx) = self.min_max_logit(output.scores, i, normalize=True)
                print("Min max prob  =", t_to_str(mm_prob), "| Index =", t_to_str(mm_prob_idx))
                print("Min max logit =", t_to_str(mm_logit), "| Index =", t_to_str(mm_logit_idx))
                for j in range(len(token_ids)):
                    # This isn't that efficient right now, I should be sorting/exping/etc in batch
                    (sorted_scores, top_token_ids) = t.sort(output.scores[j][i], descending=True)
                    sorted_probs = t.exp(sorted_scores) / t.sum(t.exp(sorted_scores))
                    top_tokens = self.tokenizer.batch_decode(top_token_ids[:self.args['num_top_tokens']])
                    if self.args['num_top_tokens'] == 1:
                        max_token_idx_len = len(str(len(token_ids)))
                        idx_str = str(j).zfill(max_token_idx_len) # pad with 0s for prettiness
                        print("Token %s |" % idx_str, t_to_str(sorted_probs[0]), '|', t_to_str(sorted_scores[0]), '|', repr(top_tokens[0]))
                    else:
                        print('\nToken %d:' % j, repr(self.tokenizer.decode(token_ids[j])))
                        print("Top tokens:", top_tokens)
                        print("Top probs:", t_to_str(sorted_probs[:self.args['num_top_tokens']]))
                        print("Top logits:", t_to_str(sorted_scores[:self.args['num_top_tokens']]))
                    
                    if self.tokenizer.decode(token_ids[j]) == self.tokenizer.pad_token:
                        # If we have prompts/responses of different lengths, some will get padded
                        break
                    
            print('\n')

    def generate(self, prompts):
        prompts = self.prepare_for_chat(prompts) if self.args['chat'] and not self.args['interactive'] else prompts # interactive mode is handled separately
        model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        output = self.model.generate(**model_inputs, max_new_tokens=self.args['max_new_tokens'], do_sample=self.args['do_sample'], output_scores=True, num_return_sequences=self.num_responses, return_dict_in_generate=True, renormalize_logits=False)
        output_just_responses = [output.sequences[i][len(model_inputs[i//self.num_responses]):] for i in range(len(output.sequences))] # non-prompt part of the output. i//num_responses in the prompt index
        text_outputs = self.tokenizer.batch_decode(output_just_responses, skip_special_tokens=True)
        self.print_output(output, model_inputs, prompts, text_outputs)
        for (i, response) in enumerate(text_outputs):
            (mm_prob, _) = self.min_max_logit(output.scores, i//self.num_responses, normalize=True)
            if self.args['check_for_halu'] and mm_prob < self.args['threshold']:
                text_outputs[i] = "E. I don't know. (I was going to maybe hallucinate but then I caught myself.)"
        return text_outputs

def parse_args():
    parser = argparse.ArgumentParser(description='Use an LLM to generate text via HuggingFace.')
    parser.add_argument('-m', '--model', type=str, help='Which LLM to use. Check this file for currently supported options and/or add your own.',required=True)
    parser.add_argument('-p', '--prompts', type=str, help='List of prompts, separated by |. For example "Hello my name is Ben|What a time to be alive". If not provided, you will be asked for a prompt by command line.', default=None)
    parser.add_argument('-n', '--max_new_tokens', type=int, help='Number of new tokens to generate on top of the prompt', default=10)
    parser.add_argument('-k', '--num_top_tokens', type=int, help='For each token, print out the top candidates considered by the model and their probabilities', default=0)
    parser.add_argument('-c', '--chat', action="store_true", help='Whether to treat the prompt as a chat message and generate a chatbot response, vs just normal text auto-complete', default=True)
    parser.add_argument('-s', '--do_sample', action="store_true", help='Should we sample from the probability distribution, or greedily pick the most likely token?', default=False)
    parser.add_argument('-r', '--num_responses', type=int, help='Number of responses to generate per prompt. This argument is ignored for greedy decoding, since that only generates one answer.', default=1)
    parser.add_argument('-i', '--interactive', action="store_true", help='Run the LLM in interactive mode where you can go back and forth with the LLM indefinitely. Automatically activates chat mode.', default=False)
    parser.add_argument('-f', '--input_filepath', type=str, default=None, help='The path to the json file containing input data (this is mostly for running experiments/tests)')
    parser.add_argument('-u', '--check_for_halu', action="store_true", help='Should we add an extra check for hallucations? Eventually there will also be an option for why detection method to use.', default=False)
    parser.add_argument('-t', '--threshold', type=float, help='When running the hallucination check, what should we compare with? Right now, this is just a comparison with the min max probability.', default=0.5)
    return dict(vars(parser.parse_args())) # turn it into a dictionary so we can easily modify it
    
def t_to_str(T):
    # Get rid of a bunch of stuff in the tensor format that I don't like
    s = str(T).replace(",\n       device='cuda:0')", "")
    s = s.replace("tensor(", "")
    s = s.replace("\n", "")
    s = s.replace("    ", "")
    s = s.replace(", device='cuda:0')", "")
    target_len = 5 # e.g. 0.534
    return s + '0' * (target_len - len(s)) if '.' in s else s # pad with 0s if decimal
    
def main():
    t.set_printoptions(sci_mode=False, precision=3)
    args = parse_args() 
    generator = Generator(args)

    if args['prompts'] == None:
        prompts = [input("\nEnter an initial prompt:\n")]
        print('\n')
    else:
        prompts = args['prompts'].split('|')
    
    if not generator.args['interactive']:
        output_text = generator.generate(prompts)
    else:
        base_text = ""
        # All the zero indices are because the functions return lists for batching, which doesn't make sense in interactive mode
        user_prompt = prompts[0] # Only use first prompt
        while True:
            # Careful with typing of lists vs strs here
            prompt = base_text + generator.prepare_for_chat([user_prompt])[0]
            base_text = generator.generate([prompt])[0] # output text becomes the base text for next prompt
            user_prompt = input("User response: ")
        
if __name__ == '__main__':
    main()
