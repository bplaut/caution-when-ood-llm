from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import argparse
import torch as t

def set_params():
    parser = argparse.ArgumentParser(description='Use an LLM to generate text via HuggingFace.')
    parser.add_argument('-m', '--model', type=str, help='Which LLM to use. Check this file for currently supported options and/or add your add.',required=True)
    parser.add_argument('-p', '--prompts', type=str, help='List of prompts, separated by |. For example "Hello my name is Ben|What a time to be alive"', required=True)
    parser.add_argument('-n', '--max_new_tokens', type=int, help='Number of new tokens to generate on top of the prompt', default=10)
    parser.add_argument('-t', '--num_top_tokens', type=int, help='For each token, print out the top candidates considered by the model and their probabilities', default=0)
    parser.add_argument('-c', '--chat_mode', action="store_true", help='Whether to treat the prompt as a chat message and generate a chatbot response, vs just normal text auto-complete', default=False)
    parser.add_argument('-s', '--do_sample', action="store_true", help='Should we sample from the probability distribution, or greedily pick the most likely token?', default=False)
    parser.add_argument('-r', '--num_responses', type=int, help='Number of responses to generate per prompt. This argument is ignored for greedy decoding, since that only generates one answer.', default=1)
    
    args = parser.parse_args()
    if args.model == 'Mistral-7B':
        model_name = 'mistralai/Mistral-7B-v0.1'
    elif args.model == 'Zephyr-7B-beta':
        model_name = 'HuggingFaceH4/zephyr-7b-beta'
    elif args.model == 'Zephyr-7B-alpha':
        model_name = 'HuggingFaceH4/zephyr-7b-alpha'
    elif args.model == 'gpt2':
        model_name = 'gpt2'
    elif args.model == 'Llama-13B-chat':
        model_name = 'meta-llama/Llama-2-13b-chat-hf'
    elif args.model == 'Llama-13B':
        model_name = 'meta-llama/Llama-2-13b-hf'
    else:
        raise Exception("Unrecognized model name. Try python generate_text -h")
    prompts = args.prompts.split('|')
    return (model_name, prompts, args.max_new_tokens, args.num_top_tokens, args.chat_mode, args.do_sample, args.num_responses)

def prepare_for_chat(prompts, tokenizer):
    chats = [[{"role": "user", "content": p}] for p in prompts]
    return [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True, return_tensors="pt") for c in chats]

    
def generate(model_name, prompts, max_new_tokens, num_top_tokens, chat_mode, do_sample, num_responses):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
    if chat_mode:
        prompts = prepare_for_chat(prompts, tokenizer)
    if not do_sample:
        num_responses = 1
    model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

    output = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, output_scores=True, num_return_sequences=num_responses, return_dict_in_generate=True, renormalize_logits=True)
    text_outputs = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)


    print('\n')
    for i in range(len(text_outputs)):
        prompt_idx = i//num_responses
        print('PROMPT %d: "%s"\n' % (prompt_idx+1, prompts[prompt_idx]))
        print('OUTPUT %d: "%s"\n' % (i % num_responses + 1, text_outputs[i]))
        token_ids = output.sequences[i][len(model_inputs[prompt_idx]):]

        if num_top_tokens > 0:
            for j in range(len(token_ids)):
                print('Token %d:' % (j+1), repr(tokenizer.decode(token_ids[j])))
                all_probs_for_token = t.exp(output.scores[j][i])
                (sorted_probs, token_ids_by_prob) = t.sort(all_probs_for_token, descending=True)
                print("Top tokens:", tokenizer.batch_decode(token_ids_by_prob[:num_top_tokens]))
                print("Top probs:", sorted_probs[:num_top_tokens])
                
                if tokenizer.decode(token_ids[j]) == tokenizer.pad_token:
                    # If we have prompts/responses of different lengths, some will get padded
                    break 
        print('\n\n')

def main():
    params = set_params()
    generate(*params)
        
main()
