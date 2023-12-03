from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import argparse
import torch as t

# TODO: chat templates
def main():
    parser = argparse.ArgumentParser(description='Use an LLM to generate text via HuggingFace.')
    parser.add_argument('-m', '--model', type=str, help='Which LLM to use. Current choices are Mistral-7B, Zephyr-7B-alpha, Zephyr-7B-beta, gpt2, Llama-13B-chat-hf',required=True)
    parser.add_argument('-p', '--prompts', type=str, help='List of prompts, separated by |. For example "Hello my name is Ben|What a time to be alive"', required=True)
    parser.add_argument('-n', '--max_new_tokens', type=int, help='Number of new tokens to generate on top of the prompt', default=10)
    parser.add_argument('-t', '--num_top_tokens', type=int, help='For each token, print out the top candidates considered by the model and their probabilities', default=0)
    args = parser.parse_args()
    if args.model == 'Mistral-7B':
        model_name = 'mistralai/Mistral-7B-v0.1'
    elif args.model == 'Zephyr-7B-beta':
        model_name = 'HuggingFaceH4/zephyr-7b-beta'
    elif args.model == 'Zephyr-7B-alpha':
        model_name = 'HuggingFaceH4/zephyr-7b-alpha'
    elif args.model == 'gpt2':
        model_name = 'gpt2'
    elif args.model == 'Llama-13B-chat-hf':
        model_name = 'meta-llama/Llama-2-13b-chat-hf'
    else:
        raise Exception("Unrecognized model name. Try python generate_text -h")
    prompts = args.prompts.split('|')
    generate(model_name, prompts, args.max_new_tokens, args.num_top_tokens)
        
    
def generate(model_name, prompts, max_new_tokens, num_top_tokens):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
    print(prompts)
    model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
    do_sample = False
    num_return_sequences = 2 if do_sample else 1

    output = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, output_scores=True, num_return_sequences=num_return_sequences, return_dict_in_generate=True, renormalize_logits=False)
    # TODO: Try renormalize_logits=False
    text_outputs = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)

    print('\n')
    for i in range(len(text_outputs)):
        prompt_idx = i//num_return_sequences
        print('PROMPT %d: "%s"\n' % (prompt_idx+1, prompts[prompt_idx]))
        print('OUTPUT: "%s"\n' % text_outputs[i])
        token_ids = output.sequences[i][-max_new_tokens:]

        if num_top_tokens > 0:
            for j in range(max_new_tokens):
                print('Token %d:' % (j+1), repr(tokenizer.decode(token_ids[j])))
                all_probs_for_token = t.exp(output.scores[j][i])
                (sorted_probs, token_ids_by_prob) = t.sort(all_probs_for_token, descending=True)
                print("Top tokens: ", tokenizer.batch_decode(token_ids_by_prob[:num_top_tokens]))
                print("Top probs: ", sorted_probs[:num_top_tokens])
        print('\n\n')

main()
