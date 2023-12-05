Currently the only file is generate_text.py, which streamlines text generation via Hugging Face.

usage: generate_text.py [-h] -m MODEL [-p PROMPTS] [-n MAX_NEW_TOKENS]
                        [-t NUM_TOP_TOKENS] [-c] [-s] [-r NUM_RESPONSES] [-i]

Use an LLM to generate text via HuggingFace.

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Which LLM to use. Check this file for currently supported options
                        and/or add your own.
  -p PROMPTS, --prompts PROMPTS
                        List of prompts, separated by |. For example "Hello my name is
                        Ben|What a time to be alive". If not provided, you will be asked
                        for a prompt by command line.
  -n MAX_NEW_TOKENS, --max_new_tokens MAX_NEW_TOKENS
                        Number of new tokens to generate on top of the prompt
  -t NUM_TOP_TOKENS, --num_top_tokens NUM_TOP_TOKENS
                        For each token, print out the top candidates considered by the
                        model and their probabilities
  -c, --chat_mode       Whether to treat the prompt as a chat message and generate a
                        chatbot response, vs just normal text auto-complete
  -s, --do_sample       Should we sample from the probability distribution, or greedily
                        pick the most likely token?
  -r NUM_RESPONSES, --num_responses NUM_RESPONSES
                        Number of responses to generate per prompt. This argument is
                        ignored for greedy decoding, since that only generates one
                        answer.
  -i, --interactive_mode
                        Run the LLM in interactive mode where you can go back and forth
                        with the LLM indefinitely. Only relevant in chat mode.
