There are three main files:
1. generate_text.py, which is a command-line interface to generate text via Hugging Face
2. take_qa_test_first_prompt.py, which runs a multiple choice Q&A test using a Hugging Face dataset and generate_text.py.
3. take_qa_test_second_prompt.py, which does the same thing, except, you guessed it...with a different prompt.
All files support the same command line arguments (shown below), although some arguments are only relevant for one file. For example, --dataset is only used for take_qa_test.py.

```
usage: generate_text.py/take_qa_test_<first/second>_prompt.py [-h] -m MODEL [-p PROMPTS] [-n MAX_NEW_TOKENS] [-k NUM_TOP_TOKENS] [-c] [-s]
                        [-r NUM_RESPONSES] [-i] [-d DATASET] [-q QUESTION_RANGE] [-b BATCH_SIZE]
                        [--abstain_option ABSTAIN_OPTION]

Perform text generation and Q&A tasks via Hugging Face models.

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Which LLM to use. Check this file for currently supported options and/or add your
                        own.
  -p PROMPTS, --prompts PROMPTS
                        List of prompts, separated by |. For example "Hello my name is Ben|What a time to
                        be alive". If not provided, you will be asked for a prompt by command line.
  -n MAX_NEW_TOKENS, --max_new_tokens MAX_NEW_TOKENS
                        Number of new tokens to generate on top of the prompt
  -k NUM_TOP_TOKENS, --num_top_tokens NUM_TOP_TOKENS
                        For each token, print out the top candidates considered by the model and their
                        probabilities
  -c, --completion_mode
                        Use traditional auto-complete mode, rather than user-assistant chat
  -s, --do_sample       Should we sample from the probability distribution, or greedily pick the most
                        likely token?
  -r NUM_RESPONSES, --num_responses NUM_RESPONSES
                        Number of responses to generate per prompt. This argument is ignored for greedy
                        decoding, since that only generates one answer.
  -i, --interactive     Run the LLM in interactive chat mode where you can go back and forth with the LLM
                        indefinitely
  -d DATASET, --dataset DATASET
                        The name of the Hugging Face dataset (needed for experiments and such)
  -q QUESTION_RANGE, --question_range QUESTION_RANGE
                        When running a Q&A test, what range of questions should we test? Format is "-q
                        startq-endq", 0 indexed. For example, "-q 0-100".
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Maximum number of prompts to batch together. Only used for experiments
  --abstain_option ABSTAIN_OPTION
                        When running a Q&A test, should we add an option that says "I don't know"?
```
