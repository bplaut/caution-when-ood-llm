import generate_text
import json
from datasets import load_dataset
import random
import string

def load_trivia_questions(file_path):
    with open(file_path, 'r') as f:
        trivia_data = json.load(f)
    return trivia_data

def load_hellaswag():
    dataset = load_dataset("Rowan/hellaswag", split="train")
    return dataset

def generate_question_string_hellaswag(question_data):
    num_original_choices = len(question_data['endings'])
    assert(num_original_choices <= 25) # we only have 26 capital letters
    choices = [string.ascii_uppercase[i] + '. ' + question_data['endings'][i] for i in range(num_original_choices)] + [string.ascii_uppercase[num_original_choices] + ". I don't know"]
    return question_data['ctx'] + '\n' + '\n'.join(choices)

def generate_question_string(question_data):
    question = question_data['question']
    # Remove the "none of the above answer, and relabel E to be D
    choices = [f"    {answer['choice']}. {answer['text']}\n" if answer != question_data['answers'][-1] else f"    {answer['choice']}. {answer['text']}" for answer in question_data['answers'] if answer['choice'] != 'D']
    return f"{question}\n{''.join(choices)}".replace("E. I don't know", "D. I don't know")

def generate_prompt(instruction):
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. Only answer the question. Keep your response as brief as possible; just state the letter corresponding to your answer, followed by a period."

### Instruction:
{instruction}

### Response:\n
"""

def grade_answers(question_data, llm_output):
    # This grading should probably be improved. It accepted "Toy Story 2" when "Toy Story" was correct
    correct_answer = None
    for answer in question_data['answers']:
        if answer['correct']:
            correct_answer = answer
            break

    if correct_answer is None:
        return "No correct answer found"

    # Find first instance of A. or B. or C. or D., if any
    # Maybe add support for (A), (B), etc. The prompt specifically asks for a period, though
    targets = ['A.', 'B.', 'C.', 'D.']
    target_idxs = [llm_output.find(t) for t in targets if llm_output.find(t) != -1]
    if len(target_idxs) > 0:
        llm_answer = llm_output[ min(target_idxs)]
        if llm_answer == 'D':
            return f"{llm_answer} (uncertain)"
        elif llm_answer == f"{correct_answer['choice']}":
            return f"{llm_answer}. (correct)"
        else:
            return f"{llm_answer}. (incorrect {correct_answer['choice']}.)"
    else:
        return f"Could not parse answer. (incorrect {correct_answer['choice']}.)"

def grade_answers_hellaswag(question_data, llm_output):
    # This grading should probably be improved. It accepted "Toy Story 2" when "Toy Story" was correct

    # Find first instance of A./B./C. etc, if any
    targets = [c + '.' for c in string.ascii_uppercase][:len(question_data['endings']) + 1] # +1 corresponds to the "I don't know" answer we added
    target_idxs = [llm_output.find(t) for t in targets if llm_output.find(t) != -1]
    if len(target_idxs) > 0:
        llm_answer = llm_output[min(target_idxs)]
        correct_answer = string.ascii_uppercase[int(question_data['label'])]
        uncertain_answer = string.ascii_uppercase[len(question_data['endings'])]
        if llm_answer == uncertain_answer:
            return f"{llm_answer} (uncertain)"
        elif llm_answer == correct_answer:
            return f"{llm_answer}. (correct)"
        else:
            return f"{llm_answer}. (incorrect {correct_answer}.)"
    else:
        return f"Could not parse answer. (incorrect {correct_answer}.)"

def run_test(model, trivia_data, start_q, end_q):
    correct = []
    incorrect = []
    abstained = []

    for i, question_data in enumerate(trivia_data):
        if start_q <= i < end_q:
            question_string = generate_question_string_hellaswag(question_data)
            prompt = generate_prompt(question_string)
            # The brackets and 0 indices are because the inputs/outputs in Generator are lists, for batching. For example, if you set num_responses > 1. For Q&A testing, we only take the first response.
            formatted_prompt = model.prepare_for_chat([prompt])[0]

            print(f"Question {i+1}: {question_string}")
            letter_for_uncertain = string.ascii_uppercase[len(question_data['endings'])]
            llm_output = model.generate([prompt], letter_for_uncertain)[0]
            print(f"LLM output: {llm_output}")
            answer_output = grade_answers_hellaswag(question_data, llm_output)
            print(f"LLM answer: {answer_output}\n")

            if "(correct)" in answer_output:
                correct.append((i+1, question_string, answer_output))
            elif "(incorrect" in answer_output:
                incorrect.append((i+1, question_string, answer_output))
            else:
                abstained.append((i+1, question_string, answer_output))
            print("Correct: %d | Wrong: %d | Abstained: %d\n" % (len(correct), len(incorrect), len(abstained)))
    return (correct, incorrect, abstained)
    
def main():
    args = generate_text.parse_args()
    (start_q, end_q) = (0, 500)
    # trivia_data = load_trivia_questions(args['input_filepath'])
    trivia_data = load_hellaswag()
    model = generate_text.Generator(args)
    (correct, incorrect, abstained) = run_test(model, trivia_data, start_q, end_q)
    halu_str = '_halu_check_' + str(args['threshold']) if args['check_for_halu'] else ''
    input_str = args['input_filepath'].split("/")[-1].split("_questions")[0]
    output_filename = "results/%s%s-%s-q%dto%d.txt" % (args['model'], halu_str, input_str, start_q, end_q)
    with open(output_filename, 'w') as f:
        f.write("model = " + args['model'] + halu_str + '\n')
        f.write("Correct: %d | Wrong: %d | Abstained: %d\n" % (len(correct), len(incorrect), len(abstained)))
        f.write("Score (even grading): %d\n" % (len(correct) - len(incorrect)))
        f.write("Score (harsher grading): %d\n" % (len(correct) - 2 * len(incorrect)))
        f.write("Score (very harsh): %d" % (len(correct) - 4 * len(incorrect)))

if __name__ == '__main__':
    main()
