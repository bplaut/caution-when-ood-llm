import generate_text
import json
from datasets import load_dataset
import random
import string

def make_question_string(question_data):
    num_original_choices = len(question_data['endings'])
    assert(num_original_choices <= 25) # we only have 26 capital letters
    choices = [string.ascii_uppercase[i] + '. ' + question_data['endings'][i] for i in range(num_original_choices)] + [string.ascii_uppercase[num_original_choices] + ". I don't know"]
    return question_data['ctx'] + '\n' + '\n'.join(choices)

def make_prompt(question_string):
        return f"""Below is a multiple-choice question. Choose the letter which best answers the question. Keep your response as brief as possible; just state the letter corresponding to your answer, followed by a period, with no explanation."

### Question:
{question_string}

### Response:\n
"""

def grade_answers(question_data, llm_output):
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

def run_test(model, questions, start_q, end_q):
    correct = []
    incorrect = []
    abstained = []
    for i, question_data in enumerate(questions):
        if start_q <= i < end_q:
            question_string = make_question_string(question_data)
            prompt = make_prompt(question_string)
            # The brackets and 0 indices are because the inputs/outputs in Generator are lists, for batching. For example, if you set num_responses > 1. For Q&A testing, we only take the first response.
            formatted_prompt = model.prepare_for_chat([prompt])[0]

            print(f"Question {i+1}: {question_string}")
            letter_for_uncertain = string.ascii_uppercase[len(question_data['endings'])]
            llm_output = model.generate([prompt], letter_for_uncertain)[0]
            print(f"LLM output: {llm_output}")
            answer_output = grade_answers(question_data, llm_output)
            print(f"LLM answer: {answer_output}\n")

            if "(correct)" in answer_output:
                correct.append((i+1, question_string, answer_output))
            elif "(incorrect" in answer_output:
                incorrect.append((i+1, question_string, answer_output))
            else:
                abstained.append((i+1, question_string, answer_output))
            print("Correct: %d | Wrong: %d | Abstained: %d\n" % (len(correct), len(incorrect), len(abstained)))
    return (correct, incorrect, abstained)

def load_questions(dataset_name):
    name_map = {'hellaswag':'Rowan/hellaswag'}
    return load_dataset(name_map[dataset_name.lower()], split='train')

def main():
    args = generate_text.parse_args()
    (start_q, end_q) = (0, 500)
    questions = load_questions(args['dataset'])
    model = generate_text.Generator(args)
    (correct, incorrect, abstained) = run_test(model, questions, start_q, end_q)
    halu_str = '_halu_check_' + str(args['threshold']) if args['check_for_halu'] else ''
    dataset_str = args['dataset'].split("/")[-1]
    output_filename = "results/%s%s-%s-q%dto%d.txt" % (args['model'], halu_str, dataset_str, start_q, end_q)
    with open(output_filename, 'w') as f:
        f.write("model = " + args['model'] + halu_str + '\n')
        f.write("Correct: %d | Wrong: %d | Abstained: %d\n" % (len(correct), len(incorrect), len(abstained)))
        f.write("Score (even grading): %d\n" % (len(correct) - len(incorrect)))
        f.write("Score (harsher grading): %d\n" % (len(correct) - 2 * len(incorrect)))
        f.write("Score (very harsh): %d" % (len(correct) - 4 * len(incorrect)))

if __name__ == '__main__':
    main()
