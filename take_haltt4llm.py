import generate_text
import json
import time
import random

def load_trivia_questions(file_path):
    with open(file_path, 'r') as f:
        trivia_data = json.load(f)
    return trivia_data

def generate_question_string(question_data):
    question = question_data['question']
    choices = [f"    {answer['choice']}. {answer['text']}\n" if answer != question_data['answers'][-1] else f"    {answer['choice']}. {answer['text']}" for answer in question_data['answers']]
    return f"{question}\n{''.join(choices)}"

def generate_prompt(instruction):
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. Only answer the question. Keep your response as brief as possible; just state the letter corresponding to your answer, followed by a period."

### Instruction:
{instruction}

### Response:\n
"""

def grade_answers(question_data, llm_response):
    # This grading should probably be improved. It accepted "Toy Story 2" when "Toy Story" was correct
    correct_answer = None
    for answer in question_data['answers']:
        if answer['correct']:
            correct_answer = answer
            break

    if correct_answer is None:
        return "No correct answer found"

    # Find first instance of A. or B. or C. or D. or E., if any
    targets = ['A.', 'B.', 'C.', 'D.', 'E.']
    target_idxs = [llm_response.find(t) for t in targets if llm_response.find(t) != -1]
    if len(target_idxs) > 0:
        llm_answer = llm_response[ min(target_idxs)]
        if llm_answer == 'E':
            return f"{llm_response} (uncertain)"
        elif llm_answer == f"{correct_answer['choice']}":
            return f"{llm_answer}. (correct)"
        else:
            return f"{llm_answer}. (incorrect {correct_answer['choice']}.)"
    else:
        print("Could not find an answer in the LLM response.")
        return f"{llm_answer}. (incorrect {correct_answer['choice']}.)"

def run_test(model, trivia_data, start_q, end_q):
    correct = []
    incorrect = []
    abstained = []

    for i, question_data in enumerate(trivia_data):
        if start_q <= i < end_q:
            question_string = generate_question_string(question_data)
            prompt = generate_prompt(question_string)
            # The brackets and 0 indices are because the inputs/outputs in Generator are lists, for batching. For example, if you set num_responses > 1. For Q&A testing, we only take the first response.
            formatted_prompt = model.prepare_for_chat([prompt])[0]

            print(f"Question {i+1}: {question_string}")
            llm_response = model.generate([prompt])[0]
            print(f"LLM response: {llm_response}")
            answer_output = grade_answers(question_data, llm_response)
            print(f"Answer: {answer_output}\n")

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
    (start_q, end_q) = (0, 1600)
    trivia_data = load_trivia_questions(args['input_filepath'])
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
