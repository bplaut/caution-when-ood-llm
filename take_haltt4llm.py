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
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. Only answer the question. Keep token limit low.

### Instruction:
{instruction}

### Response:\n
"""

def grade_answers(question_data, llm_answer):
    correct_answer = None
    for answer in question_data['answers']:
        if answer['correct']:
            correct_answer = answer
            break

    if correct_answer is None:
        return "No correct answer found"

    normalized_llm_answer = llm_answer.lower().strip()
    normalized_correct_answer = correct_answer['text'].lower().strip()

    # lower case of the full text answer is in the llm's answer
    if normalized_correct_answer in normalized_llm_answer:
        return f"{correct_answer['choice']}. {correct_answer['text']} (correct)"

    # Upper case " A." or  " B." or " C." or " D." or " E." for instance
    if f" {correct_answer['choice']}." in llm_answer:
            return f"{correct_answer['choice']}. {correct_answer['text']} (correct)"

    # Upper case " (A)" or  " (B)" or " (C)" or " (D)" or " (E)" for instance
    if f"({correct_answer['choice']})" in llm_answer:
            return f"{correct_answer['choice']}. {correct_answer['text']} (correct)"

    if "i don't know" in normalized_llm_answer or normalized_llm_answer == "d" or normalized_llm_answer == "d.":
        return f"{llm_answer} (uncertain)"

    return f"{llm_answer} (incorrect {correct_answer['choice']}.)"

def run_test(model, trivia_data):
    total_score = 0
    incorrect = []
    unknown = []
    random.shuffle(trivia_data) # Randomize question order
    num_questions = 10

    for i, question_data in enumerate(trivia_data):
        question_string = generate_question_string(question_data)
        prompt = generate_prompt(question_string)
        # The brackets and 0 indices are because the inputs/outputs in Generator are lists, for batching. For example, if you set num_responses > 1. For Q&A testing, we only take the first response.
        formatted_prompt = model.prepare_for_chat([prompt])[0]

        print(f"Question {i+1}: {question_string}")
        llm_answer = model.generate([prompt])[0]
        print(f"LLM answer: {llm_answer}")
        answer_output = grade_answers(question_data, llm_answer)
        print(f"Answer: {answer_output}\n")

        if "(correct)" in answer_output: # They scored it 2/1/0 but I'm doing 1/0/-1 for now
            total_score += 1
        elif "(incorrect" in answer_output:
            incorrect.append((i+1, question_string, answer_output))
            total_score -= 1
        else:
            total_score += 0
            unknown.append((i+1, question_string, answer_output))
        if i == num_questions:
            break
    return total_score
    
def main():
    args = generate_text.parse_args()
    trivia_data = load_trivia_questions(args['input_filepath'])
    model = generate_text.Generator(args)
    score = run_test(model, trivia_data)
    print("MODEL:", args['model'], "| SCORE:", score)


if __name__ == '__main__':
    main()
