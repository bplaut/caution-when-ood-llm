import generate_text
import json
from datasets import load_dataset
import random
import string

class Test(object):
    def __init__(self, args):
        (self.start_q, self.end_q) = (0, 1000)
        self.model = generate_text.Generator(args)

        dataset_name = args['dataset'].lower()
        dataset_args = {'hellaswag':('Rowan/hellaswag'),
                        'arc-easy':('ai2_arc', 'ARC-Easy'),
                        'arc-challenge':('ai2_arc', 'ARC-Challenge'),
        }
        # Different datasets have different keys for the questions and answers
        self.get_q = lambda x: (x['ctx'] if dataset_name == 'hellaswag' else
                                x['question'] if dataset_name == 'ai2_arc' else
                                None)
        self.get_a = lambda x: (string.ascii_uppercase[int(x['label'])] if dataset_name == 'hellaswag' else
                                x['answerKey'] if dataset_name == 'ai2_arc' else
                                None)
        self.get_choices = lambda x: (x['endings'] if dataset_name == 'hellaswag' else
                                      x['choices']['text'] if dataset_name == 'ai2_arc' else
                                      None)

        if dataset_name not in dataset_args:
            raise Exception("Unsupported dataset name")
        self.args = args
        self.questions = load_dataset(*dataset_args[dataset_name], split='train')

    def write_output(self, correct, incorrect, abstained):
        halu_str = '_halu_thresh_' + str(self.args['threshold']) if self.args['check_for_halu'] else ''
        dataset_str = self.args['dataset'].split("/")[-1]
        output_filename = "results/%s%s-%s-q%dto%d.txt" % (self.args['model'], halu_str, dataset_str, start_q, end_q)
        with open(output_filename, 'w') as f:
            f.write("model = " + self.args['model'] + halu_str + '\n')
            f.write("Correct: %d | Wrong: %d | Abstained: %d\n" % (len(correct), len(incorrect), len(abstained)))
            f.write("Score (even): %d\n" % (len(correct) - len(incorrect)))
            f.write("Score (harsher): %d\n" % (len(correct) - 2 * len(incorrect)))

    def make_question_string(self, choices, question):
        assert(len(choices) <= 25) # we only have 26 capital letters and need 1 for uncertain
        choices_with_uncertain = [string.ascii_uppercase[i] + '. ' + choice for (i, choice) in enumerate(choices)] + [string.ascii_uppercase[len(choices)] + ". I don't know"]
        return question + '\n' + '\n'.join(choices_with_uncertain)

    def make_prompt(self, question_string):
            return f"""Below is a multiple-choice question. Choose the letter which best answers the question. Keep your response as brief as possible; just state the letter corresponding to your answer, followed by a period, with no explanation."

### Question:
{question_string}

### Response:\n
"""

    def grade_answers(self, choices, correct_answer, llm_output):
        # Find first instance of A./B./C. etc, if any
        targets = [c + '.' for c in string.ascii_uppercase][:len(choices) + 1] # +1 corresponds to the "I don't know" answer we added
        target_idxs = [llm_output.find(t) for t in targets if llm_output.find(t) != -1]
        if len(target_idxs) > 0:
            llm_answer = llm_output[min(target_idxs)]
            uncertain_answer = string.ascii_uppercase[len(choices)]
            if llm_answer == uncertain_answer:
                return f"{llm_answer} (uncertain)"
            elif llm_answer == correct_answer:
                return f"{llm_answer}. (correct)"
            else:
                return f"{llm_answer}. (incorrect {correct_answer}.)"
        else:
            return f"Could not parse answer. (incorrect {correct_answer}.)"

    def run_test(self):
        correct = []
        incorrect = []
        abstained = []
        for (i, question_data) in enumerate(self.questions):
            if self.start_q <= i < self.end_q:
                print('question data:', question_data)
                choices = self.get_choices(question_data)
                question = self.get_q(question_data)
                correct_answer = self.get_a(question_data)
                question_string = self.make_question_string(choices, question)
                prompt = self.make_prompt(question_string)
                # The brackets and 0 indices are because the inputs/outputs in Generator are lists, for batching. For example, if you set num_responses > 1. For Q&A testing, we only take the first response.
                formatted_prompt = self.model.prepare_for_chat([prompt])[0]

                print(f"Question {i+1}: {question_string}")
                letter_for_uncertain = string.ascii_uppercase[len(choices)]
                llm_output = self.model.generate([prompt], letter_for_uncertain)[0]
                print(f"LLM output: {llm_output}")
                answer_output = self.grade_answers(choices, correct_answer, llm_output)
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
    test = Test(args)
    (correct, incorrect, abstained) = test.run_test()
    test.write_output(correct, incorrect, abstained)

if __name__ == '__main__':
    main()
