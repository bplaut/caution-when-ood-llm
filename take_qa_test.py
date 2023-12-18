import generate_text
import json
from datasets import load_dataset
import random
from string import ascii_uppercase
class Test(object):
    def __init__(self, args):
        bounds = args['question_range'].split('-')
        (self.start_q, self.end_q) = (int(bounds[0]), int(bounds[1]))
        self.model = generate_text.Generator(args)

        dset_name = args['dataset'].lower()
        dset_args = {'hellaswag':('Rowan/hellaswag',),
                        'arc-easy':('ai2_arc', 'ARC-Easy'),
                        'arc-challenge':('ai2_arc', 'ARC-Challenge'),
        }
        # Different datasets have different keys for the questions and answers
        self.get_q = lambda x: (x['ctx'] if dset_name == 'hellaswag' else
                                x['question'] if dset_name == 'ai2_arc' else
                                None)
        self.get_a = lambda x: (ascii_uppercase[int(x['label'])] if dset_name=='hellaswag' else
                                x['answerKey'] if dset_name == 'ai2_arc' else
                                None)
        self.get_choices = lambda x: (x['endings'] if dset_name == 'hellaswag' else
                                      x['choices']['text'] if dset_name == 'ai2_arc' else
                                      None)

        if dset_name not in dset_args:
            raise Exception("Unsupported dataset name")
        self.args = args
        self.questions = load_dataset(*dset_args[dset_name], split='train')

    def write_output(self, correct, incorrect, abstained):
        halu_str = '_halu_thresh_' + str(self.args['threshold']) if self.args['check_for_halu'] else ''
        dataset_str = self.args['dataset'].split("/")[-1]
        output_filepath = "results/%s%s-%s-q%dto%d.txt" % (self.args['model'], halu_str, dataset_str, self.start_q, self.end_q)
        print('\nWriting results to', output_filepath)
        with open(output_filepath, 'w') as f:
            f.write("model = " + self.args['model'] + halu_str + '\n')
            f.write("Correct: %d | Wrong: %d | Abstained: %d\n" % (len(correct), len(incorrect), len(abstained)))
            f.write("Score (even): %d\n" % (len(correct) - len(incorrect)))
            f.write("Score (harsher): %d\n" % (len(correct) - 2 * len(incorrect)))

    def make_question_string(self, choices, question):
        assert(len(choices) <= 25) # we only have 26 capital letters and need 1 for uncertain
        choices_with_uncertain = [ascii_uppercase[i] + '. ' + choice for (i, choice) in enumerate(choices)] + [ascii_uppercase[len(choices)] + ". I don't know"]
        return question + '\n' + '\n'.join(choices_with_uncertain)

    def make_prompt(self, question_string):
            return f"""Below is a multiple-choice question. Choose the letter which best answers the question. Keep your response as brief as possible; just state the letter corresponding to your answer, followed by a period, with no explanation.

Question:

{question_string}

Response:\n
"""

    def grade_answers(self, choices, correct_answer, llm_output):
        # Find first instance of A./B./C. etc, if any
        targets = [c + '.' for c in ascii_uppercase][:len(choices) + 1] # +1 corresponds to the "I don't know" answer we added
        target_idxs = [llm_output.find(t) for t in targets if llm_output.find(t) != -1]
        if len(target_idxs) > 0:
            llm_answer = llm_output[min(target_idxs)]
            uncertain_answer = ascii_uppercase[len(choices)]
            if llm_answer == uncertain_answer:
                return (f"{llm_answer} (uncertain)", 0)
            elif llm_answer == correct_answer:
                return (f"{llm_answer}. (correct)", 1)
            else:
                return (f"{llm_answer}. (incorrect {correct_answer}.)", -1)
        else:
            return (f"Could not parse answer. (incorrect {correct_answer}.)", 1)

    def run_test(self):
        correct = []
        incorrect = []
        abstained = []

        # First assemble all of the prompts
        num_prompts = self.end_q - self.start_q
        prompts = [None] * (num_prompts)
        choices = [None] * (num_prompts)
        question_strings = [None] * (num_prompts)
        correct_answers = [None] * (num_prompts)
        letters_for_uncertain = [None] * (num_prompts)
        for i in range(self.start_q, self.end_q):
            question_data = self.questions[i]
            choices_for_q = self.get_choices(question_data)
            question = self.get_q(question_data)
            correct_answer = self.get_a(question_data)
            question_string = self.make_question_string(choices_for_q, question)
            prompt = self.make_prompt(question_string)
            prompts[i - self.start_q] = prompt
            choices[i - self.start_q] = choices_for_q
            question_strings[i - self.start_q] = question_string
            correct_answers[i - self.start_q] = correct_answer
            letters_for_uncertain[i - self.start_q] = ascii_uppercase[len(choices_for_q)]

        # Batch inference
        llm_outputs = self.model.generate(prompts, letters_for_uncertain)

        # Grade outputs
        for (i, llm_output) in enumerate(llm_outputs):
            print(f"Question {i+1}: {question_strings[i]}")
            print(f"LLM output: {llm_output}")
            (answer_output, grade) = self.grade_answers(choices[i], correct_answers[i], llm_output)
            print(f"LLM answer: {answer_output}\n")

            if grade == 1:
                correct.append(i)
            elif grade == -1:
                incorrect.append(i)
            else:
                abstained.append(i)
            print("Correct: %d | Wrong: %d | Abstained: %d\n" % (len(correct), len(incorrect), len(abstained)))
        return (correct, incorrect, abstained)

def main():
    args = generate_text.parse_args()
    test = Test(args)
    (correct, incorrect, abstained) = test.run_test()
    test.write_output(correct, incorrect, abstained)

if __name__ == '__main__':
    main()
