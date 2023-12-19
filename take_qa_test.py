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
                     'winogrande':('winogrande', 'winogrande_l'),
        }
        # Different datasets have different keys for the questions and answers
        self.get_q = (lambda x:
                      x['ctx'] if dset_name == 'hellaswag' else
                      x['question'] if dset_name in ['arc-easy', 'arc-challenge'] else
                      x['sentence'] if dset_name == 'winogrande' else
                      None)
        self.get_a = (lambda x:
                      ascii_uppercase[int(x['label'])] if dset_name=='hellaswag' else
                      x['answerKey'] if dset_name in ['arc-easy', 'arc-challenge'] else
                      ascii_uppercase[int(x['answer'])-1] if dset_name=='winogrande' else
                      None)
        self.get_choices = (lambda x:
                            x['endings'] if dset_name == 'hellaswag' else
                            x['choices']['text'] if dset_name in ['arc-easy', 'arc-challenge'] else
                            [x['option1'], x['option2']] if dset_name=='winogrande' else
                            None)

        if dset_name not in dset_args:
            raise Exception("Unsupported dataset name")
        self.args = args
        self.questions = load_dataset(*dset_args[dset_name], split='train')

    def write_output(self, correct, incorrect, abstained):
        thresh_str = '_thresh=' + str(self.args['threshold'])
        dataset_str = self.args['dataset'].split("/")[-1]
        output_filepath = "results/%s%s-%s-q%dto%d.txt" % (self.args['model'], thresh_str, dataset_str, self.start_q, self.end_q)
        print('\nWriting results to', output_filepath)
        with open(output_filepath, 'w') as f:
            f.write("Correct: %d | Wrong: %d | Abstained: %d" % (correct, incorrect, abstained))

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

    def determine_llm_answer(self, choices, llm_output):
        # Find first instance of A./B./C. etc, if any
        targets = [c + '.' for c in ascii_uppercase][:len(choices) + 1] # +1 corresponds to the "I don't know" answer we added
        target_idxs = [llm_output.find(t) for t in targets if llm_output.find(t) != -1]
        if len(target_idxs) > 0:
            return llm_output[min(target_idxs)]
        else: # If the model includes the text of the answer without the letter, we'll allow it
            targets = choices + ["I don't know"]
            result = [(i,t,llm_output.find(t)) for (i,t) in enumerate(targets) if llm_output.find(t) != -1]
            if len(result) > 0:
                found_answers = [(i,start) for (i,t,start) in result if llm_output[start:start+len(t)] == t]
                (choice_idx, _) = min(found_answers, key=lambda x:x[1]) # Choice index for the found answer with the earliest starting index in the llm output
                print("Grading note: could not find A./B./C./etc, but did find the text of an answer without the letter")
                return ascii_uppercase[choice_idx]
            return "Could not parse answer"

    def grade_answer(self, choices, correct_answer, llm_output):
        uncertain_answer = ascii_uppercase[len(choices)]
        llm_answer = self.determine_llm_answer(choices, llm_output)
        if llm_answer == uncertain_answer:
            return (f"{llm_answer} (uncertain)", 0)
        elif llm_answer == correct_answer:
            return (f"{llm_answer}. (correct)", 1)
        else:
            return (f"{llm_answer}. (incorrect {correct_answer}.)", -1)

    def run_test(self, start_q, end_q):
        correct = 0
        incorrect = 0
        abstained = 0

        # First assemble all of the prompts
        end_q = min(end_q, len(self.questions))
        num_prompts = end_q - start_q
        prompts = [None] * (num_prompts)
        choices = [None] * (num_prompts)
        question_strings = [None] * (num_prompts)
        correct_answers = [None] * (num_prompts)
        letters_for_uncertain = [None] * (num_prompts)
        for i in range(start_q, end_q):
            question_data = self.questions[i]
            choices_for_q = self.get_choices(question_data)
            question = self.get_q(question_data)
            correct_answer = self.get_a(question_data)
            question_string = self.make_question_string(choices_for_q, question)
            prompt = self.make_prompt(question_string)
            prompts[i - start_q] = prompt
            choices[i - start_q] = choices_for_q
            question_strings[i - start_q] = question_string
            correct_answers[i - start_q] = correct_answer
            letters_for_uncertain[i - start_q] = ascii_uppercase[len(choices_for_q)]

        # Batch inference
        llm_outputs = self.model.generate(prompts, letters_for_uncertain)

        # Grade outputs
        for (i, llm_output) in enumerate(llm_outputs):
            print(f"Question {i+1}: {question_strings[i]}")
            print(f"LLM output: {llm_output}")
            (answer_output, grade) = self.grade_answer(choices[i], correct_answers[i], llm_output)
            print(f"LLM answer: {answer_output}\n")

            if grade == 1:
                correct += 1
            elif grade == -1:
                incorrect += 1
            else:
                abstained += 1
            print(f"Correct: {correct} | Wrong: {incorrect} | Abstained: {abstained}\n")
        return (correct, incorrect, abstained)

def main():
    args = generate_text.parse_args()
    test = Test(args)
    (all_correct, all_incorrect, all_abstained) = (0,0,0)
    for start_q in range(test.start_q, test.end_q, args['batch_size']):
        end_q = min(start_q + args['batch_size'], test.end_q)
        print(f"\nSTARTING NEW BATCH: questions {start_q} to {end_q}\n")
        (correct, incorrect, abstained) = test.run_test(start_q, end_q)
        all_correct += correct
        all_incorrect += incorrect
        all_abstained += abstained
        print(f"\nTOTAL SO FAR: Correct: {all_correct} | Wrong: {all_incorrect} | Abstained: {all_abstained}")
    test.write_output(all_correct, all_incorrect, all_abstained)

if __name__ == '__main__':
    main()
