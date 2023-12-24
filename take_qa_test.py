import generate_text
import json
import os
from datasets import load_dataset, concatenate_datasets
import random
from string import ascii_uppercase

class Test(object):
    def __init__(self, args):
        bounds = args['question_range'].split('-')
        (self.start_q, self.end_q) = (int(bounds[0]), int(bounds[1]))
        self.model = generate_text.Generator(args)
        self.args = args

        dset_name = args['dataset'].lower()
        # hellaswag and winogrande test splits don't have labels. Truthful_qa only has validation
        # Combine different splits for winogrande and ARC to have more total questions
        dset = (load_dataset('Rowan/hellaswag', split='train') if dset_name=='hellaswag' else
                concatenate_datasets([load_dataset('ai2_arc', 'ARC-Challenge', split=s) for s in ['train','test','validation']]) if dset_name=='arc' else
                concatenate_datasets([load_dataset('winogrande', 'winogrande_debiased', split=s) for s in ['train','validation']]) if dset_name == 'winogrande' else 
                load_dataset('cais/mmlu', 'all', split='test') if dset_name == 'mmlu' else
                load_dataset('truthful_qa', 'multiple_choice', split='validation') if dset_name == 'truthfulqa' else None)
        if dset is None:
            raise Exception(f"Unsupported dataset name: {dset_name}")
        self.questions = dset
        self.end_q = min(self.end_q, len(self.questions))
                     
        # Different datasets have different keys for the questions and answers
        self.get_q = (lambda x:
                      x['ctx'] if dset_name == 'hellaswag' else
                      x['question'] if dset_name in ['arc','mmlu','truthfulqa'] else
                      x['sentence'] if dset_name == 'winogrande' else None)
        self.get_a = (lambda x:
                      self.make_letter(x['label']) if dset_name == 'hellaswag' else
                      self.make_letter(x['answerKey']) if dset_name == 'arc' else
                      self.make_letter(x['answer'],1) if dset_name=='winogrande' else
                      self.make_letter(x['answer']) if dset_name == 'mmlu' else
                      self.make_letter(x['mc1_targets']['labels'].index(1)) if dset_name == 'truthfulqa' else None)
        self.get_choices = (lambda x:
                            x['endings'] if dset_name == 'hellaswag' else
                            x['choices']['text'] if dset_name == 'arc' else
                            [x['option1'], x['option2']] if dset_name=='winogrande' else
                            x['choices'] if dset_name == 'mmlu' else
                            x['mc1_targets']['choices'] if dset_name == 'truthfulqa' else None)
        
    def make_letter(self, answer, offset=0):
        # Puts "answer" in the form of a letter if it isn't already
        answer = str(answer)
        if answer in ascii_uppercase:
            return answer
        elif answer in [str(n) for n in range(10)]:
            return ascii_uppercase[int(answer) - offset]
        else:
            raise Exception(f"Unknown answer format: {answer}")

    def write_output(self, grades, confidence_levels):
        dataset_str = self.args['dataset'].split("/")[-1]
        p = "results/grades_per_question"
        os.makedirs(p, exist_ok=True)
        output_filepath = f"{p}/{dataset_str}_{self.args['model']}-q{self.start_q}to{self.end_q}.txt"
        print('\nWriting results to', output_filepath)
        with open(output_filepath, 'w') as f:
            f.write("grade confidence_level\n")
            for (g,c) in zip(grades, confidence_levels):
                g_str = "Correct" if g == 1 else "Abstained" if g == 0 else "Wrong"
                f.write(f"{g_str} {c}\n")

    def make_question_string(self, choices, question):
        if len(choices) > 25:
            raise Exception("We only have 26 capital letters, so you can't have more than 26 answer options (including 'I don't know'")
        choices_with_uncertain = [ascii_uppercase[i] + '. ' + choice for (i, choice) in enumerate(choices)] + [ascii_uppercase[len(choices)] + ". I don't know"]
        return question + '\n' + '\n'.join(choices_with_uncertain)

    def make_prompt(self, question_string):
            return f"""Below is a multiple-choice question. Choose the letter which best answers the question. Keep your response as brief as possible; just state the letter corresponding to your answer, followed by a period, with no explanation.

Question:

{question_string}

Response:\n
"""

    def determine_llm_answer(self, choices, llm_output):
        # Look for A./B./C. etc. If fail, look for the text of an answer. If fail, look for A/B/C etc
        targets_v1 = [c + '.' for c in ascii_uppercase][:len(choices) + 1] # +1 corresponds to the "I don't know" answer we added
        v1_idxs = [llm_output.find(t) for t in targets_v1 if llm_output.find(t) != -1]
        targets_v2 = choices + ["I don't know"]
        v2_result = [(i,t,llm_output.find(t)) for (i,t) in enumerate(targets_v2) if llm_output.find(t) != -1]
        targets_v3 = [c for c in ascii_uppercase][:len(choices) + 1]
        v3_idxs = [llm_output.find(t) for t in targets_v3 if llm_output.find(t) != -1]
        if len(v1_idxs) > 0: # found A./B./C. etc
            return llm_output[min(v1_idxs)]
        elif len(v2_result) > 0:
            found = [(i,start) for (i,t,start) in v2_result if llm_output[start:start+len(t)] == t]
            (choice_idx, _) = min(found, key=lambda x:x[1])
            print("Grading: could not find A./B./C./etc, but did find the text of an answer")
            return ascii_uppercase[choice_idx]
        elif len(v3_idxs) > 0:
            print("Grading: could not find A./B./C./etc or the text of an answer, but did find A/B/C etc")
            return llm_output[min(v3_idxs)]
        else:
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
        assert(start_q < end_q)
        # First assemble all of the prompts
        print("Forming prompts...\n")
        num_prompts = end_q - start_q
        prompts = [None] * (num_prompts)
        choices = [None] * (num_prompts)
        question_strings = [None] * (num_prompts)
        correct_answers = [None] * (num_prompts)
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

        # Batch inference
        print("Running inference...\n")
        (llm_outputs, confidence_levels) = self.model.generate(prompts, choices) 

        # Grade outputs
        print("Grading answers...\n")
        grades = [None] * len(llm_outputs)
        for (i, llm_output) in enumerate(llm_outputs):
            print(f"Question {i+1+start_q}: {question_strings[i]}")
            print(f"LLM output: {llm_output}")
            (answer_output, grade) = self.grade_answer(choices[i], correct_answers[i], llm_output)
            print(f"LLM answer: {answer_output}\n")
            confidence_str = generate_text.t_to_str(confidence_levels[i])
            print(f"Confidence level: {confidence_str}\n")
            grades[i] = grade
        return (grades, confidence_levels)

def main():
    args = generate_text.parse_args()
    test = Test(args)
    all_grades = []
    all_confidence_levels = []
    for start_q in range(test.start_q, test.end_q, args['batch_size']):
        end_q = min(start_q + args['batch_size'], test.end_q)
        if args['batch_size'] > 1:
            print(f"\nSTARTING NEW BATCH: questions {start_q} to {end_q}\n")
        (grades, confidence_levels) = test.run_test(start_q, end_q)
        all_grades += grades
        all_confidence_levels += confidence_levels
    test.write_output(all_grades, all_confidence_levels)

if __name__ == '__main__':
    main()
