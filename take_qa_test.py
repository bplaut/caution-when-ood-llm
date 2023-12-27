import generate_text
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
                load_dataset('piqa', split='train') if dset_name == 'piqa' else
                load_dataset('truthful_qa', 'multiple_choice', split='validation') if dset_name == 'truthfulqa' else None)
        if dset is None:
            raise Exception(f"Unsupported dataset name: {dset_name}")
        self.questions = list(dset)
        random.shuffle(self.questions)
        self.end_q = min(self.end_q, len(self.questions))
                     
        # Different datasets have different ways of accessing questions/answers/choices
        self.get_q = (lambda x:
                      x['ctx'] if dset_name == 'hellaswag' else
                      x['question'] if dset_name in ['arc','mmlu','truthfulqa'] else
                      x['goal'] if dset_name == 'piqa' else
                      x['sentence'] if dset_name == 'winogrande' else None)
        self.get_a = (lambda x:
                      self.make_index(x['label']) if dset_name == 'hellaswag' else
                      self.make_index(x['answerKey'],1) if dset_name == 'arc' else # nearly all ARC answers are letters, but a few are numbers with offset 1
                      self.make_index(x['answer'],1) if dset_name=='winogrande' else
                      x['answer'] if dset_name == 'mmlu' else
                      x['label'] if dset_name == 'piqa' else
                      x['mc1_targets']['labels'].index(1) if dset_name == 'truthfulqa' else None)
        self.get_choices = (lambda x:
                            x['endings'] if dset_name == 'hellaswag' else
                            x['choices']['text'] if dset_name == 'arc' else
                            [x['option1'], x['option2']] if dset_name=='winogrande' else
                            [x['sol1'], x['sol2']] if dset_name == 'piqa' else
                            x['choices'] if dset_name == 'mmlu' else
                            x['mc1_targets']['choices'] if dset_name == 'truthfulqa' else None)
        
    def make_index(self, answer, offset=0):
        # Puts "answer" in the form of a index if it isn't already
        answer = str(answer)
        if answer in ascii_uppercase:
            return ascii_uppercase.index(answer)        
        elif answer in [str(n) for n in range(10)]:
            return int(answer) - offset
        else:
            raise Exception(f"Unknown answer format: {answer}")

    def write_output(self, grades, confidence_levels):
        dataset_str = self.args['dataset'].split("/")[-1]
        p = "results/"
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

    def compute_confidence_levels(self, text_outputs, token_outputs, scores, choices, just_letter=True):
        # Find the max probability for the token which determines the answer
        confidence_levels = [None] * len(text_outputs)
        for (i, response) in enumerate(text_outputs):
            num_choices = len(choices[i]) if len(choices) > i else 0
            # Main targets are (1) uppercase letters corresponding to choices, (2) same but with weird underscore in front because some models include that. Backup targets are the first tokens in the text of each choice. If just_letter=True, ignore (3).
            main_targets = ([c for c in ascii_uppercase][:num_choices] +
                          ['▁' + c for c in ascii_uppercase][:num_choices])
            backup_targets = [self.model.tokenizer.tokenize(choice)[0] for choice in choices[i]]
            token_idx1 = self.model.first_token_instance(token_outputs[i], main_targets)
            token_idx2 = self.model.first_token_instance(token_outputs[i], backup_targets)
            token_idx = token_idx1 if just_letter or token_idx1<len(token_outputs[i]) else token_idx2
            (conf, _) = self.model.min_max_logit(scores, i, lo=token_idx, hi=token_idx+1, normalize=True)
            confidence_levels[i] = conf
        return confidence_levels
    
    def determine_llm_answer(self, choices, llm_output):
        # Look for A./B./C. etc. 
        targets_v1 = [c + '.' for c in ascii_uppercase][:len(choices) + 1] # +1 because of "I don't know"
        v1_idxs = [llm_output.find(t) for t in targets_v1 if llm_output.find(t) != -1]
        # If that fails, look for the text of an answer. Normalize casing.
        targets_v2 = [(i,choice.lower()) for (i,choice) in enumerate(choices + ["I don't know"])]
        output_lower = llm_output.lower()
        v2_result = [(i,t,output_lower.find(t)) for (i,t) in targets_v2 if output_lower.find(t) != -1]
        # If that fails, look for just A/B/C
        targets_v3 = [c for c in ascii_uppercase][:len(choices) + 1]
        v3_idxs = [llm_output.find(t) for t in targets_v3 if llm_output.find(t) != -1]
        if len(v1_idxs) > 0: # found A./B./C. etc
            return llm_output[min(v1_idxs)]
        elif len(v2_result) > 0: # found text of answer
            found = [(i,start_idx) for (i,t,start_idx) in v2_result
                     if output_lower[start_idx:start_idx+len(t)] == t]
            (i, _) = min(found, key=lambda x:x[1])
            print("Grading note: could not find A./B./C./etc, but did find the text of an answer")
            return ascii_uppercase[i]
        elif len(v3_idxs) > 0: # found A/B/C/
            print("Grading note: could not find A./B./C./etc or the text of an answer, but did find A/B/C etc")
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
            # Shuffle choices. Recall that self.get_a returns an index
            correct_answer_text = choices_for_q[self.get_a(question_data)]
            random.shuffle(choices_for_q)
            correct_answer = ascii_uppercase[choices_for_q.index(correct_answer_text)]
            question_string = self.make_question_string(choices_for_q, question)
            prompt = self.make_prompt(question_string)
            prompts[i - start_q] = prompt
            choices[i - start_q] = choices_for_q
            question_strings[i - start_q] = question_string
            correct_answers[i - start_q] = correct_answer

        # Batch inference
        print("Running inference...\n")
        (text_outputs, token_outputs, scores) = self.model.generate(prompts)
        confidence_levels = self.compute_confidence_levels(text_outputs, token_outputs, scores, choices)

        # Grade outputs
        print("Grading answers...\n")
        grades = [None] * len(text_outputs)
        for (i, llm_output) in enumerate(text_outputs):
            print(f"Question {i+1+start_q}: {question_strings[i]}")
            print(f"LLM output: {llm_output}")
            (answer_output, grade) = self.grade_answer(choices[i], correct_answers[i], llm_output)
            print(f"LLM answer: {answer_output}\n")
            confidence_str = generate_text.t_to_str(confidence_levels[i])
            print(f"Confidence level: {confidence_str}\n")
            grades[i] = grade
        return (grades, confidence_levels)

def main():
    random.seed(2549900867) # We'll randomize the order of questions and of answer choices, but we want every run to have the same randomization
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
