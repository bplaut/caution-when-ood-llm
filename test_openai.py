import sys
from openai_generate import OpenAIGenerator

def test_compute_confidence_levels(generator):
        # Test the confidence level computation
        text_outputs = ["A. This is a test"]
        token_outputs = [["A", ".", " This", " is", " a", " test"]]
        scores = [5, 3, 1, 10, 11]
        choices = [["hi", "hello", "bye"]]
        assert(generator.compute_confidence_levels(text_outputs, token_outputs, scores, choices) == [5])
        text_outputs = ["This is a test, bye"]
        token_outputs = [["This", " is", " a", " test", ",", " bye"]]
        scores = [5, 3, 1, 10, 11, 12]
        assert(generator.compute_confidence_levels(text_outputs, token_outputs, scores, choices) == [12])
        text_outputs = ["The answer is B, hello"]
        token_outputs = [["The", " answer", " is", " B", " ,", " hello"]]
        scores = [5, 3, 1, 10, 11, 12]
        assert(generator.compute_confidence_levels(text_outputs, token_outputs, scores, choices) == [10])
        text_outputs = ["hello, the answer is B."]
        token_outputs = [["hello", ",", " the", " answer", " is", " B", "."]]
        scores = [5, 3, 1, 10, 11, 12, 13]
        assert(generator.compute_confidence_levels(text_outputs, token_outputs, scores, choices) == [12])
        text_outputs = ["The answer is Freddie Mercury"]
        token_outputs = [["The", " answer", " is", " Freddie", " Mercury"]]
        scores = [5, 3, 1, 10, 11]
        choices = [["hi", "hello", "bye", "Freddie Mercury"]]
        assert(generator.compute_confidence_levels(text_outputs, token_outputs, scores, choices) == [10])
        text_outputs = ["The answer is Freddie Mercury, or B"]
        token_outputs = [["The", " answer", " is", " Freddie", " Mercury", ",", " or", " B"]]
        scores = [5, 3, 1, 10, 11, 12, 13, 14]
        assert(generator.compute_confidence_levels(text_outputs, token_outputs, scores, choices) == [10])
        text_outputs = ["The answer is Freddie Mercury, not B."]
        token_outputs = [["The", " answer", " is", " Freddie", " Mercury", ",", " or", " B", "."]]
        scores = [5, 3, 1, 10, 11, 12, 13, 14, 15]
        assert(generator.compute_confidence_levels(text_outputs, token_outputs, scores, choices) == [14])
        text_outputs = ["The answer is Freddie Mercury, not B. or C."]
        token_outputs = [["The", " answer", " is", " Freddie", " Mercury", ",", " not", " B", ".", " or", " C", "."]]
        scores = [5, 3, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        assert(generator.compute_confidence_levels(text_outputs, token_outputs, scores, choices) == [14])
        print("All tests passed")

def main():
    generator = OpenAIGenerator({})
    test_compute_confidence_levels(generator)

if __name__ == "__main__":
    main()
    
