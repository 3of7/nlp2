import jsonlines
from spellchecker import SpellChecker
import re
# Initialize the spellchecker
spell = SpellChecker()

def correct_text(text):
    """
    Correct spelling mistakes in the text using PySpellChecker.
    Punctuation and symbols are ignored in the correction process.
    Returns the corrected text and a list of corrections made.
    """
    # Tokenize the text into words while keeping punctuation separate
    tokens = re.findall(r"[\w']+|[.,!?;()\-]", text)

    corrections = []
    corrected_tokens = []

    for token in tokens:
        # Check only alphanumeric tokens for spelling
        if token.isalpha() or "'" in token:  # Include contractions like "don't"
            corrected_word = spell.correction(token) 
            if not corrected_word:
                corrected_word = token  
            if token != corrected_word:
                corrections.append({
                    "original_word": token,
                    "corrected_word": corrected_word
                })
            corrected_tokens.append(corrected_word)
        else:
            # Keep punctuation/symbols unchanged
            corrected_tokens.append(token)

    # Reassemble the corrected text
    corrected_text = " ".join(corrected_tokens).replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    return corrected_text, corrections


# File paths
input_file = "C:\\Users\\gandh\\Work\\Edu\\MSAI coursework\\Course NLP\\fp\\snli_append_test.jsonl" #"C:\\Users\\gandh\\Work\\Edu\\MSAI coursework\\Course NLP\\fp\\fp-dataset-artifacts\\eval_output\\eval_predictions.jsonl"  # Replace with your SNLI dataset file

output_file = "C:\\Users\\gandh\\Work\\Edu\\MSAI coursework\\Course NLP\\fp\\fp-dataset-artifacts\\preprocessed.jsonl"  # Replace with your SNLI dataset file
report_file = "C:\\Users\\gandh\\Work\\Edu\\MSAI coursework\\Course NLP\\fp\\fp-dataset-artifacts\\preprop_correction_report.jsonl"  # Replace with your SNLI dataset file


count = 0
# Open the input and output files
with jsonlines.open(input_file) as reader, \
     jsonlines.open(output_file, mode='w') as writer, \
     jsonlines.open(report_file, mode='w') as report_writer:

    for obj in reader:
        # Extract fields
        premise = obj.get('premise', '')
        hypothesis = obj.get('hypothesis', '')
        label = obj.get('label', None)

        # Correct premise and hypothesis
        corrected_premise, premise_corrections = correct_text(premise)
        corrected_hypothesis, hypothesis_corrections = correct_text(hypothesis)

        # Write the corrected data to the output file
        corrected_entry = {
            "premise": corrected_premise,
            "hypothesis": corrected_hypothesis,
            "label": label
        }
        writer.write(corrected_entry)

        # Write the corrections report
        correction_report = {
            "original_premise": premise,
            "corrected_premise": corrected_premise,
            "premise_corrections": premise_corrections,
            "original_hypothesis": hypothesis,
            "corrected_hypothesis": corrected_hypothesis,
            "hypothesis_corrections": hypothesis_corrections
        }
        report_writer.write(correction_report)
        count += 1
        print("Processed", count/98.42 , "%")
