import jsonlines
import os
import random
def add_negation(hypothesis):
    """
    Add negation to the hypothesis by inserting 'not' or negating common verbs.
    For simplicity, this example uses basic string replacement.
    A more robust solution would use linguistic parsing.
    """
    # List of simple verb negations (can expand this)
    negation_map = {
        "in": "out of",
        "outside": "inside",
        "on": "off",
        "off": "on",
        "up": "down",
        "down": "up",
        "inside": "outside",
        "outside": "inside",
        "at": "not at",
        
        "is": "is not",
        "are": "are not",
        "was": "was not",
        "were": "were not",
        "has": "does not have",
        "have": "do not have",
        "does": "does not",
        "do": "do not",
        "can": "cannot",
        "will": "will not",
        "did": "did not",
        "should": "should not",
        "would": "would not",
        "could": "could not",
        "may": "may not",
        "might": "might not",
        "must": "must not",
        "need": "need not",
        
    }
    skip_words = ["doesn't", "not"]

    # Split hypothesis into words and negate
    words = hypothesis.split()
    for i, skip in enumerate(skip_words):
        if skip in words:
            if skip == "not" and i < len(words) - 1 and words[i+1] == "have":
                return hypothesis, False
            return hypothesis, False
    for i, word in enumerate(words):
        if word.lower() == "nobody":
            words[i] = "somebody"
            return " ".join(words), True
        if word in negation_map and i < len(words) - 1 and (words[i+1] != "not" and words[i+1] != "no"):
            words[i] = negation_map[word]
            return " ".join(words), True  # Apply only the first negation
    
    # If no verb found, prepend "not" to introduce negation
    return hypothesis, False

def add_spelling_error(sentence, count=1):
    
    # Insert a random character into the hypothesis
    for _ in range(count):
        bad_char = chr(random.randint(97, 122))
        loc = random.randint(0, len(sentence)-1)
        if sentence[loc] != ' ':
            sentence = sentence[:loc] + bad_char + sentence[loc+1:]
        else:
            sentence = sentence[:loc+1] + bad_char + sentence[loc+2:]
    return sentence


def process_snli(input_file, output_file, spurious_append_file, only_negated = False):

    """
    Process SNLI dataset to create examples with negated hypotheses.
    """
    negated_examples = []
    extended_premise_examples = []

    
    with jsonlines.open(input_file, 'r') as reader:
        for example in reader:
            premise = example['premise']
            hypothesis = example['hypothesis']
            label = example['label']  # 0: entailment, 1: neutral, 2: contradiction
            
            # Negate hypothesis and flip label appropriately
            negated_hypothesis, negated = add_negation(hypothesis)
            if negated:
                if label == 0:  # entailment -> contradiction
                    new_label = 2
                elif label == 2:  # contradiction -> entailment
                    new_label = 0
                else:  # neutral remains neutral
                    new_label = 1
            else:
                new_label = label
            # Add modified example to the list
            if only_negated and new_label == 1:
                continue
            negated_examples.append({
                "premise": premise,
                "hypothesis": negated_hypothesis,
                "label": new_label
            })
            
            # Add extended premise examples to the list
            extended_premise_examples.append({
                "premise": premise, #add_spelling_error(premise, count=1), 
                "hypothesis": add_spelling_error(hypothesis, count=1), #,hypothesis, #
                "label": label
            })
    
    # Write modified examples to output JSONL file
    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(negated_examples)
    with jsonlines.open(spurious_append_file, 'w') as writer:
        writer.write_all(extended_premise_examples)
    
    print(f"Modified examples written to {output_file}")
#'''
# Input and output file paths 
input_snli_file = "C:\\Users\\gandh\\Work\\Edu\\MSAI coursework\\Course NLP\\fp\\fp-dataset-artifacts\\eval_output\\eval_predictions.jsonl"  # Replace with your SNLI dataset file
output_snli_file = "C:\\Users\\gandh\\Work\\Edu\\MSAI coursework\\Course NLP\\fp\\snli_negation_test.jsonl"
output_snli_append_file = "C:\\Users\\gandh\\Work\\Edu\\MSAI coursework\\Course NLP\\fp\\snli_append_test.jsonl"

# Ensure input file exists
if os.path.exists(input_snli_file):
    process_snli(input_snli_file, output_snli_file, output_snli_append_file)
else:
    print(f"Input file '{input_snli_file}' not found.")
#'''


import spacy
import re

# Load spaCy's language model (use 'en_core_web_sm' for lightweight parsing)
nlp = spacy.load("en_core_web_sm")

def is_awkward(hypothesis):
    """
    Detect awkward constructs in a sentence using heuristics and spaCy parsing.
    Returns True if the sentence is flagged as awkward.
    """
    # Heuristic 1: Detect double negatives using regex
    double_negatives = re.findall(r"\b(not\s+\w+\s+not|nobody\s+.*\bnot|never\s+.*\bnot)\b", hypothesis)
    if double_negatives:
        return True

    # Heuristic 2: Flag sentences starting with negation and a redundant pattern
    if re.search(r"\b(nobody|nothing|never)\s+.*\bnot\b", hypothesis, re.IGNORECASE):
        return True

    # Heuristic 3: Use spaCy parsing to flag invalid constructs
    doc = nlp(hypothesis)
    negations = [token for token in doc if token.dep_ == "neg"]  # Detect negation tokens
    if len(negations) > 1:  # More than one negation is often awkward
        return True

    # Heuristic 4: Flag overly complex constructs with multiple negation-related words
    negation_words = ["not", "no", "nobody", "nothing", "never", "cannot"]
    negation_count = sum(1 for word in hypothesis.split() if word in negation_words)
    if negation_count > 2:  # Adjust threshold as needed
        return True

    return False

def flag_invalid_hypotheses(input_file, output_file, adverse_output_file):
    """
    Reads a JSONL file, flags hypotheses with awkward constructs,
    and writes flagged examples to a new JSONL file.
    """
    import jsonlines

    flagged_examples = []
    attack_examples = []

    with jsonlines.open(input_file, 'r') as reader:
        for example in reader:
            hypothesis = example['hypothesis']
            if is_awkward(hypothesis):
                flagged_examples.append(example)
            else:
                attack_examples.append(example)

    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(flagged_examples)
    with jsonlines.open(adverse_output_file, 'w') as writer:
        writer.write_all(attack_examples)
    
    print(f"Flagged {len(flagged_examples)} awkward examples. Output written to {output_file}")
    print(f"Generated {len(attack_examples)} attack examples. Output written to {adverse_output_file}")

# Input and output file paths
input_snli_file = "C:\\Users\\gandh\\Work\\Edu\\MSAI coursework\\Course NLP\\fp\\snli_negation_test.jsonl"  # Replace with your generated file
output_flagged_file = "C:\\Users\\gandh\\Work\\Edu\\MSAI coursework\\Course NLP\\fp\\flagged_hypotheses.jsonl"
adverse_snli_file = "C:\\Users\\gandh\\Work\\Edu\\MSAI coursework\\Course NLP\\fp\\adverse_snli_file.jsonl"

# Run the flagging process
#flag_invalid_hypotheses(input_snli_file, output_flagged_file, adverse_snli_file)
