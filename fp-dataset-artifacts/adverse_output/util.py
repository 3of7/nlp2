import json
import csv
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def lexical_overlap(df):
    model = SentenceTransformer('all-mpnet-base-v2')
    df['premise_embedding'] = df['premise'].apply(lambda x: model.encode(x))
    df['hypothesis_embedding'] = df['hypothesis'].apply(lambda x: model.encode(x))
    df['semantic_similarity'] = df.apply(
        lambda row: cosine_similarity(
            [row['premise_embedding']], 
            [row['hypothesis_embedding']]
        )[0][0],
        axis=1
    )
    print(df['semantic_similarity'].corr(df['error']))

def ngram_analysis(df):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words='english')
    error_hypotheses = df[df['error']==1]
    error_hypotheses = error_hypotheses['hypothesis']
    error_hypotheses = vectorizer.fit_transform(error_hypotheses)
    top_ngrams = vectorizer.get_feature_names_out()
    print(top_ngrams)


def explore_basic_error_patterns(df):
    df['error'] = df['predicted_label'] != df['label']
    df.error = df.error.apply(lambda x: 1 if x else 0)
    error_rate_by_label = df.groupby('label')['error'].mean()
    print(error_rate_by_label)
    df['premise_length'] = df['premise'].apply(lambda x: len(x.split()))
    df['hypothesis_length'] = df['hypothesis'].apply(lambda x: len(x.split()))
    error_rate_by_premise_length = df.groupby(['premise_length'])['error'].mean()
    error_rate_by_hypothesis_length = df.groupby(['hypothesis_length'])['error'].mean()
    print(f"Error rate by premise length: {error_rate_by_premise_length}, number of errors per premise length: {df.groupby(['premise_length'])['error'].sum()}" )
    print(f"Error rate by hypothesis length: {error_rate_by_hypothesis_length}, number of errors per hypothesis length: {df.groupby(['hypothesis_length'])['error'].sum()}" )
    print(df.premise_length.corr(df.error))
    print(df.hypothesis_length.corr(df.error))
    df['overlap'] = df.apply(lambda row: lexical_overlap(row['premise'], row['hypothesis']), axis=1)
    print(df['overlap'].corr(df['error'])) #df.overlap.corr()
    #lexical_overlap(df)
    ngram_analysis(df)




from nltk.tokenize import word_tokenize
def lexical_overlap(p, h):
    p_set = set(word_tokenize(p))
    h_set = set(word_tokenize(h))
    return len(p_set & h_set) / len(p_set | h_set)



'''
'''
def convert_jsonl_to_csv(input_json_file, output_csv_file):
# Read the JSONL file line by line
    with open(input_json_file, "r") as file:
        for line in file:
            try:
                json_object = json.loads(line.strip())  # Parse each line as a JSON object
                data.append(json_object)  # Append the object to the list
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")

    # If data is valid and contains dictionaries, convert to CSV
    if data and isinstance(data[0], dict):
        # Extract field names from the first JSON object
        fieldnames = data[0].keys()

        # Write to CSV
        with open(output_csv_file, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()  # Write the header
            writer.writerows(data)  # Write the rows

        print(f"CSV file successfully created: {output_csv_file}")
    else:
        print("No valid JSON data found or structure is not a list of dictionaries.")
'''
'''
if __name__ == "__main__":
    # Load the JSON file
    input_json_file = "adverse_output\\eval_predictions.jsonl"  # Replace with your JSON file path
    output_csv_file = "eval_predictions.csv"  # Replace with your desired CSV file path

    # Initialize an empty list to store JSON objects
    data = []
    #
    convert_jsonl_to_csv(input_json_file, output_csv_file)
    #df = pd.read_json(input_json_file, lines=True)
    #explore_basic_error_patterns(df)

