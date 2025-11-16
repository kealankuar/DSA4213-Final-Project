import pandas as pd

INPUT_FILE = "../../data/ner/synthetic_ner_dataset_industry_specific.csv"
OUTPUT_FILE = "../../data/ner/ner_training_data_final.csv"

def parse_labeled_tokens(row):
    """Splits the 'word/TAG' string into separate lists of tokens and tags."""
    tokens = []
    tags = []
    # Split the string by spaces to get individual 'word/TAG' pairs
    labeled_pairs = row['labeled_tokens'].split(' ')
    for pair in labeled_pairs:
        # Find the last slash to handle words that might contain slashes
        split_point = pair.rfind('/')
        if split_point == -1:
            # Handle cases where a token might not have a tag (though it should)
            token, tag = pair, 'O'
        else:
            token = pair[:split_point]
            tag = pair[split_point+1:]
        
        tokens.append(token)
        tags.append(tag)
    return tokens, tags

print(f"Reading generated data from '{INPUT_FILE}'...")
df = pd.read_csv(INPUT_FILE)

# This will hold our final, parsed data
parsed_data = []

# Iterate through each generated sentence
for _, row in df.iterrows():
    sentence_id = row['sentence_id']
    tokens, tags = parse_labeled_tokens(row)
    
    # Create a new row for each token
    for i in range(len(tokens)):
        parsed_data.append({
            'sentence_id': sentence_id,
            'token': tokens[i],
            'tag': tags[i]
        })

# Create the final DataFrame in the correct format
final_df = pd.DataFrame(parsed_data)

print("\n--- Data Before Parsing (from your generated file) ---")
print(df.head())

print("\n--- Data After Parsing (ready for training) ---")
print(final_df.head(10)) # Show first 10 tokens

# Save the final, training-ready dataset
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Successfully parsed data and saved to '{OUTPUT_FILE}'")