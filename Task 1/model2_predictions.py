import pandas as pd
import re
from transformers import RobertaTokenizer
import torch
from simpletransformers.classification import ClassificationModel, ClassificationArgs

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

def load_data(filename):
    try:
        return pd.read_json(filename, lines=True)
    except ValueError as e:
        print(f"Error loading JSON Lines file {filename}: {e}")
        return pd.DataFrame()

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)  
        text = re.sub(r'[^\w\s]', '', text)  
    return text

def preprocess_data(df):
    if 'postText' in df.columns:
        df['postText'] = df['postText'].apply(lambda x: [preprocess_text(p) if isinstance(p, str) else '' for p in x])
    if 'targetParagraphs' in df.columns:
        df['targetParagraphs'] = df['targetParagraphs'].apply(lambda x: [preprocess_text(p) if isinstance(p, str) else '' for p in x])
    if 'targetTitle' in df.columns:
        df['targetTitle'] = preprocess_text(df['targetTitle'])
    return df

def tokenize_data(texts, tokenizer, max_length=512):
    flat_texts = [text for sublist in texts for text in sublist if isinstance(text, str)]
    encodings = tokenizer(flat_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    return encodings

def generate_predictions_csv(model, tokenizer, test_df, output_file='predictions.csv'):
    # Tokenize test data
    test_encodings = tokenize_data(test_df['postText'], tokenizer)
    test_dataset = TextDataset(test_encodings, [0] * len(test_df))  

    predictions, _ = model.predict([text for text in test_df['postText']])
    
    results = pd.DataFrame({
        'id': range(len(test_df)),  
        'spoilerType': [list(spoiler_type_map.keys())[p] for p in predictions]
    })
    
    results.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

def main():
    test_df = load_data('test.jsonl')
    test_df = preprocess_data(test_df)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    model = ClassificationModel(
        'roberta',
        'outputs/',  
        num_labels=3,
        use_cuda=False
    )

    global spoiler_type_map
    spoiler_type_map = {'phrase': 0, 'passage': 1, 'multi': 2}

    generate_predictions_csv(model, tokenizer, test_df)

if __name__ == '__main__':
    main()
