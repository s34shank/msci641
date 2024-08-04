import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer
import torch
from simpletransformers.classification import ClassificationModel

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

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

def preprocess_text(text, stop_words=None, lemmatizer=None):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)  
        text = re.sub(r'[^\w\s]', '', text)  
        
        words = text.split()
        if stop_words:
            words = [word for word in words if word not in stop_words]
        if lemmatizer:
            words = [lemmatizer.lemmatize(word) for word in words]
        text = ' '.join(words)
    return text

def preprocess_data(df, stop_words=None, lemmatizer=None):
    if 'postText' in df.columns:
        df['postText'] = df['postText'].apply(lambda x: [preprocess_text(p, stop_words, lemmatizer) if isinstance(p, str) else '' for p in x])
    if 'targetParagraphs' in df.columns:
        df['targetParagraphs'] = df['targetParagraphs'].apply(lambda x: [preprocess_text(p, stop_words, lemmatizer) if isinstance(p, str) else '' for p in x])
    if 'targetTitle' in df.columns:
        df['targetTitle'] = preprocess_text(df['targetTitle'], stop_words, lemmatizer)
    return df

def encode_labels(df, mapping):
    if 'tags' in df.columns:
        if df['tags'].apply(lambda x: isinstance(x, list)).any():
            df['tags'] = df['tags'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        df['tags'] = df['tags'].astype(str).map(mapping)
    return df

def tokenize_data(texts, tokenizer, max_length=512):
    flat_texts = [text for sublist in texts for text in sublist if isinstance(text, str)]
    encodings = tokenizer(flat_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    return encodings

def generate_predictions_csv(model, tokenizer, test_df, spoiler_type_map, output_file='predictions.csv'):
    test_encodings = tokenize_data(test_df['postText'], tokenizer)
    test_dataset = TextDataset(test_encodings, [0] * len(test_df))  # Dummy labels for test set

    # Generate predictions
    predictions, _ = model.predict([text for text in test_df['postText']])
    
    # Reverse the spoiler type mapping for conversion
    spoiler_type_map_reverse = {v: k for k, v in spoiler_type_map.items()}
    results = pd.DataFrame({
        'id': test_df.index,  
        'spoilerType': [spoiler_type_map_reverse[p] for p in predictions]
    })

    results.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

def main():
    test_df = load_data('test.jsonl')

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    test_df = preprocess_data(test_df, stop_words, lemmatizer)

    # Load pre-trained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = ClassificationModel(
        'bert',
        'outputs/',
        num_labels=3,
        use_cuda=False
    )
    global spoiler_type_map
    spoiler_type_map = {'phrase': 0, 'passage': 1, 'multi': 2}

    generate_predictions_csv(model, tokenizer, test_df, spoiler_type_map)

if __name__ == '__main__':
    main()
