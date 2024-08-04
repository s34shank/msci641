import pandas as pd
import re
from transformers import BertTokenizer
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

def load_preprocessed_data(file_path):
    with open(file_path, 'rb') as f:
        return torch.load(f)

def dataset_to_df(dataset, tokenizer):
    data = []
    for item in dataset:
        input_ids = item['input_ids'].tolist()
        labels = item['labels'].tolist()
        texts = tokenizer.convert_ids_to_tokens(input_ids)
        data.append({'text': texts, 'labels': labels})
    return pd.DataFrame(data)

def main():
    train_df = load_data('train.jsonl')
    val_df = load_data('val.jsonl')
    test_df = load_data('test.jsonl')

    print("Train DataFrame columns:", train_df.columns)
    print("Validation DataFrame columns:", val_df.columns)
    print("Test DataFrame columns:", test_df.columns)

    train_df = preprocess_data(train_df)
    val_df = preprocess_data(val_df)
    test_df = preprocess_data(test_df)

    spoiler_type_map = {'phrase': 0, 'passage': 1, 'multi': 2}
    train_df = encode_labels(train_df, spoiler_type_map)
    val_df = encode_labels(val_df, spoiler_type_map)

    if 'tags' not in train_df.columns or 'tags' not in val_df.columns:
        print("Error: 'tags' column is missing in one or more DataFrames.")
        return

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_encodings = tokenize_data(train_df['postText'], tokenizer)
    val_encodings = tokenize_data(val_df['postText'], tokenizer)
    test_encodings = tokenize_data(test_df['postText'], tokenizer)

    train_dataset = TextDataset(train_encodings, train_df['tags'].tolist())
    val_dataset = TextDataset(val_encodings, val_df['tags'].tolist())
    test_dataset = TextDataset(test_encodings, [0] * len(test_df))  # Dummy labels for test set

    torch.save(train_dataset, 'train_dataset.pt')
    torch.save(val_dataset, 'val_dataset.pt')
    torch.save(test_dataset, 'test_dataset.pt')

    train_dataset = load_preprocessed_data('train_dataset.pt')
    val_dataset = load_preprocessed_data('val_dataset.pt')
    
    train_df = dataset_to_df(train_dataset, tokenizer)
    val_df = dataset_to_df(val_dataset, tokenizer)

    model_args = ClassificationArgs()
    model_args.num_train_epochs = 3
    model_args.train_batch_size = 8
    model_args.eval_batch_size = 8
    model_args.warmup_steps = 500
    model_args.weight_decay = 0.01
    model_args.logging_steps = 10

    model = ClassificationModel(
        'bert',
        'bert-base-uncased',
        num_labels=3,
        args=model_args,
    use_cuda=False
    )

    model.train_model(train_df, eval_df=val_df)
    model.save_model()

if __name__ == '__main__':
    main()