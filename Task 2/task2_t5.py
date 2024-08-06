import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from transformers import T5Tokenizer
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
import torch
from torch.utils.data import Dataset

nltk.download('stopwords')

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        self.labels = tokenizer(labels, truncation=True, padding=True, max_length=max_length, return_tensors='pt')['input_ids']

    def __getitem__(self, idx):
        item = {key: self.encodings[key][idx] for key in self.encodings}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def load_data(filename):
    try:
        return pd.read_json(filename, lines=True)
    except ValueError as e:
        print(f"Error loading JSON Lines file {filename}: {e}")
        return pd.DataFrame()

def preprocess_text(text, stemmer=None, stop_words=None):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        if stop_words:
            words = [word for word in words if word not in stop_words]
        if stemmer:
            words = [stemmer.stem(word) for word in words]
        text = ' '.join(words)
    return text

def preprocess_data(df, stemmer=None, stop_words=None):
    if 'postText' in df.columns:
        df['postText'] = df['postText'].apply(lambda x: ' '.join([preprocess_text(p, stemmer, stop_words) if isinstance(p, str) else '' for p in x]))
    if 'targetParagraphs' in df.columns:
        df['targetParagraphs'] = df['targetParagraphs'].apply(lambda x: ' '.join([preprocess_text(p, stemmer, stop_words) if isinstance(p, str) else '' for p in x]))
    if 'spoiler' in df.columns:
        df['spoiler'] = df['spoiler'].apply(lambda x: preprocess_text(x, stemmer, stop_words) if isinstance(x, str) else '')
    return df

def prepare_data(df):
    df['inputText'] = df['postText'] + ' ' + df['targetParagraphs']
    return df

def main():
    train_df = load_data('train.jsonl')
    val_df = load_data('val.jsonl')
    test_df = load_data('test.jsonl')

    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    train_df = preprocess_data(train_df, stemmer, stop_words)
    val_df = preprocess_data(val_df, stemmer, stop_words)
    test_df = preprocess_data(test_df, stemmer, stop_words)

    train_df = prepare_data(train_df)
    val_df = prepare_data(val_df)
    test_df = prepare_data(test_df)

    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    train_dataset = TextDataset(
        texts=train_df['inputText'].tolist(),
        labels=train_df['spoiler'].tolist(),
        tokenizer=tokenizer,
        max_length=512
    )

    val_dataset = TextDataset(
        texts=val_df['inputText'].tolist(),
        labels=val_df['spoiler'].tolist(),
        tokenizer=tokenizer,
        max_length=512
    )

    model_args = Seq2SeqArgs(
        num_train_epochs=3,
        train_batch_size=8,
        eval_batch_size=8,
        max_seq_length=512,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=10,
        overwrite_output_dir=True,
        output_dir='./results',
        evaluate_during_training=True,
        evaluate_during_training_steps=10,
        save_steps=10
    )
    
    model = Seq2SeqModel(
        encoder_decoder_type='t5',
        encoder_decoder_name='t5-small',
        args=model_args,
        use_cuda=torch.cuda.is_available()
    )

    train_data = train_df[['inputText', 'spoiler']].rename(columns={'inputText': 'input_text', 'spoiler': 'target_text'})
    eval_data = val_df[['inputText', 'spoiler']].rename(columns={'inputText': 'input_text', 'spoiler': 'target_text'})

    model.train_model(train_data, eval_data=eval_data)
    model.save_model()

    test_texts = test_df['inputText'].tolist()
    test_ids = test_df['id'].tolist()  

    predictions = model.predict(test_texts)

    results = pd.DataFrame({
        'id': test_ids,
        'spoiler': predictions
    })

    results.to_csv('task2_t5_predictions.csv', index=False)
    print(f"Predictions saved to task2_t5_predictions.csv")

if __name__ == '__main__':
    main()
