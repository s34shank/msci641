import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from itertools import chain
from transformers import T5Tokenizer

nltk.download('stopwords')

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output)
        return prediction, hidden, cell

# Define the Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.size(0)
        max_len = trg.size(1)
        vocab_size = self.decoder.embedding.num_embeddings
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(self.device)

        hidden, cell = self.encoder(src)
        
        input = trg[:, 0]
        
        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input.unsqueeze(1), hidden, cell)
            outputs[:, t, :] = output.squeeze(1)
            
            top1 = output.argmax(2)
            input = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1.squeeze(1)
        
        return outputs

# Define a custom collate function
def collate_fn(batch):
    src_batch = [item['src'] for item in batch]
    trg_batch = [item['trg'] for item in batch]

    src_batch_padded = pad_sequence(src_batch, padding_value=0, batch_first=True)
    trg_batch_padded = pad_sequence(trg_batch, padding_value=0, batch_first=True)

    return {'src': src_batch_padded, 'trg': trg_batch_padded}

# Define the Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        src = torch.tensor(self.tokenizer.encode(self.texts[idx], max_length=self.max_length, truncation=True, padding='max_length'), dtype=torch.long)
        trg = torch.tensor(self.tokenizer.encode(self.labels[idx], max_length=self.max_length, truncation=True, padding='max_length'), dtype=torch.long)
        return {'src': src, 'trg': trg}

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

    # Tokenize and create vocab
    all_texts = train_df['inputText'].tolist() + train_df['spoiler'].tolist()
    all_texts = [tokenizer.encode(text, truncation=True, padding='max_length', max_length=512) for text in all_texts]
    vocab = set(chain.from_iterable(all_texts))
    vocab_size = len(vocab)

    embedding_dim = 256
    hidden_dim = 512
    n_layers = 2
    dropout = 0.5
    max_length = 512

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = Encoder(vocab_size, embedding_dim, hidden_dim, n_layers, dropout).to(device)
    decoder = Decoder(vocab_size, embedding_dim, hidden_dim, n_layers, dropout).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the padding index
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = TextDataset(
        texts=train_df['inputText'].tolist(),
        labels=train_df['spoiler'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    val_dataset = TextDataset(
        texts=val_df['inputText'].tolist(),
        labels=val_df['spoiler'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # Training loop
    for epoch in range(3):
        model.train()
        for batch in train_loader:
            src = batch['src'].to(device)
            trg = batch['trg'].to(device)
            
            optimizer.zero_grad()
            output = model(src, trg, teacher_forcing_ratio=0.5)
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1} complete')

    torch.save(model.state_dict(), 'seq2seq_lstm_model.pth')

    # Evaluation
    model.eval()
    test_texts = test_df['inputText'].tolist()
    test_ids = test_df['id'].tolist()

    predictions = []
    with torch.no_grad():
        for text in test_texts:
            src = torch.tensor(tokenizer.encode(text, max_length=max_length, truncation=True, padding='max_length'), dtype=torch.long).unsqueeze(0).to(device)
            output = model(src, src, teacher_forcing_ratio=0.0)
            pred = output.argmax(dim=-1).squeeze().cpu().numpy()
            predictions.append(tokenizer.decode(pred))

    results = pd.DataFrame({
        'id': test_ids,
        'spoiler': predictions
    })

    results.to_csv('task2_lstm_predictions.csv', index=False)
    print(f"Predictions saved to task2_lstm_predictions.csv")

if __name__ == '__main__':
    main()
