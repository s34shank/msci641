import pandas as pd
import json
from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from transformers import RobertaTokenizer
import torch

def load_custom_qa_data(filename):
    with open(filename, 'r') as file:
        data = [json.loads(line) for line in file]
        
    for i, line in enumerate(data[:5]):
        print(f"Line {i} keys: {line.keys()}")
    
    return pd.DataFrame(data)

def prepare_qa_data(df, is_test=False):
    df['context'] = df['targetParagraphs'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    df['question'] = df['postText'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    
    if not is_test:
        if 'spoiler' in df.columns:
            df['answer'] = df['spoiler'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
        else:
            print("Warning: 'spoiler' column is missing in the DataFrame.")
            df['answer'] = ''
    
    qa_data = []
    for i in range(len(df)):
        context = df.loc[i, 'context']
        
        if not is_test:
            answer = df.loc[i, 'answer']
            answer_start = context.find(answer)
        else:
            answer = ""
            answer_start = -1
        
        qa_data.append({
            'context': context,
            'qas': [{
                'question': df.loc[i, 'question'],
                'id': df.loc[i, 'postId'],
                'answers': [{
                    'text': answer,
                    'answer_start': answer_start
                }] if not is_test else [],
                'is_impossible': answer_start == -1
            }]
        })
    return qa_data

def predict_answers(model, test_data):
    predictions = []
    for item in test_data:
        to_predict = [{'context': item['context'], 'qas': item['qas']}]
        preds, _ = model.predict(to_predict)
        if preds and 'answer' in preds[0]:
            answers = preds[0]['answer']
            answer = next((ans for ans in answers if ans), "")
        else:
            answer = ""
        
        predictions.append(answer)
    return predictions

def main():
    test_df = load_custom_qa_data('test.jsonl')
    test_data = prepare_qa_data(test_df, is_test=True)

    model_args = QuestionAnsweringArgs()
    model_args.train_batch_size = 4
    model_args.num_train_epochs = 3
    model_args.logging_dir = './logs'
    model_args.logging_steps = 10
    model_args.save_steps = float('inf')
    model_args.use_cuda = False  
    model_args.overwrite_output_dir = True  
    model_args.save_model_every_epoch = False

    model = QuestionAnsweringModel('roberta', 'outputs', args=model_args, use_cuda=False)
    
    predictions = predict_answers(model, test_data)

    results = pd.DataFrame({
        'id': range(len(test_df)),
        'spoiler': predictions
    })

    results.to_csv('task2_roberta_predictions.csv', index=False)
    print(f"Predictions saved to task2_roberta_predictions.csv")

if __name__ == '__main__':
    main()
