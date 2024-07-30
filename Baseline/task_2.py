#!/usr/bin/python3
import argparse
import json
import csv

def parse_args():
    parser = argparse.ArgumentParser(description='This is a baseline for task 2 that spoils each clickbait post with the title of the linked page.')
    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The spoiled posts in jsonl format.', required=False)
    return parser.parse_args()

def predict(inputs):
    for i in inputs:
        yield {'id': i['postId'], 'spoiler': i['targetTitle']}


def run_baseline(input_file, output_file):
    with open(input_file, 'r') as inp, open(output_file, 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['id', 'spoiler'])
        inputs = [json.loads(line) for line in inp]
        for idx, output in enumerate(predict(inputs)):
            writer.writerow([idx, output['spoiler']])


if __name__ == '__main__':
    args = parse_args()
    run_baseline(args.input, args.output)