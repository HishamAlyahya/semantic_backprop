
from typing import Literal, Union
from datasets import load_dataset

def get_dataset():
    dataset = load_dataset("liar")['train']
    for row in dataset:
        if row['label'] in {0, 3}: # true, barely true
            ex = {
                'label': 1 if row['label'] == 0 else 0,
                'text': f'Statement: {row["statement"]}\nJob title: {row["job_title"]}\nState: {row["state_info"]}\nParty: {row["party_affiliation"]}\nContext: {row["context"]}'
            }
            yield ex

def is_yes(response):
    return 1 if response.strip().upper().startswith('YES') else 0

def get_samples():
        exs = []
        for i, row in enumerate(get_dataset()):
            sample = {'id': f'train-{i}', 'label': row['label'], 'text': row['text'], 'question_id': row['text']}
            include_sample = True
            for line in row['text'].split('\n'):
                key = line.split(':')[0]
                content = line.split(':')[1][1:]
                sample[key] = content
                if ''.join(filter(str.isalpha, content)) in ['', 'none']:
                    include_sample = False 

            if include_sample:
                sample['Source'] = sample['Context']
                sample['text'] = sample['text'].replace('Context', 'Source')
                sample['output_format'] = 'Answer Yes or No as labels'
                sample['extract_answer'] = is_yes
                sample['expected_output'] = 'Yes' if sample['label'] == 1 else 'No'
                exs.append(sample)
        return exs