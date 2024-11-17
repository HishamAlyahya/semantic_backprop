import random
from datasets import load_dataset

def get_split(split, seed=0):
    
    ds = load_dataset("openai/gsm8k", 'main')
    questions = []
    for question in ds[split]:
        question['text'] = question['question']
        question['label'] = question['answer'].split('#### ')[1]
        question['output_format'] = ''
        question['extract_answer'] = lambda x: x
        question['question_id'] = question['question']
        question['expected_output'] = question['label']
        questions.append(question)
    
    random.shuffle(questions)
    return questions[:128] if split == 'train' else questions