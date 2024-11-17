import random
from datasets import load_dataset

categories = ['word_sorting', 'object_counting', 'dyck_languages', 'boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa', 'formal_fallacies', 'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'logical_deduction_three_objects', 'movie_recommendation', 'multistep_arithmetic_two', 'navigate', 'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names', 'salient_translation_error_detection', 'snarks', 'sports_understanding', 'temporal_sequences', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects', 'web_of_lies', 'all']

def get_split(split, category, seed=0):
    if category == 'all':
        questions = []
        for category in categories[:-1]:
            questions += get_split(split, category)
        random.shuffle(questions)
        return questions[:20] if split == 'train' else questions
    
    ds = load_dataset("lukaemon/bbh", category)
    questions = []
    for question in ds['test']:
        question['text'] = question['input']
        question['label'] = question['target']
        question['output_format'] = ''
        question['extract_answer'] = lambda x: x
        question['question_id'] = question['input']
        question['expected_output'] = question['label']
        questions.append(question)
    
    random.shuffle(questions)
    return questions[:20] if split == 'train' else questions[20:]