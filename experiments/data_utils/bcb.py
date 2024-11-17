from functools import partial

from bigcodebench.data import get_bigcodebench
from bigcodebench.evaluate import check_correctness
from bigcodebench.sanitize import sanitize


def evaluate_and_extract(
        output,
        task,
        min_time_limit: float = 1,
        max_as_limit: int = 30*1024,
        max_data_limit: int = 30*1024,
        max_stack_limit: int = 10,
    ):
    res = check_correctness(completion_id=None, problem=task, solution=sanitize(output, task['entry_point']), max_as_limit=max_as_limit, max_data_limit=max_data_limit, min_time_limit=min_time_limit, max_stack_limit=max_stack_limit)
    return task["canonical_solution"] if res["base"][0] == "pass" else output


def get_split(split, subset="full"):
    dataset = get_bigcodebench(subset=subset)
    if split == "train":
        dataset = dict(list(dataset.items())[:50])
    elif split == "test":
        dataset = dict(list(dataset.items())[50:])
    else:
        raise ValueError("split must be 'train' or 'test'")

    questions = []
    for task_id, task in dataset.items():
        questions.append({
            'text': task["instruct_prompt"],
            'label': task["canonical_solution"],
            'expected_output': task["canonical_solution"],
            'output_format': '',
            'extract_answer': partial(evaluate_and_extract, task=task),
            'question_id': task_id
        })

    return questions
