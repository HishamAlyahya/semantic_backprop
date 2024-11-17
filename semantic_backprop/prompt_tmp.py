import semantic_backprop.utils as utils

fdbk_final = """The answer should be {desire}."""

def fdbk_prompt_no_sibling(hint=None, answer=None, desire=None, idx=None):
    return f"""A task is performed given a context and some hints.
One of the hints is:
{hint}

Answered: {answer}

However, the desired answer is {desire}.

How the hint needs to be changed to get the desired output? Respond one line."""

def fdbk_prompt(final_task=None, context=None, hints=None, answer=None, desire=None, idx=None): 
    return f"""A task is performed given a context and some hints
Task: 
{final_task}

Context: 
{context}

Hints:
{hints}

Answered: {answer}

However, the desired answer is {desire}.

How each hint needs to be changed to get the desired output? Respond one line per hint. Start with "Hint x" for the xth line.
"""

def example_str(i, input, output, fdbk): 
    return f"""## Example {i}
Input:
{utils.add_indent(input)}

My output: 
{utils.add_indent(output)}

Feedback received on my output: 
{utils.add_indent(fdbk)}
"""
#may change to explicitly say fdbk is a feedback on my output.

def example_str_no_grad(i, input, output): 
    return f"""## Example {i}
Input:
{utils.add_indent(input)}

My output: 
{utils.add_indent(output)}
"""

def meta_prefix(task, examples, bad_examples):
    prompt = f"""I'm trying to write a task-specific question answering assistant.

My current prompt is:
"{task}"
"""

    if bad_examples:
        prompt += f"""\nHere are some examples that it did not answer well:
{utils.add_indent(examples)}
"""
    else:
        prompt += f"""\nHere are some examples:
{utils.add_indent(examples)}
"""
    return prompt

def opt_prompt(task, examples, bad_examples=True, short_prompt=False):
    prompt = meta_prefix(task, examples, bad_examples)
    if short_prompt:
        short_prompt = " in no more than three sentences"

    prompt += f"""\nBased on the above examples, write an improved prompt.
Show your reasoning steps.
Do not include the keyword "feedback" or any example-specific content in the prompt.
Finish with the improved prompt wrapped by <prompt> and </prompt>{short_prompt}.
"""
    return prompt

def prompt_feedback(task, examples, bad_examples):
    prompt = meta_prefix(task, examples, bad_examples)
    prompt += f'\nBased on the above examples, analyze the pros and cons of the prompt.'
    return prompt

def opt_prompt_via_feedback(task, feedback, short_prompt=False):
    if short_prompt:
        short_prompt = " in no more than three sentences"

    prompt = f"""I'm trying to write a task-specific question answering assistant.

My current prompt is:
"{task}"
"""
    
    prompt += f"""\nComments on the prompt:
{utils.add_indent(feedback)}

Based on the comments, write an improved prompt.
Show your reasoning steps.
Do not include the keyword "comments" or "feedback" in the prompt.
Finish with the improved prompt wrapped by <prompt> and </prompt>{short_prompt}.
"""
    return prompt
