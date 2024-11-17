import asyncio

from time import time

from semantic_backprop.engine import Question, Instruction
from semantic_backprop.llm import llm
import semantic_backprop.utils as utils

class Statement:
    def __init__(self, question: Question, pred_statements: list['Statement'], instruction: Instruction, is_final:bool, num_successors=0, node=None):
        self.pred_statements = pred_statements
        self.instruction = instruction
        self.node = node
        self.question = question
        self.is_final = is_final
        self.num_successors = num_successors
        self.feedback = []
        self.externel_feedback = []
        self.filtered_feedback = []
        self.had_feedback = False

    @property
    def input(self):
        return self.prompt[:self.prompt.find('\nTask:')]
    
    @property
    def output(self):
        return self.statement_str
    
    @property
    def fdbk(self):
        return utils.listing(self.filtered_feedback)
    
    async def forward(self, parse_output_statement=True):
        prompt = \
f"""Context:
{utils.add_indent(self.question.question_str)}
"""
        if len(self.pred_statements) != 0:
            prompt += \
f"""
Consider the following hints:
{utils.add_indent(utils.listing([statement.statement_str for statement in self.pred_statements]))}
"""
        prompt += \
f"""
Task:
{utils.add_indent(self.instruction.instruciton_str)}

Show your reasoning steps. 
"""
        if parse_output_statement:
            prompt += \
f"""
Finish with an output statement wrapped by <output statement> and </output statement>.
"""

        if self.is_final:
            prompt += self.question.output_format
        response = (await llm.chat(prompt))[0]
        statement = utils.parse_tagged_text(response, '<output statement>', '</output statement>')[0]
        

        self.prompt = prompt
        self.response = response
        self.statement_str = statement if parse_output_statement else response
        
    async def backward(self, target=True):
        if self.had_feedback:
            return 
        def feedback_filter(feedback):
            return [fdbk for fdbk in feedback if fdbk is not None and fdbk != '']

        start_time = time()
        while True:
            if len(self.feedback) == self.num_successors:
                break
            # if time() - start_time > utils.params['time_opt']:
            #     exit(utils.params['time_opt'])
            else:
                await asyncio.sleep(.1)
                
        feedback = self.feedback + self.externel_feedback
        feedback = feedback_filter(feedback)
        self.filtered_feedback = feedback
        
        if len(self.pred_statements) == 0:
            return
        # prompt the question, statements, response, and feedback, give feedback to statements
        if target:
            prompt = \
f"""A task is performed given a question and some hints.

Task:
{utils.add_indent(self.instruction.instruciton_str)}

Question:
{utils.add_indent(self.question.question_str)}

Hints:
{utils.add_indent(utils.listing([statement.statement_str for statement in self.pred_statements]))}

Output attempt in response to the task:
{utils.add_indent(self.statement_str)}

Feedback on the output:
{utils.add_indent(utils.listing(feedback))}

Based on the feedback, how each hint should to be changed? 
Respond one line per hint. Start with "Hint x" for the xth line.
"""    
#Based one the feedback, how each hint should to be changed? Respond one line per hint. Start with "Hint x" for the xth line.

            if len(feedback) == 0:
                pred_feedbacks_str = ''
            else:
                pred_feedbacks_str = (await llm.chat(prompt, is_opt=True))[0]
        else:
            prompt = \
f"""A task is performed given a question and some hints.

Task:
{utils.add_indent(self.instruction.instruciton_str)}

Question:
{utils.add_indent(self.question.question_str)}

Hints:
{utils.add_indent(utils.listing([statement.statement_str for statement in self.pred_statements]))}

Output attempt in response to the task:
{utils.add_indent(self.statement_str)}

Feedback on the output:
{utils.add_indent(utils.listing(feedback))}

Based one the feedback, comment on the quality of each hint without mentioning other hints. 
Respond one line per hint. Start with "Hint x: This hint" for the xth line.
"""
            system_prompt = """You will be commenting on some hints based on their contribution to derive an output. 
When commenting a hint, do not refer to other hints. 
As a negative example, the following comments are prohibited as the second to fourth comments refer to the first hint:

Hint 1: This hint directly points to the correct answer, making it very useful for solving the problem.
Hint 2: This hint repeats the first, which does not add additional value or new information.
Hint 3: This hint is redundant as it repeats the first hint, providing no further clarification or assistance.
Hint 4: This hint again repeats the same information, contributing no new insights or help beyond the first hint.

As a positive example, the following comments are allowed:
Hint 1: This hint directly points to the correct answer, making it very useful for solving the problem.
Hint 2: This hint provides a new perspective on the problem, offering a different approach to the solution.
Hint 3: This hint offers additional information, providing further insights into the problem.
Hint 4: This hint is misleading, leading to an incorrect solution. It should be revised to avoid confusion.
"""
            messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}]
            if len(feedback) == 0:
                pred_feedbacks_str = ''
            else:
                pred_feedbacks_str = (await llm.chat(messages, is_opt=True))[0]        
#may change above prompt to directly ask for feedback on each hint
        0
        for i, statement in enumerate(self.pred_statements):
            if target:
                temp = pred_feedbacks_str[pred_feedbacks_str.find(f'Hint {i+1}'):]
                end = temp.find('\n')
                statement.feedback.append(temp[7:end])
            else:
                temp = pred_feedbacks_str[pred_feedbacks_str.find(f'Hint {i+1}: This hint'):]
                end = temp.find('\n')
                statement.feedback.append('The output' + temp[17:end])
        self.had_feedback = True
