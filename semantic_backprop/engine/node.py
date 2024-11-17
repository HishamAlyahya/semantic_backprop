import random
import asyncio
import multiprocessing
from time import time
from typing import Dict

import semantic_backprop.prompt_tmp as prompt_tmp
import semantic_backprop.utils as utils

from semantic_backprop.llm import llm
from semantic_backprop.engine import Instruction, Question, Statement

lock = multiprocessing.Lock()

class Node:
    def __init__(self, instruction: Instruction|None=None, is_final=False, parse_output_statement=True):
        if instruction is None:
            instruction = Instruction()
        self.instruction = instruction
        self.is_final = is_final
        self.parse_output_statement = parse_output_statement
        self.successors = []
        self.predecessors = []
        self.candidates = []
        self.cache = {}
        self.history = {}

    def add_successor(self, child: 'Node'):
        self.successors.append(child)
        child.predecessors.append(self)
    
    def __str__(self):
        return self.instruction.instruciton_str

    def clear_history(self):
        self.cache: Dict[str, Statement] = {}

    async def __call__(self, question: Question):
        start_time = time()
        while True:
            pred_statements = []
            for pred in self.predecessors:
                if question.id in pred.cache:
                    pred_statements.append(pred.cache[question.id])
            if len(pred_statements) == len(self.predecessors):
                break
            # if time() - start_time > utils.params['time_base']:
            #     exit(utils.params['time_base'])
            await asyncio.sleep(.1)

        statement = Statement(question, pred_statements, self.instruction, self.is_final, len(self.successors), self)
        await statement.forward(parse_output_statement=self.parse_output_statement)
        # with lock:
        self.cache[question.id] = statement

    def prepare_examples(self, error_samples):
        self.example_str = ''
        for i, sample in enumerate(error_samples):
            question = sample['text']
            statement = self.cache[question]
            if len(statement.filtered_feedback) == 0:
                self.example_str += prompt_tmp.example_str_no_grad(i=i+1, input=statement.input, output=statement.output)
            else:
                self.example_str += prompt_tmp.example_str(i=i+1, input=statement.input, output=statement.output, fdbk=statement.fdbk)

    async def generage_candidates(self, use_bad_samples=True, short_prompt=False, via_feedback=False):
        if via_feedback:
            feedback_gen_prompt = prompt_tmp.prompt_feedback(self.instruction.instruciton_str, self.example_str, use_bad_samples)
            prompt_feedback = (await llm.chat(feedback_gen_prompt, is_opt=True))[0]
            candidate_gen_prompt = prompt_tmp.opt_prompt_via_feedback(self.instruction.instruciton_str, prompt_feedback, short_prompt)
        else:
            candidate_gen_prompt = prompt_tmp.opt_prompt(self.instruction.instruciton_str, self.example_str, use_bad_samples, short_prompt)
        res = (await llm.chat(candidate_gen_prompt, is_opt=True))[0]
        instruction = utils.parse_tagged_text(res, "<prompt>", "</prompt>")[0]
        self.candidates = [Instruction(instruction)]

    def choose_from_candidates(self):
        self.instruction = random.choice(self.candidates + [self.instruction])        
        self.candidates = []
