import random
import asyncio

from functools import total_ordering
from collections import defaultdict
from copy import deepcopy
from typing import List
import numpy as np

import semantic_backprop.prompt_tmp as prompt_tmp
import semantic_backprop.utils as utils
from semantic_backprop.engine import Question, Instruction, Statement, Node
from semantic_backprop.llm import llm


async def aprocess_example(ex, solution):
    pred = await solution.ainference(ex)
    return ex, pred

async def arun_evaluate(solution, test_exs):
    try:
        results = {}
        tasks = [aprocess_example(ex, solution) for ex in test_exs]
        from tqdm.asyncio import tqdm
        for task in tqdm.as_completed(tasks):
            ex, pred = await task
            results[ex['question_id']] = pred 
        return [results[ex['question_id']] for ex in test_exs], np.mean([ex['label'] == results[ex['question_id']] for ex in test_exs])
    except Exception as e:
        print(e)
        return [''] * len(test_exs), 0

def sample_error_batch(batch, preds, n=4):
    """ Sample n error strings from the given texts, labels, and preds"""
    error_batch = [(sample, p) for sample, p in zip(batch, preds) if sample['label'] != p]
    error_batch = random.sample(error_batch, min(len(error_batch), n))
    return error_batch

def get_mlp_solution(n_layers, layer_size, across_layer=False, parse_final_output=True):
    solution = GraphSolution()
    for i in range(n_layers):
        if i == n_layers - 1:
            layer_nodes = [Node(Instruction('Solve the problem'), is_final=True, parse_output_statement=parse_final_output)]
        else:
            layer_nodes = [Node() for _ in range(layer_size)]
        for node in layer_nodes:
            for pred in solution.nodes[-layer_size:]:
                pred.add_successor(node)
            if across_layer:
                for pred in solution.nodes[:-layer_size]:
                    pred.add_successor(node)
        for node in layer_nodes:
            solution.add_node(node)
    
    return solution


class GraphSolution:
    @total_ordering
    def __eq__(self, other):
        return str(self) == str(other)
    
    def __lt__(self, other):
        return str(self) < str(other)
    
    def __hash__(self):
        return hash(str(self))
    
    def __init__(self):
        self.nodes: List[Node] = []
        self.history = {}

    def clear_history(self):
        for node in self.nodes:
            node.clear_history()

    @property
    def output_node(self):
        return self.nodes[-1]   
    

    def add_node(self, node: Node):
        self.nodes.append(node)

    def __str__(self):
        return ('\n' + '.' * 20 + '\n').join([str(node) for node in self.nodes])

    async def __call__(self, question: Question) -> Statement:
        for node in self.nodes:
            node.cache.pop(question.id, None)
        tasks = [node(question) for node in self.nodes]
        await asyncio.gather(*tasks)
        return self.output_node.cache[question.id]
    
    async def add_feedback(self, sample, target=True):
        output_statement = self.output_node.cache[sample['text']]
        answer = sample.get('expected_output', sample['label'])
        feedback = f'The expected output is: "{answer}".'
        output_statement.externel_feedback.append(feedback)
        tasks = [node.cache[sample['text']].backward(target=target) for node in self.nodes]
        await asyncio.gather(*tasks)

    async def ainference(self, sample):
        question = Question(sample['text'], answer=sample['label'], output_format=sample['output_format'])
        pred_statement = (await self(question))
        extract_answer = sample.get('extract_answer', lambda x: x)
        pred = extract_answer(pred_statement.statement_str)
        return pred
    
    def inference(self, sample, return_hist=False):
        question = Question(sample['text'])
        pred = asyncio.run(self(question)).statement_str
        pred = sample['extract_answer'](pred)
        if return_hist:
            return pred, (sample['text'], '')
        return pred

    def get_gradients(self, batch, preds, errors_per_gradient, include_grad=True, use_bad_samples=True, target=True):
        #cannot BP on sample more than one time
        async def aapply():
            if use_bad_samples:
                samples_pres = sample_error_batch(batch, preds, errors_per_gradient)
                samples = [sample for sample, _ in samples_pres]
            else:
                samples = random.sample(batch, errors_per_gradient)
            if include_grad:
                tasks = [self.add_feedback(sample, target=target) for sample in samples]
                await asyncio.gather(*tasks)
            
            for node in self.nodes:
                node.prepare_examples(samples)
        asyncio.run(aapply())
        
    def apply_gradients(self, include_final=True, use_bad_samples=True, short_prompt=False, via_feedback=False):
        async def aapply(): 
            tasks = []
            for node in self.nodes:
                if not (node is self.output_node) or include_final:
                    tasks.append(node.generage_candidates(use_bad_samples, short_prompt, via_feedback))
            await asyncio.gather(*tasks)
        asyncio.run(aapply())
        
    def sample_solutions(self, n):
        solutions = []
        for _ in range(n):
            solution = deepcopy(self)
            for node in solution.nodes:
                node.choose_from_candidates()
            solutions.append(solution)
        return solutions
    
    def get_full_update(self):
        solution = deepcopy(self)
        for node in solution.nodes:
            node.instruction = (node.candidates + [node.instruction])[0]
        return solution


class LiarSolution:
    @staticmethod
    def template_context(task, problem):
        return  """# Task
{task_content}

# Output format
Answer in no more than two sentences.

# Context
{text}

#Answer""".format(task_content=task, text=problem['text'])

    def template_final(self, task, problem, inputs):
        return """# Task
{task_content}

# Output format
Answer Yes or No as labels

# Context
{text}

# Hints 
{input}

# Answer""".format(task_content=task, text=problem['text'], input=inputs)

    @total_ordering
    def __eq__(self, other):
        return str(self) == str(other)
    
    def __lt__(self, other):
        return str(self) < str(other)
    
    def __hash__(self):
        return hash(str(self))

    def __init__(self, tasks=None, final_field='Final', example=None, include_sibling=True):
        if tasks is None:
            tasks = {'Statement': 'Analyse the statement itself.',
                    'Party': 'How does the party feel about the statement?', 
                    'Job title': 'Is the statement consistent with the job title?',
                    'Source': 'What might be the reason that this source released the statement?',
                    'State': 'How does the state feel about the statement?',
                    'Final': 'Determine whether the Statement is a lie (Yes) or not (No) based on the Context and other information.'}
        self.tasks = tasks
        self.final_field = final_field
        self.history = {}
        self.example = example
        self.include_sibling = include_sibling

    def __str__(self):
        return ('\n' + '.' * 20 + '\n').join(self.tasks.values())

    def clear_history(self):
        self.history = {}

    async def ainference(self, sample):
        hist = {}
        atasks = []
        prompts = []
        for task in self.tasks.keys():
            if task == self.final_field:
                continue
            prompt = self.template_context(task=self.tasks[task], problem=sample)
            atasks.append(llm.chat(prompt))
            prompts.append(prompt)

        results = await asyncio.gather(*atasks)
        for task, prompt, result in zip(self.tasks.keys(), prompts, results):
            hist[task] = (prompt, result[0])
        hist['hints'] = utils.listing([v[1] for v in hist.values()])
        prompt = self.template_final(task=self.tasks[self.final_field], problem=sample, inputs=hist['hints'])
        response = (await llm.chat(prompt))[0]
        hist[self.final_field] = (prompt, response)
        pred = 1 if response.strip().upper().startswith('YES') else 0
        self.history[sample['text']] = hist
        return pred
    
    def _sample_error_batch(self, batch, preds, n=4):
        """ Sample n error strings from the given texts, labels, and preds"""
        error_batch = [(sample, p) for sample, p in zip(batch, preds) if sample['label'] != p]
        error_batch = random.sample(error_batch, min(len(error_batch), n))
        return error_batch

    def back_prop(self, sample):
        record = self.history[sample['text']]
        _, final_answer = record[self.final_field]
        desire = 'Yes' if sample['label'] == 1 else 'No'
        fdbks = defaultdict(str)
        fdbks[self.final_field] = prompt_tmp.fdbk_final.format(desire=desire)
        if self.include_sibling:
            fdbk_prompt = prompt_tmp.fdbk_prompt(final_task=self.tasks[self.final_field], 
                                                    context=sample['text'], 
                                                    hints=record['hints'], 
                                                    answer=final_answer,
                                                    desire=desire)
            fdbk = asyncio.run(llm.chat(fdbk_prompt, is_opt=True))[0]
            for i, key in enumerate(list(self.tasks.keys())[:-1]):
                for line in fdbk.split('\n'):
                    if line.startswith(f'Hint {i+1}'):
                        fdbks[key] = line[7:]
        else:
            async def aget_fdbk():
                fdbks = {}
                atasks = []
                for key in list(self.tasks.keys())[:-1]:
                    fdbk_prompt = prompt_tmp.fdbk_prompt_no_sibling(record[key][1], final_answer, desire)
                    atasks.append(llm.chat(fdbk_prompt, is_opt=True))
                results = await asyncio.gather(*atasks)
                for key, result in zip(list(self.tasks.keys())[:-1], results):
                    fdbks[key] = result[0]
                return fdbks
            fdbks.update(asyncio.run(aget_fdbk()))
        return fdbks

    def get_gradient(self, error_batch, include_grad):
        exps = defaultdict(str)
        for i, (sample, _) in enumerate(error_batch):
            if include_grad:
                fdbk = self.back_prop(sample)
            record = self.history[sample['text']]
            for key in self.tasks.keys():
                if key == self.final_field:
                    input = '\n'.join(record[key][0].split('\n')[-15:-2])
                else:
                    input = '\n'.join(record[key][0].split('\n')[-7:-2])
                if include_grad:
                    exps[key] += prompt_tmp.example_str(i+1, input, record[key][1], fdbk[key])
                else:
                    exps[key] += prompt_tmp.example_str_no_grad(i+1, input, record[key][1])
        self.grads.append(exps)

    def get_gradients(self, batch, preds, n, errors_per_gradient, gradients_per_error, include_grad=True):
        self.grads = []
        for _ in range(n):
            error_batch = sample_error_batch(batch, preds, errors_per_gradient)
            self.get_gradient(error_batch, include_grad=include_grad)

    async def apply_task_gradient(self, feedback, task):
        """ Incorporate feedback gradient into a task."""
        transformation_prompt = prompt_tmp.opt_prompt(task, feedback)
        res = await llm.chat(transformation_prompt, is_opt=True)
        new_prompts = []
        for r in res:   
            new_prompts += utils.parse_tagged_text(r, "<prompt>", "</prompt>")
        return new_prompts

    def apply_gradients(self, opt_keys=None):
        if opt_keys is None:
            opt_keys = self.tasks.keys()
        async def aapply_task_gradient(opt_keys):
            self.candidates = {}
            for key in self.tasks.keys():
                self.candidates[key] = [self.tasks[key]]
            for grads in self.grads:
                atasks = []
                opted_keys = []
                for key, grad in grads.items():
                    if key in opt_keys:
                        atasks.append(self.apply_task_gradient(grad, self.tasks[key]))
                        opted_keys.append(key)
                for key, new_tasks in zip(opted_keys, await asyncio.gather(*atasks)):
                    self.candidates[key] += new_tasks
        asyncio.run(aapply_task_gradient(opt_keys))
    
    def get_full_update(self):
        """ Get a full update of the solution."""
        new_tasks = {}
        for key, values in self.candidates.items():
            new_tasks[key] = values[-1]
        return self.__class__(tasks=new_tasks, final_field=self.final_field)

    def sample_solutions(self, n):
        """ Sample n solutions from the candidates."""
        if n == 1:
            return [self.get_full_update()]
        solutions = []
        for _ in range(n):
            prompts = {}
            for key, values in self.candidates.items():
                prompts[key] = random.choice(values)
            solutions.append(self.__class__(tasks=prompts, final_field=self.final_field))
        return solutions
