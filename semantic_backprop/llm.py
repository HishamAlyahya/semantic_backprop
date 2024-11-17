import numpy as np
import asyncio
import os

from copy import deepcopy
from openai import AsyncOpenAI

openAI_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
port = str(2776 + int(os.getenv('SLURM_ARRAY_TASK_ID', 0)))
local_client = AsyncOpenAI(
    base_url=f"http://localhost:{port}/v1",
    api_key="key1",
)
models_key = ['gpt-3.5', 'gpt-4', 'gpt-4t', 'gpt-4o', 'gpt-4om', 
              'mistral', 'mistral-q',
              'llama', 'llama-q', 'llama405', 
              'qwen', 'phi', 'nemotron']
models = ['gpt-3.5-turbo-0125', 'gpt-4-0125-preview', 'gpt-4-turbo-2024-04-09', 'gpt-4o-2024-05-13', 'gpt-4o-mini-2024-07-18', 
          'mistralai/Mistral-Large-Instruct-2407', 'qeternity/Mistral-Large-Instruct-2407-w8a8',
          'meta-llama/Meta-Llama-3.1-70B-Instruct', 'neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8', 'neuralmagic/Meta-Llama-3.1-405B-Instruct-quantized.w4a16',
          'Qwen/Qwen2.5-72B-Instruct', 'microsoft/Phi-3-medium-128k-instruct', 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF']
timeout = [32, 512, 256, 128, 64, 1, 1, 1, 1]
models_dict = dict(zip(models_key, models))
timeout_dict = dict(zip(models_key, timeout))

class LLM:
    def __init__(self, use_cache=True, base_model='llama', opt_model=None,):
        self.reset(use_cache, base_model, opt_model)

    def reset(self, use_cache=True, base_model='llama', opt_model=None,):
        self.cache = {}
        self.use_cache = use_cache
        self.base_model = base_model
        self.opt_model = opt_model
        if opt_model is None or opt_model=='base':
            self.opt_model = base_model
            if base_model in ['gpt-3.5', 'gpt-4om', 'phi']:
                self.opt_model = 'gpt-4t'
    async def chat(self, messages, temperature=0, n=1, top_p=1, max_tokens=1024, is_opt=False):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        model_key = self.opt_model if is_opt else self.base_model
        model = models_dict.get(model_key, model_key)
        timeout = timeout_dict.get(model_key, 1)
        if 'gpt' in model:
            client = openAI_client
        else:
            client = local_client
        id = str(messages) +  model
        if id in self.cache and temperature == 0 and self.use_cache:
            return deepcopy(self.cache[id])
        retries = 0
        while True:
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    n=n,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    timeout= np.random.exponential(timeout) * min(10, retries) + timeout if 'gpt' in model else 10000
                )
                break
            except Exception as e:
                print(e)
                retries += 1
                await asyncio.sleep(np.random.exponential(3))
        self.cache[id] = [choice.message.content for choice in r.choices]
        return deepcopy(self.cache[id])
        
llm = LLM(base_model='gpt-4om', opt_model='gpt-4t')
