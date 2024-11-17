import tqdm
import asyncio
import random
try:
    import wandb
except ImportError:
    wandb = None
import os
import numpy as np

from semantic_backprop.llm import llm
from semantic_backprop.engine.solutions import Solution, arun_evaluate
from semantic_backprop.utils import get_config
from data_utils.liar import get_samples

if __name__ == "__main__":
    config = get_config()
    random.seed(config["seed"])
    np.random.seed(config["seed"])

    array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    solution = Solution(include_sibling=not config["no_include_sibling"])
    keys = list(solution.tasks.keys())
    config["seed"] = array_id

    if os.environ.get("SLURM_JOB_ID") is not None:
        wandb.init(project="LLMBP_one_variant", config=config, tags=["liar", "train"])
    print(config)


    samples = get_samples()
    random.shuffle(samples)
    samples = samples[:50]

    llm.reset(base_model=config["base_model"], opt_model=config["opt_model"])
    solutions = [solution]
    preds, score = asyncio.run(arun_evaluate(solutions[-1], samples))
    preds_list = [preds]
    scores = [score]
    print(scores[-1])

    for _ in tqdm.tqdm(range(8)):
        solution = solutions[-1]
        solution.get_gradients(
            batch=samples,
            preds=preds_list[-1],
            n=1,
            errors_per_gradient=config["errors_per_gradient"],
            gradients_per_error=None,
            include_grad=config["include_grad"],
        )
        solution.apply_gradients(config["opt_keys"])
        new_solution = solution.get_full_update()
        print(str(new_solution))
        print("\n" + "-" * 50 + "\n")
        preds, score = asyncio.run(arun_evaluate(new_solution, samples))
        print(score)
        if score > scores[-1] or config["no_select"]:
            solutions.append(new_solution)
            preds_list.append(preds)
            scores.append(score)

        if os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
            wandb.log({"score": scores[-1], "new_score": score})
