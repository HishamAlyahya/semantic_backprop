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
from semantic_backprop.engine.solutions import LiarSolution, arun_evaluate
from semantic_backprop.utils import get_config
from data_utils.liar import get_samples

if __name__ == "__main__":
    config = get_config()
    random.seed(config["seed"])
    np.random.seed(config["seed"])

    solution = LiarSolution(include_sibling=not config["no_include_sibling"])

    if wandb and config["use_wandb"]:
        wandb.init(project="SBP_liar", config=config, tags=["liar", "train"])

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
    for _ in tqdm.tqdm(range(config["n_iters"])):
        solution = solutions[-1]
        solution.get_gradients(
            batch=samples,
            preds=preds_list[-1],
            n=1,
            errors_per_gradient=config["errors_per_gradient"],
            gradients_per_error=None,
            include_grad=config["include_grad"],
        )
        solution.apply_gradients(opt_keys=config["opt_keys"])
        new_solution = solution.get_full_update()
        print(str(new_solution))
        print("\n" + "-" * 50 + "\n")
        preds, score = asyncio.run(arun_evaluate(new_solution, samples))
        print(score)
        if score > scores[-1] or config["no_select"]:
            solutions.append(new_solution)
            preds_list.append(preds)
            scores.append(score)

        if wandb and config["use_wandb"]:
            wandb.log({"score": scores[-1], "new_score": score})

        print("Scores", scores)
        print("Best score", scores[-1])
