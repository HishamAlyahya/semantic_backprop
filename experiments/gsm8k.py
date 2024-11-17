import asyncio
import random
import tqdm
try:
    import wandb
except ImportError:
    wandb = None
import numpy as np

from semantic_backprop.utils import get_config
from semantic_backprop.engine.solutions import arun_evaluate, get_mlp_solution
from semantic_backprop.llm import llm
from data_utils.gsm8k import get_split


if __name__ == "__main__":
    config = get_config()
    random.seed(config["seed"])

    if wandb and config["use_wandb"]:
        wandb.init(project="SBP_gsm8k", config=config, tags=["gsm8k"])
    
    print(config)

    train_data = get_split("train")
    test_data = get_split("test")
    llm.reset(base_model=config["base_model"], opt_model=config["opt_model"])

    solutions = [get_mlp_solution(2, 2, False)]
    preds, score = asyncio.run(arun_evaluate(solutions[-1], train_data))
    preds_list = [preds]
    scores = [score]
    print(scores[-1])

    for _ in tqdm.tqdm(range(config["n_iters"])):
        solutions[-1].get_gradients(
            batch=train_data,
            preds=preds_list[-1],
            errors_per_gradient=config["errors_per_gradient"],
            include_grad=config["include_grad"],
            use_bad_samples=config["use_bad_samples"],
            target=config["target"],
        )
        solutions[-1].apply_gradients(
            use_bad_samples=config["use_bad_samples"],
            short_prompt=config["short_prompt"],
            via_feedback=config["opt_via_feedback"],
        )
        if config["full_update"]:
            new_solution = solutions[-1].get_full_update()
        else:
            new_solution = solutions[-1].sample_solutions(1)[0]
        print(str(new_solution))
        print("\n" + "-" * 50 + "\n")
        preds, score = asyncio.run(arun_evaluate(new_solution, train_data))
        print(score)
        if score > scores[-1]:
            solutions.append(new_solution)
            preds_list.append(preds)
            scores.append(score)
        if wandb and config["use_wandb"]:
            wandb.log({"score": scores[-1], "new_score": score})

    scores = []
    llm_scores = []
    for i in range(0, len(test_data), 128):
        _, score = asyncio.run(arun_evaluate(solutions[-1], test_data[i : i + 128]))
        scores.append(score * len(test_data[i : i + 128]))
    score = np.sum(scores) / len(test_data)
    print(score)

    if wandb and config["use_wandb"]:
        wandb.log({"test": score})
