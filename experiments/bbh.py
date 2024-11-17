import asyncio
import random
import tqdm
import os
import json
try:
    import wandb
except ImportError:
    wandb = None
import numpy as np

from semantic_backprop.llm import llm
from semantic_backprop.engine.solutions import get_mlp_solution, arun_evaluate
from semantic_backprop.utils import get_config

from data_utils.bbh import get_split, categories

if __name__ == "__main__":
    config = get_config()
    random.seed(config["seed"])
    np.random.seed(config["seed"])


    if config["bbh_cat"] not in categories:
        raise ValueError("Please specify a category from the following list: " + ", ".join(categories))
    
    print(config)
    if wandb and config["use_wandb"]:
        wandb.init(
            project="SBP_bbh",
            config=config,
            tags=["bbh"],
        )

    train_data = get_split("train", category=config["bbh_cat"])
    test_data = get_split("test", category=config["bbh_cat"])

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
    for i in range(0, len(test_data), 128):
        _, score = asyncio.run(arun_evaluate(solutions[-1], test_data[i : i + 128]))
        scores.append(score * len(test_data[i : i + 128]))
    score = np.sum(scores) / len(test_data)
    total_questions = len(get_split("test", "all"))
    weight = len(test_data) / total_questions * 27
    print("SCORE", score)
    print("WEIGHTED", score * weight)

    if wandb and config["use_wandb"]:
        wandb.log({"test": score, "weighted score": score * weight})
    
    if not os.path.exists(config["results_dir"]):
        os.makedirs(config["results_dir"], exist_ok=True)
    with open(os.path.join(config["results_dir"], f"{config['bbh_cat']}.json"), "w") as f:
        result = {
            "score": score,
            "weighted_score": score * weight,
            "test_questions": len(test_data),
            "total_questions": total_questions,
            "category": config["bbh_cat"],
            "solution": str(solutions[-1]),
        }
        json.dump(result, f)
