import asyncio
import tqdm
import os
import numpy as np

from semantic_backprop.llm import llm
from semantic_backprop.engine.solutions import get_mlp_solution, arun_evaluate
from semantic_backprop.utils import get_config

from data_utils.bcb import get_split

if __name__ == "__main__":
    config = get_config()
    print(config)

    train_data = get_split("train")
    test_data = get_split("test")

    llm.reset(base_model=config["base_model"], opt_model=config["opt_model"])

    solutions = [get_mlp_solution(n_layers=2, layer_size=2, parse_final_output=False)]

    preds, score = asyncio.run(arun_evaluate(solutions[-1], train_data))
    preds_list = [preds]
    scores = [score]

    for i in tqdm.tqdm(range(config["n_iters"])):
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

        preds, score = asyncio.run(arun_evaluate(new_solution, train_data))
        if score > scores[-1]:
            solutions.append(new_solution)
            preds_list.append(preds)
            scores.append(score)

        print(str(new_solution))
        print("SCORE", score)
        print("\n" + "-" * 50 + "\n")

    print("SCORES", scores)
    scores = []
    for i in range(0, len(test_data), 128):
        _, score = asyncio.run(arun_evaluate(solutions[-1], test_data[i : i + 128]))
        scores.append(score * len(test_data[i : i + 128]))
    score = np.sum(scores) / len(test_data)
    print(score)


