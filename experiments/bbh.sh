#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Define categories
categories=('word_sorting' 'object_counting' 'dyck_languages' 'boolean_expressions' 'causal_judgement' 'date_understanding' 'disambiguation_qa' 'formal_fallacies' 'geometric_shapes' 'hyperbaton' 'logical_deduction_five_objects' 'logical_deduction_seven_objects' 'logical_deduction_three_objects' 'movie_recommendation' 'multistep_arithmetic_two' 'navigate' 'penguins_in_a_table' 'reasoning_about_colored_objects' 'ruin_names' 'salient_translation_error_detection' 'snarks' 'sports_understanding' 'temporal_sequences' 'tracking_shuffled_objects_five_objects' 'tracking_shuffled_objects_seven_objects' 'tracking_shuffled_objects_three_objects' 'web_of_lies')

# Run for each category
for category in "${categories[@]}"; do
    python bbh.py --bbh_cat "$category" --include_grad --use_bad_sample --full_update --target --n_iters 4 > "logs/${category}.log" 2>&1 &
    
    # Limit to 4 parallel jobs
    if [[ $(jobs -r -p | wc -l) -ge 4 ]]; then
        wait -n
    fi
done

wait

echo "All categories completed. Logs saved in logs directory."