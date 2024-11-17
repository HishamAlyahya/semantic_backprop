# SBP
python liar.py --n_iters 8 --include_grad

# No gradient
python liar.py --n_iters 8

# No neighborhood gradients
python liar.py --n_iters 8 --include_grad --no_include_sibling

# One-instruction variants
mkdir -p liar_logs

for key in Statement Party "Job title" Source State Final; do
    echo "Running liar.py with --opt_keys $key"
    python liar.py --n_iters 8 --include_grad --opt_keys "$key" > "liar_logs/$key.txt" 2>&1 &
done
wait

# No Validation
python liar.py --n_iters 8 --include_grad --no_select