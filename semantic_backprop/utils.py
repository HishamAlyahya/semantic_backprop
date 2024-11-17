import time
import argparse
import os

def add_indent(s, n_indents=1):
    return '\n'.join(['\t' * n_indents + line for line in s.split('\n')])

def listing(strs):
    return '\n'.join([f'{i+1}. {s}' for i, s in enumerate(strs)])

def parse_tagged_text(text, start_tag, end_tag):
    """ Parse text that is tagged with start and end tags."""
    texts = []
    while True:
        start_index = text.find(start_tag)
        if start_index == -1:
            break
        end_index = text.find(end_tag, start_index)
        if end_index == -1:
            break
        start_index += len(start_tag)
        texts.append(text[start_index:end_index].strip())
        text = text[end_index+len(end_tag):]
    if len(texts) == 0:
        return [text]
    return texts

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='gpt-4om')
    parser.add_argument('--opt_model', default='gpt-4t')
    parser.add_argument('--time_base', default=600, type=int)
    parser.add_argument('--time_opt', default=600, type=int)
    parser.add_argument('--max_threads', default=64, type=int)
    parser.add_argument('--temperature', default=0.0, type=float)
    parser.add_argument('--errors_per_gradient', default=2, type=int)
    parser.add_argument('--gradients_per_error', default=1, type=int)
    parser.add_argument('--include_grad', action='store_true')
    parser.add_argument('--full_update', action='store_true')
    parser.add_argument('--include_final', action='store_false')
    parser.add_argument('--use_bad_samples', action='store_true')
    parser.add_argument('--short_prompt', action='store_true')
    parser.add_argument('--exp_num', default=0, type=int)    
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--n_iters', default=12, type=int)
    
    parser.add_argument('--bbh_cat', default="", type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--target', action='store_true')    
    parser.add_argument('--across_layers', action='store_true')
    parser.add_argument('--category', default='all')
    parser.add_argument('--opt_via_feedback', action='store_true')
    parser.add_argument('--no_select', action='store_true')
    parser.add_argument('--opt_keys', default=None, type=str)
    parser.add_argument('--no_include_sibling', action='store_true')
    parser.add_argument('--use_wandb', action='store_true') 
    parser.add_argument('--results_dir', default='./results')

    args, _ = parser.parse_known_args()
    # args.out = f"./results/{args.include_final}.{args.include_grad}.{args.use_test}.{args.exp_num}"

    config = vars(args)
    config['opt_keys'] = config['opt_keys'].split(',') if config['opt_keys'] else None

    config['date'] = time.strftime("%Y-%m-%d", time.localtime())
    config['hour'] = time.strftime("%H", time.localtime())
    config['minute'] = time.strftime("%M", time.localtime())
    config['seed'] = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
 
    return config
