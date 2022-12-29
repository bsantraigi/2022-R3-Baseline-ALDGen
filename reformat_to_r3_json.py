# python -u translate.py -model saved_models/dd_exp1 \
#     -src data/ijcnlp_dailydialog/test.src.txt \
#     -tgt data/ijcnlp_dailydialog/test.tgt.txt \
#     -ctx data/ijcnlp_dailydialog/test.ctx.txt \
#     -output outputs/dd_exp1_test \
#     -attn_debug -beam_size 10 -n_best 1 \
#     -batch_size 1 -verbose -gpu 0 -use_dc -1

import json
import argparse
from transformers import AutoTokenizer

def cmdline_args():
    p = argparse.ArgumentParser()
    p.add_argument('-src', '--source', required=True, help="""Source sequence (one line per sequence)""")
    # p.add_argument('-ctx', '--context', required=True, help="""Source sequence to decode (one line per sequence)""")
    p.add_argument('-tgt', '--target', required=True, help='True target sequence (optional)')
    p.add_argument('-pred', '--prediction', required=True, help="Path to output predictions file (each line will be one decoded sequence")
    p.add_argument('-jp', '--json_path', required=True, help="Path for the output json file (will be created by this script)")
    
    
    return p.parse_args()

args = cmdline_args()
print(f"Arguments: {args}")

def read_array_file(fname):
    a = []
    bad_s = 0
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if line == "":
                line = "the is" # some bad sentence
                bad_s += 1
            # if line == "":
                # break
            a.append(line)
    print(f"{bad_s} empty sentences in {fname}")
    return a

# ctx = read_array_file(args.context)
src = read_array_file(args.source)
tgt = read_array_file(args.target)
pred = read_array_file(args.prediction)

assert len(tgt) == len(src), f"LT: {len(tgt)}, LS: {len(src)}, LP: {len(pred)}"
assert len(pred) == len(tgt), f"LT: {len(tgt)}, LS: {len(src)}, LP: {len(pred)}"

#vocab_model_reference = 'facebook/blenderbot-3B'
#tokenizer = AutoTokenizer.from_pretrained(vocab_model_reference, use_fast=True, verbose=False)
#SEP = tokenizer.sep_token

collector_json = []
for s,t,p in zip(src, tgt, pred):
    collector_json.append({
        'con': s,
        'gt': t,
        'aldgen': p
    })
    
with open(args.json_path, 'w') as f:
    json.dump(collector_json, f)
