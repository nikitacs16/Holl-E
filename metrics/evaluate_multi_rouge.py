from __future__ import print_function

import pandas as pd
from rouge import *
import pickle
from collections import Counter
import string
import re
import argparse
import json
import sys
from nltk.tokenize import word_tokenize
reload(sys)
sys.setdefaultencoding("utf-8")

#Please convert all your results to a csv with three coloumns: example_id, decoded, reference
#decoded is the prediction
#reference is the ground truth

fname = 'gttp_oracle.csv' #input your results file
data = pd.read_csv(fname)
predictions = data['decoded'] 
example_id = data['example_id']
ground_truth = data['reference']
also_ground_truths = json.load(open('multi_reference_test.json')) #json file given with the repo

rouge_1 = 0
rouge_2 = 0
rouge_l = 0
count = 0

def compute_single(p,g):
	w = rouge([str(p)],[str(g)])
	r1 = w['rouge_1/f_score']
	r2 = w['rouge_2/f_score']
	rl = w['rouge_l/f_score']
	return r1,r2,rl


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(word_tokenize(text))

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(lower(str(s)))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def max_over_metric(prediction, ground_truths):
	max_r1 = 0
	max_r2 = 0
	max_rl = 0

	for g in ground_truths:
		r1,r2,rl = compute_single(prediction,normalize_answer(g))
		if rl > max_rl:
			max_rl= rl
			max_r1 = r1
			max_r2 = r2

	return max_r1, max_r2, max_rl

f1 = 0
for p,g ,e in zip(predictions,ground_truth,example_id):
	other_g = also_ground_truths[e]['responses'] #change to 'spans' when evaluating BiDAF
	full_g = other_g + [g]
	count = count + 1
	r1, r2, rl = max_over_metric(p,full_g)
	rouge_1 +=r1
	rouge_2 +=r2
	rouge_l +=rl
	#f1 += metric_max_over_ground_truths(f1_score, p, full_g) #Use only for BiDAF




print("%.2f\t%.2f\t%.2f"%(rouge_1*100/count,rouge_2*100/count, rouge_l*100/count))
#print('%.2f'%(f1*100/count)) #Use for BiDAF
