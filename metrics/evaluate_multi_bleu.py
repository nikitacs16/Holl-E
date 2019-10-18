import pandas as pd
from rouge import *
from bleu import * 
import pickle
import time
import os
import sys
from nltk.tokenize import word_tokenize
reload(sys)
sys.setdefaultencoding("utf-8")


#Please convert all your results to a csv with three coloumns: example_id, decoded, reference
#decoded is the prediction
#reference is the ground truth


fname = 'gttp_oracle.csv' #input results file
data = pd.read_csv(fname)
predictions = data['decoded']
example_id = data['example_id']
ground_truth = data['reference']
also_ground_truths = json.load(open('multi_reference_test.json')) #json file given with the repo
#x = moses_multi_bleu(predictions,ground_truth)

count = 0
bleu = 0

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(word_tokenize(text))

    def lower(text):
        return text.lower()

    return white_space_fix(lower(str(s)))

def bleu_score(prediction,ground_truth):
	if len(str(prediction)) ==  0 or len(str(ground_truth))==0:
		return 0.0
	return moses_multi_bleu([str(prediction)],[str(ground_truth)])
	

def max_over_metric(prediction, ground_truths):
	bl = 0
	max_bl = 0
	for g in ground_truths:
		bl = bleu_score(prediction,normalize_answer(g))
		if bl > max_bl:
			max_bl = bl
		
	return max_bl

bl_fails = []
for p,g,e in zip(predictions,ground_truth,example_id):
	other_g = also_ground_truths[e]['responses'] #Please change to spans for BiDAF
	full_g = other_g + [g]
	try:
		bl = max_over_metric(p,full_g)
		time.sleep(0.001)
	except:
		bl_fails.append(example_id) #this rarely occurs. If an example is not evaluated because of moses error, please re-evaluate that example and add to the final sumed up bleu score
		count = count - 1
		bl = 0	
	bleu += bl
	if count%100 == 0:
		print(count)#code is very slow
		os.system('rm /tmp/tmp*') #workaround to avoid system loading

print(bleu)
print(bleu/count)
print('Unable to evaluate for :')
print(bl_fails)