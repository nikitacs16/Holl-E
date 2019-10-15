import json
from token_utils import *
import random
from random import randint
import numpy as np
from tqdm import *
import re
random.seed(123)
from bisect import bisect_left


max_response_length = 80 #token level
oracle_length = 256
full_length = 1200

alternate_responses_only = True

'''
Deprecated
'''
full_stop_pattern = re.compile(r"[\.|\.\.\.|!|\?|\,|:][a-z]",re.IGNORECASE)
bracket_pattern = re.compile(r"[a-z]\(",re.IGNORECASE)
continuation_pattern_dot = '\.(\.)*\.'
continuation_pattern_question = '\?(\?)*\.'
continuation_pattern_interjection = '!(!)*!'

def rolling_window(a, window): #deprecated
	shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
	strides = a.strides + (a.strides[-1],)
	return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
		   
def find_first_numpy(a, b): #deprecated
	a = np.array([i.lower() for i in a])
	b = np.array([i.lower() for i in b])

	temp = rolling_window(a, len(b))
	result = np.where(np.all(temp == b, axis=1))
	return result[0][0] if result else None

def line_remover(s): #deprecated
	return re.sub(r'(\n\s*)', ' ',s)
		
def space_remover(s): #deprecated
	return re.sub(' +',' ',s)

def sub_punctuations(s): #deprecated
	if isinstance(s,float):
		return ""
	s = str(s)
	s = s.replace("\t"," ")
	s = s.replace("''", '"').replace("``", '"')
	s = s.replace( "''",'"')
	s = s.replace('`',"'")
	s = s.strip()
	#s = s.replace('. "','."')
	s = s.replace(' :',':')
	s = s.replace(' .','.')
	s = s.replace(' ?','?')
	s = s.replace(' ,',',')
	s = s.replace(' !','!')
	s = s.replace(' )',')')
	s = s.replace('( ','(')
	s = re.sub(continuation_pattern_dot,'...',s)
	s = re.sub(continuation_pattern_interjection,'!',s)
	s = re.sub(continuation_pattern_question,'?',s)
	
	all_pat = re.finditer(full_stop_pattern,s)
	all_start = [0] + [i.start(0)+1 for i in all_pat]
	all_pat = re.finditer(bracket_pattern,s)
	all_start = all_start+ [i.start(0)+1 for i in all_pat]
	all_start = sorted(all_start)
	
	x = ""
	for k in range(1,len(all_start)):
		x = x +" "+s[all_start[k-1]:all_start[k]]
	
	x = x + " "+s[all_start[-1]:]
	s = x	
	s = s.strip()

	return s


def get_start_index(document,span):

	starts = [match.start() for match in re.finditer(re.escape(span.lower()), document.lower())]
	doc_tokens, doc_start, doc_end = tokenize_with_indices(document)
	ends = [i+len(span)for i in starts]
	for s,e in zip(starts,ends):
		if e in doc_end:
			return s
	if len(starts) == 0:
		print('error')
		print('span')
		print(span)
		print(document)
		#input()
		return -1	


def trim_response_retaining_span(response,span):
	response_tokens,response_start, response_end = tokenize_with_indices(response)
	span_tokens, span_start, span_end = tokenize_with_indices(span)

	l_response = len(response_tokens)
	l_span = len(span_tokens)
	
	

	r_start = response_start[0:max_response_length]
	r_end = response_end[0:max_response_length]
	
	
	start = find_first_numpy(response_tokens,span_tokens)
	end = start + l_span
	span_diff = max_response_length - start
	
	if l_span >= max_response_length:
		r_start = response_start[start:start+max_response_length]
		r_end = response_end[start:start+max_response_length]
		s_start = span_start[0:max_response_length]
		s_end = span_end[0:max_response_length]
		return response[r_start[0]:r_end[-1]], span[s_start[0]:s_end[-1]]

	if span_diff < 3:
		r_start = response_start[start:start+max_response_length]
		r_end = response_end[start:start+max_response_length]
		s_start = span_start[0:max_response_length]
		s_end = span_end[0:max_response_length]

	else:
		r_start = response_start[0:max_response_length]
		r_end = response_end[0:max_response_length]
		s_start = span_start[0:span_diff]
		s_end = span_end[0:span_diff]
	
		
	return response[r_start[0]:r_end[-1]], span[s_start[0]:s_end[-1]]


		



def trim_document_with_span(document, span, trim_length,chat_id):
	document_tokens, doc_start, doc_end = tokenize_with_indices(document)
	l_document = len(document_tokens)
	if l_document < trim_length:
		return document, document.lower().index(span.lower())
	
	answer_start = get_start_index(document,span) #character level
	answer_end = answer_start + len(span)
	if answer_start == -1:
		print()
		print(document)
		input()
	
	start = bisect_left(doc_start,answer_start)
	end = bisect_left(doc_start,answer_end)
	
	left = randint(0,l_document)
	while(not(start >= left and end <= (left+trim_length))):
		left = randint(0,l_document)
		
	
	d_start = doc_start[left:left+trim_length]
	d_end = doc_end[left:left+trim_length]
	reduced_document = document[d_start[0]:d_end[-1]]
	answer_start = get_start_index(reduced_document, span)
	return reduced_document, answer_start	


	
def trim_document_without_span(document,trim_length):
	document_tokens, doc_start, doc_end = tokenize_with_indices(document)
	act_len = len(document_tokens)
	if act_len < 2:
		return document
	if trim_length < 1:
		return ""
	if act_len - trim_length < 1:
		return "EOD"
	left = randint(0, act_len - trim_length)
	d_start = doc_start[left:left+trim_length]
	d_end = doc_end[left:left+trim_length]
	reduced_document = document[d_start[0]:d_end[-1]]
	
	return reduced_document 
	
	
def create_mini_documents(documents, document_lengths, span, label,trim_length,chat_id):
	document_array = [tokenize(i) for i in documents]
	span_tokens = tokenize(span)
	if np.sum(document_lengths) < trim_length:
		flat_doc = " ".join(i for i in documents)
	
	else:	
		l_span = len(span_tokens)
		all_doc_lens = np.array(document_lengths)	
		all_doc_lens = all_doc_lens*1.0/np.sum(all_doc_lens)
		all_doc_lens = all_doc_lens * trim_length
		all_doc_lens = [int(x) for x in all_doc_lens]
		
		if l_span > all_doc_lens[label]:
			diff = l_span - all_doc_lens[label]
			all_doc_lens[label]  = l_span 
			for i in range(0,len(all_doc_lens)):
				if i!=label and all_doc_lens[i] - diff > 0:
					all_doc_lens[i] = all_doc_lens[i] - diff
					break
				
		new_documents = []

		for k,doc in enumerate(documents):
			if k == label:
				f_doc,_ = trim_document_with_span(doc, span,all_doc_lens[k],chat_id)
			else:
				f_doc = trim_document_without_span(doc,all_doc_lens[k])
			
			new_documents.append(f_doc)
		
		flat_doc = " ".join(i for i in new_documents)
	answer_start = get_start_index(flat_doc, span)

	return flat_doc, answer_start




fname_input = 'train_data.json' #This is the data available under the main_data folder.
fname_output = 'train_data.json' #This is the data available under the raw_data folder. Please rename this file to avoid confusion. 
data = json.load(open(fname_input))
all_examples = []
stride = 1
start = 0

if alternate_responses_only:
	stride = 2
	start = 1
count = 0
for example in tqdm(data):
	labels = example['labels']
	spans = example['spans']
	utterances = example['chat']
	chat_id = example['chat_id']
	curr_imdb_id = example['imdb_id']
	documents = example['documents']
	len_chat = len(utterances)
	curr_comments = " ".join(c for c in documents['comments'])
	curr_comments = space_remover(line_remover((curr_comments)))
	flat_comments = curr_comments
	flat_doc_array = [documents['plot'].replace( "''",'"'), documents['review'].replace( "''",'"'),flat_comments.replace( "''",'"') ,documents['fact_table']['kb_flat'].replace( "''",'"')]	
	#flat_doc_array = [documents['plot'], documents['review'],flat_comments ,documents['fact_table']['kb_flat']]	
	
	flat_doc = " ".join(i for i in flat_doc_array)
	flat_doc_tokens = [ tokenize(i) for i in flat_doc_array]
	flat_doc_lens = [len(i) for i in flat_doc_tokens]
	example_id = 0

	
	for k in range(start,len_chat,stride):
		if labels[k] == 4:
			continue
		label = labels[k]
		response = utterances[k]
		response_tokens = tokenize(response)
		span = spans[k]
		
		query = utterances[k-1]
		

		if k < 3:
			history = ["NH"]
			context = query
		else: 
			history = utterances[:k-1] #without the query  
			context = " ".join(i for i in utterances[k-3:k]) #fincludes the query!

		
		
		if len(response_tokens) > max_response_length:
			response, span = trim_response_retaining_span(response,span)
			
		response_start_v = get_start_index(response,span)

			#oracle
		oracle = flat_doc_array[label]
		answer_start_o = get_start_index(oracle,span)
		
		#full
		answer_start_f = get_start_index(flat_doc,span)
		#oracle_reduced
		oracle_reduced, answer_start_o_r = trim_document_with_span(oracle, span, oracle_length,chat_id)
		
		#full_reduced
		full_reduced, answer_start_f_r = create_mini_documents(flat_doc_array, flat_doc_lens, span, label,full_length,chat_id)
		#mixed_reduced	
		mixed_doc, answer_start_m = create_mini_documents(flat_doc_array, flat_doc_lens, span, label, oracle_length,chat_id)
		if answer_start_o == -1 or answer_start_m == -1 or answer_start_f == -1 or answer_start_o_r == -1 or answer_start_f_r == -1:
			count = count + 1
			continue

		d = {'label':label,'chat_id':chat_id,'example_id': chat_id + str('_') + str(example_id),'imdb_id':curr_imdb_id,'movie_name':example['movie_name'],'query':query,'response':response, 'span':span, 'full_history': history, 'short_history': history[-2:],'context':context, 'chat': utterances, 'all_documents':documents, 
				'answer_start_oracle':answer_start_o,'answer_start_oracle_reduced':answer_start_o_r,'answer_start_mixed':answer_start_m,'answer_start_full':answer_start_f,'answer_start_full_reduced':answer_start_f_r,
				'oracle': oracle, 'full':flat_doc, 'full_reduced':full_reduced,'oracle_reduced':oracle_reduced,'mixed':mixed_doc,'response_start':response_start_v}
		example_id = example_id + 1
		
		all_examples.append(d)	
print(count)
json.dump(all_examples,open(fname_output,'w'))