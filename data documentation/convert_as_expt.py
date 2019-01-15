import pickle
import sys
import os
import json
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2
import pickle
import re
import spacy
from tqdm import *
import nltk
import numpy as np
from bisect import bisect_left
nlp = spacy.load('en',disable=['tagger','ner'],vectors=False)
print('Spacy loaded')
def get_tokens(doc):
	doc = nlp(doc)
	new_tokens = []
	for k in doc:
		new_tokens.append(k.text)
	return new_tokens

def word_tokenize(tokens):
	return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]



answer_start_dict = {'full':'answer_start_full','oracle':'answer_start_oracle','mixed':'answer_start_mixed','oracle_reduced':'answer_start_oracle_reduced','full_reduced':'answer_start_full_reduced'}
query_dict = {'context':'context','query':'query'}

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

	# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
CHUNK_SIZE = 1000
def chunk_file(chunks_dir,finished_files_dir, set_name):
  in_file = finished_files_dir+'/%s.bin' % set_name
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1


def chunk_all(chunks_dir,finished_files_dir): #need a common file ?
  # Make a dir to hold the chunks
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  for set_name in ['train', 'val', 'test']:
    print ("Splitting %s data into chunks..." % set_name)
    chunk_file(chunks_dir,finished_files_dir,set_name)
  print ("Saved chunked data in %s" % chunks_dir)



def write_to_bin(url_file, out_file, finished_files_dir,makevocab=False):
  url_list = url_file #convert every example to a file ?
  #''
  VOCAB_SIZE = 25000
  if makevocab:
	vocab_counter = collections.Counter()

  with open(out_file, 'wb') as writer:
	for idx,s in enumerate(url_list):
	  if idx % 1000 == 0:
		print ("Writing story %i  percent done" % (idx))

	  article, query, abstract = get_art_abs((s))

	  # Write to tf.Example
	  tf_example = example_pb2.Example()
	  tf_example.features.feature['article'].bytes_list.value.extend([article])
	  tf_example.features.feature['query'].bytes_list.value.extend([query])

	  tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
	  tf_example_str = tf_example.SerializeToString()
	  str_len = len(tf_example_str)
	  writer.write(struct.pack('q', str_len))
	  writer.write(struct.pack('%ds' % str_len, tf_example_str))

	  # Write the vocab to file, if applicable
	  if makevocab:
		art_tokens = article.split(' ')
		abs_tokens = abstract.split(' ')
		que_tokens = query.split(' ')

		abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
		tokens = art_tokens + abs_tokens + que_tokens
		tokens = [t.strip() for t in tokens] # strip
		tokens = [t for t in tokens if t!=""] # remove empty
		vocab_counter.update(tokens)

  print ("Finished writing file %s\n" % out_file)

  # write vocab to file
  if makevocab:
	print ("Writing vocab file...")
	with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
	  for word, count in vocab_counter.most_common(VOCAB_SIZE):
		writer.write(word + ' ' + str(count) + '\n')
	print ("Finished writing vocab file")



def get_art_abs(story_file): 
  
  article = str(story_file['article'].lower() )
  query = str(story_file['query'].lower())

  abstract = SENTENCE_START + ' ' + str(story_file['abstract'].lower()) +' '+SENTENCE_END

  re.sub("\s+"," ",article)
  re.sub("\s+"," ",query)
  re.sub("\s+"," ",abstract)

  return article, query, abstract


def process_tokens(st):
	x = get_tokens(st)
	return " ".join(x)

def gttp_format(data,query_type,data_type):
		
	folder_name = data_type +'_' + query_type
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)


	finished_files_dir = folder_name + "/finished_files"
	chunks_dir = os.path.join(finished_files_dir, "chunked")
	train_data = []
	valid_data = []
	test_data = []
	for k,type_data in enumerate(data):
		for count,i in enumerate(type_data):
			if count%1000==0:
				print(count)
			example = {'article':process_tokens(i[data_type].lower()),'query':process_tokens(i[query_dict[query_type]].lower()),'abstract':process_tokens(i['response'].lower())}
			if k==0:
				train_data.append(example)
			elif k==1:
				valid_data.append(example)
			else:
				test_data.append(example)






	VOCAB_SIZE = 25000
	CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data

	all_train_urls = train_data
	all_val_urls = valid_data
	all_test_urls = test_data
	
	
	if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

	write_to_bin(all_test_urls, os.path.join(finished_files_dir, "test.bin"),finished_files_dir)
	write_to_bin(all_val_urls, os.path.join(finished_files_dir, "val.bin"),finished_files_dir)
	write_to_bin(all_train_urls, os.path.join(finished_files_dir, "train.bin"),finished_files_dir, makevocab=True)

	# Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
	chunk_all(chunks_dir,finished_files_dir)

	print(len(train_data))
	print(len(valid_data))
	print(len(test_data))

def squad_format(data,query_type,data_type):
	all_examples_train = []
	all_examples_test = []
	all_examples_valid	= []
	folder_name = data_type +'_' + query_type
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)

	for k,type_data in enumerate(data):	
		for example in type_data:				
			qas = []
			
			id_ = example['example_id']
			answer = [{'text': example['span'],'answer_start':example[answer_start_dict[data_type]]}]
			question = " ".join(word_tokenize(example[query_dict[query_type]])[-65:])
			q = {'question': question,'id':id_,'answers':answer}
			qas.append(q)
			title = example['imdb_id']
			context = example[data_type]

			new_d = {'qas':qas,'context':context}

		#if used_chat:
			curr_imdb_id = example['imdb_id']
			title = str(example['example_id'])
			j_d = {'paragraphs':[new_d],'title':title}
			if k==0:
				all_examples_train.append(j_d)
			elif k==1:
				all_examples_valid.append(j_d)
				
			else:
				all_examples_test.append(j_d)
		
	json_train = {'version':1.1,'data':all_examples_train}
	json_valid = {'version':1.1,'data':all_examples_valid}
	json_test = {'version':1.1,'data':all_examples_test}
	print(len(json_train))
	with open(folder_name+'/train-v1.1.json', 'w') as fp:
		json.dump(json_train, fp)

	with open(folder_name+'/dev-v1.1.json', 'w') as fp:
		json.dump(json_valid, fp)

	with open(folder_name+'/test-v1.1.json', 'w') as fp:
		json.dump(json_test, fp)

	print('Completed')	

def get_list_of_list(flat_list,count,keep): #flat list already has keep elements
	reverse_count = count[::-1]
	reverse_cum_sum = np.cumsum(reverse_count)
	ind_ = bisect_left(reverse_cum_sum,keep)
	if ind_ == 0:
		return [flat_list]
	rev_count = reverse_count[0:ind_]

	if ind_ <len(count):
		rev_count.append(keep - reverse_cum_sum[ind_-1])
	split_count = rev_count[::-1]
	start = 0
	list_of_list = []
	for i in split_count:

		list_of_list.append(flat_list[start:start+i])
		start = start + i
	return list_of_list 





def hred_format(data):
	folder_name = 'HRED'
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)
	
	full_history_train = []
	response_train = []
	full_history_test = []
	response_test = []
	full_history_valid = []
	response_valid = []
	keep = 90
	hist_count = 0
	for k,type_data in enumerate(data):
		for i in tqdm(type_data):
			history = i['short_history']
			query = i['query']
			this_history = []
			count = []
			if len(history) > 1:
				for j in history:
					tokens = get_tokens(j.lower())
					this_history.append(tokens)
					count.append(len(tokens))


			tokens = get_tokens(query.lower())
			this_history.append(tokens)
			count.append(len(tokens))
			if np.sum(count) > keep:
				flat_history = [item for sublist in this_history for item in sublist]
				final_tokens = flat_history[-keep:] #convert to list of list
				this_history = get_list_of_list(final_tokens,count,keep)
				if hist_count < 5:
					print(this_history)
				hist_count = hist_count + 1


			curr_imdb_id = i['imdb_id']
			if k==0:
				full_history_train.append(this_history)
				response_train.append(get_tokens(i['response'].lower())+[])
			elif k==1:
				full_history_valid.append(this_history)
				response_valid.append(get_tokens(i['response'].lower()))
			else:
				full_history_test.append(this_history)
				response_test.append(get_tokens(i['response'].lower()))	


	full_list_train = [full_history_train,response_train]
	data_train = json.dumps(full_list_train)
	with open(folder_name+'/train.json',"w") as f: 
		f.write(data_train)
	
	full_list_valid = [full_history_valid,response_valid]
	data_valid = json.dumps(full_list_valid)
	with open(folder_name+'/dev.json',"w") as f: 
		f.write(data_valid)
	
	full_list_test = [full_history_test,response_test]
	data_test = json.dumps(full_list_test)
	with open(folder_name+'/test.json',"w") as f: 
		f.write(data_test)

	print('Completed')





expt_type = sys.argv[1]
query_type = sys.argv[3]
data_type = sys.argv[2]

train_data = json.load(open('train_data.json','r'))
test_data = json.load(open('test_data.json','r'))
valid_data = json.load(open('dev_data.json','r'))
data = [train_data,valid_data, test_data]

if expt_type == 'squad':
	squad_format(data,query_type,data_type)
if expt_type == 'hred':
	hred_format(data)
if expt_type == 'gttp':
	gttp_format(data,query_type,data_type)