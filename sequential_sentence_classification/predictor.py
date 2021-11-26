from typing import List
from overrides import overrides
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
import argparse
import jsonlines
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
os.environ["cuda_device"]= '0'
import random
import json
import numpy as np
import regex as re
from fuzzywuzzy import fuzz, process
import itertools
from itertools import combinations, compress
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize,word_tokenize
import pandas as pd
import stanza
stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize',use_gpu=True,tokenize_batch_size=4)
from email_reply_parser import EmailReplyParser
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
MAX_LEN = 50
from tqdm import tqdm
import multiprocessing
import torch
from ast import literal_eval
from multiprocessing import Pool, Manager #, set_start_method
# try:
# 	set_start_method('spawn', force=True)
# except RuntimeError:
#     pass
    


# mn = Manager()
candidate = dict()
email_sent = dict()

def process_(text):
  text = text.replace("\r\n"," ").replace("\n"," ")
  text = " ".join(re.sub('>',"",text, flags=re.IGNORECASE).split())
  return text

def sent_break(text):
	doc = nlp(text)
	lines = [line.text for line in doc.sentences if len(line.text) > 1]
	return lines

def single_entries(url,item):
  lim_ = MAX_LEN;idx = 0
  email_temp = email_sent[url].copy()[max(0,email_sent[url].index(item)-2):min(email_sent[url].index(item)+3,len(email_sent[url]))]
  org_len = len(email_temp)

  while 1:
    try:
      if email_temp[idx] != item:
        elem = email_temp[idx]
        if len(tokenizer.tokenize(elem)) > lim_:
          email_temp[idx] = decode(tokenizer.tokenize(elem)[:lim_])
          email_temp.insert(idx+1,decode(tokenizer.tokenize(elem)[lim_:]))
      idx += 1
    except: break
  return email_temp 

def decode(seg):
	# chunk = tokenizer.convert_ids_to_tokens(seg)
	text = tokenizer.convert_tokens_to_string(seg)
	return text 
	
def fit_len(sent):
  order = [4,0,3,1]
  while len(tokenizer.tokenize(" ".join(sent))) > 500:
    try:
      sent[order[0]] = '[PAD]'
      order.pop(0)
    except:
      sent[2] = decode(tokenizer.tokenize(sent[2])[:500])
      break
  return sent


def segmenter(url):
	global candidate
	candidate[url].append([])
	template = [-2,-1,0,1,2]
	for idx, elem in enumerate(email_sent[url]):
		temp = ["[PAD]"]*2 + single_entries(url,elem) + ["[PAD]"]*2
		indices = np.add(temp.index(elem),np.array(template))
		new_ = np.array(temp)[indices.astype(int)].tolist()
		new_ = fit_len(new_)
		assert len(new_) == 5
		candidate[url].append(new_)

def email_to_json(url):
	global candidate
	json_data = []
	entry = {url:dict()}
	entry[url]["sentences"] = candidate[url]
	entry[url]["labels"] = [["0"]*len(row) for row in candidate[url]]
	entry[url]["abstract_id"] = 0
	assert len(entry[url]["labels"]) == len(entry[url]["sentences"])
	return entry


def segment_text(chunk,url,current):
	global candidate, email_sent
	email_sent[url] = chunk["last_reply"][:50]
	candidate[url] = []
	segmenter(url) #;print(len(email_sent[url]),len(candidate[url]))

	# candidate[url] = [cand for cand in candidate[url] if len(cand)]
	# if len(candidate[url]) > 20 or not len(candidate[url]): candidate[url] = candidate[url][:20]
	# print(len(candidate[url]))
	json_results = email_to_json(url)
	return json_results


def process_text(f):
	n_cpu = 2 # multiprocessing.cpu_count() //2 #1 # multiprocessing.cpu_count() //2

	pool = Pool(n_cpu)
	results = []
	print("Segmenting emails")
	for i ,row in f.iterrows():
		# results.append(segment_text(row, row["message_id"], i))
		results.append(pool.apply_async(segment_text, args=(row,row["message_id"],i)))

	pool.close()
	pool.join()
	out = dict()
	for r in results:
		out.update(r.get())
	return out
 
 
@Predictor.register('SeqClassificationPredictor')
class SeqClassificationPredictor(Predictor):
	"""
	Predictor for the abstruct model
	""" 
	def predict_json(self, json_dict: JsonDict) -> JsonDict:
		print("Enter full file path: ")
		filename = os.environ["FILE_PREDS"];print(filename)
		#outfile = json to store IS month and project wise 
		outfile = "/content/gdrive/MyDrive/full_messages/sample_10_IS.json"

		
		f = pd.read_csv(filename);print("No of entires: ",f.shape[0])
		f.dropna(subset=['body','message_id','month'],inplace=True)
		f['month'] = f['month'].apply(lambda x: int(x))
		f['is_bot'] = f['is_bot'].apply(lambda x: str(x))
		f['from_commit'] = f['from_commit'].apply(lambda x: str(x));print(f.columns)
		
		#only reads months 0 - 24
		f = f[(f['month'].isin(list(range(0,24))))]
		##excludes bot emails and commit emails
		f = f[(f["is_bot"] == 'False') & (f["from_commit"] == 'False')]

		##sample fraction of data to extract IS

		# sample_size = int(0.01*f.shape[0])

		# try: f = f.sample(sample_size)
		# except : pass

		try:
			with open(outfile) as fin: 
				final_res = json.load(fin)
		except:
			final_res = dict()
			for month in list(range(0,24)): 
				final_res[str(month)] = dict()
				for proj in f['project_name'].unique(): final_res[str(month)][proj] = []

		row_count = 0 #(f["status"] == 'graduated') & 
		print("No of entires: ",f.shape[0])
		print("Reading file")
		# f['last_reply'] = f.last_reply.apply(lambda x: literal_eval(str(x)))
		#comment for only segmentation and prediction
		f["last_reply"] = f["body"].apply(lambda x: sent_break(process_(EmailReplyParser.parse_reply(x.replace('.>','\n>')))))
		f["IS"] = [""]*f.shape[0]
		print("Processing emails")

		json_data = process_text(f)
		print("Emails processed: ", len(list(json_data.keys())))
		print("Segmentation done. Starting predictions")
		for indx, row in tqdm(f.iterrows()):
			url = row["message_id"]
			sentences = json_data[url]["sentences"]
			labels = json_data[url]["labels"]
			embeddings = []
			predictions = []

			for sentence, label in zip(sentences,labels):
				try:
					self._dataset_reader.predict = True
					instances = self._dataset_reader.text_to_instance(sentences=sentence)
					output = self._model.cuda().forward_on_instances([instances])
					# print(sentence)
					idx = output[0]['action_probs'].argmax(axis=1).tolist()
					logits = [self._model.vocab.get_token_from_index(i, namespace='labels') for i in idx]
					binary_labels = [int(item.split("_")[0]) for item in logits]
					# embeddings.extend(list(itertools.compress(output[0]['embeddings'].tolist(),binary_labels))); #print(np.shape(embeddings))
					ind_interest = 2;#print(binary_labels)
					if binary_labels[ind_interest]: predictions.append(sentence[ind_interest]) #.extend(list(itertools.compress(sentence,binary_labels))) #;print(sum(binary_labels))
				except Exception as e: pass
			
			##store IS to csv
			if predictions: f.at[indx,'IS'] = "<IS>".join(predictions)
			final_res[str(row['month'])][row['project_name']].extend(predictions)
			##save results by project and month in json
		with open(outfile, 'w') as fout: json.dump(final_res, fout, indent=4)
		##save csv with IS to external csv
		f.to_csv(filename.replace(".csv","_IS.csv"))
		exit()


# from typing import List
# from overrides import overrides

# from allennlp.common.util import JsonDict, sanitize
# from allennlp.data import Instance
# from allennlp.predictors.predictor import Predictor
# import jsonlines

# import os
# file_path = os.environ["file_path"]

# @Predictor.register('SeqClassificationPredictor')
# class SeqClassificationPredictor(Predictor):
#   """
#   Predictor for the abstruct model
#   """
#   def predict_json(self, json_dict: JsonDict) -> JsonDict:
#       pred_labels = []
#       sentences = json_dict['sentences']
#       paper_id = json_dict['abstract_id']
#       try:
#         labels = json_dict['labels']
#       except:
#         labels = [["1"]*len(sent) for sent in sentences]
#       print(sentences,labels)
#       self._dataset_reader.predict= True
#       instance = self._dataset_reader.text_to_instance(sentences=sentences)
#       output = self._model.cuda().forward_on_instances([instance])
#       # print(output)
#       idx = output[0]['action_probs'].argmax(axis=1).tolist()
#       # print(idx)
#       labels = [self._model.vocab.get_token_from_index(i, namespace='labels') for i in idx]
#       # print(labels)
#       pred_labels.extend(labels)
#       assert len(pred_labels) == len(sentences)
#       preds = list(zip(sentences, pred_labels))


#       with jsonlines.open(file_path, mode='a') as writer:
#         json_dict["predictions"] = pred_labels
#         writer.write(json_dict)
#       return paper_id, preds