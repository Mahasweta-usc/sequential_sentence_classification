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
MAX_LEN = 100
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


# def single_entries(url):
# 	global candidate
# 	for idx, elem in enumerate(candidate[url]):
# 		if len(elem) == 1:
# 			if MAX_LEN > 2*len(tokenizer.tokenize(elem[0])):
# 				candidate[url][idx].append(elem[0])
# 			else:
# 				str_len = int(len(elem[0])/2)
# 				candidate[url][idx] = [elem[0][:str_len]]
# 				candidate[url][idx].append(elem[0][str_len:])

def single_entries(url):
  global email_sent
  lim_ = MAX_LEN;idx = 0

  while 1:
    try:
      elem = email_sent[url][idx]
      if len(elem) > lim_:
        email_sent[url][idx] = elem[:lim]
        email_sent[url][idx].insert(idx+1,elem[lim:])
      idx += 1
    except: break

# def segmenter(url):
# 	global candidate, email_sent
# 	sum_ = 0;
# 	lim = int(0.9*MAX_LEN)
# 	candidate[url].append([])
# 	for idx, elem in enumerate(email_sent[url]):
# 		remain = lim - sum_
# 		sum_ += len(tokenizer.encode(elem))
# 		if sum_ > lim:
# 			retain = tokenizer.convert_tokens_to_string(tokenizer.tokenize(elem[:remain]))
# 			carryover = tokenizer.convert_tokens_to_string(tokenizer.tokenize(elem[remain:]))
# 			if not idx:
# 				email_sent[url][idx] = carryover
# 				email_sent[url].insert(idx,retain)
# 				candidate[url][-1].append(retain) 
# 				# return idx + 1
# 			return idx
# 		candidate[url][-1].append(elem)


def segmenter(url):
  global candidate
  global email_sent
  templates = [[-2,-1,0,1,2],[-1,0,1,2],[-2,-1,0,1],[0,1,2],[-2,-1,0]]
  for idx, elem in enumerate(email_sent[url]):
    for template in templates:
      try:
        indices = np.add(idx,np.array(template))
        new_ = [str(x) for x in np.array(email_sent[url])[indices.astype(int)]]
        candidate[url].append(new_)
        break
      except: pass

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
	single_entries(url)
	segmenter(url)

	# candidate[url] = [cand for cand in candidate[url] if len(cand)]
	# if len(candidate[url]) > 20 or not len(candidate[url]): candidate[url] = candidate[url][:20]
	# print(len(candidate[url]))
	json_results = email_to_json(url)
	if not current%100: print("{} emails segmented".format(current))
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
		#outfile = filename ##for only segmentation and prediction
		outfile = filename.replace(".csv","_IS_graduated.json")
		final_res = dict()
		for month in range(24): final_res[month] = [] 

		f = pd.read_csv(filename,lineterminator='\n');f.dropna(subset=["content","message_id"],inplace=True)
		f = f[(f["status"] == 'graduated') & (f['month'] < 24)] #[:1000]
		f = f[f["folder"].isin(["dev","user","users","announce"])]
		print("No of entires: ",f.shape[0])
		row_count = 0
		print("Reading file")
		f['last_reply'] = f.last_reply.apply(lambda x: literal_eval(str(x)))
		#comment for only segmentation and prediction
		# f["last_reply"] = f["content"].apply(lambda x: sent_break(process_(EmailReplyParser.parse_reply(x.replace('.>','\n>')))))
		# f["IS_count"] = [0]*f.shape[0]
		f["embeddings"] = [""]*f.shape[0]

		json_data = process_text(f);miss_count = [0,0,0]
		print("Emails processed: ", len(list(json_data.keys())))
		print("Segmentation done. Starting predictions")
		for indx, row in tqdm(f.iterrows()):
			url = row["message_id"]
			sentences = json_data[url]["sentences"]
			labels = json_data[url]["labels"]
			predictions = []#;print(url)
			embeddings = []
			final_embed = []
			
			for sentence, label in zip(sentences,labels):
				try:
					# print(sentence,label)
					instances = self._dataset_reader.text_to_instance(sentences=sentence,labels=label)
					output = self._model.cuda().forward_on_instances([instances])
					# print(sentence)
					idx = output[0]['action_probs'].argmax(axis=1).tolist()
					logits = [self._model.vocab.get_token_from_index(i, namespace='labels') for i in idx]
					binary_labels = [int(item.split("_")[0]) for item in logits]
					# embeddings.extend(list(itertools.compress(output[0]['embeddings'].tolist(),binary_labels))); #print(np.shape(embeddings))
					ind_interest = np.floor((len(binary_labels)-1)/2).astype(int)
					if binary_labels[ind_interest] : predictions.append(sentence[ind_interest]) #.extend(list(itertools.compress(sentence,binary_labels))) #;print(sum(binary_labels))
				except: pass
			
			# assert len(embeddings) == len(predictions)
			final_res[row['month']].extend(final_embed)
			org_preds = row["IS_"].split("<Institutional>")
			miss_count[0] += len(org_preds)
			pred_out = list(set(predictions))
			if len(org_preds) != len(pred_out): miss_count[1] += len(set(predictions).difference(set(org_preds)));#print(org_preds,'\n',pred_out)

			for index,pred in enumerate(predictions):
				for x in org_preds: 
					if fuzz.ratio(x,pred) > 90: 
						# final_embed.append(embeddings[index])
						org_preds.remove(x)

			miss_count[2] += len(org_preds)
			if not indx%100: 
				print(miss_count)
				with open(outfile, 'w') as fout: json.dump(final_res, fout, indent=4)
			# assert len(final_embed) == len(row["IS_"].split("<Institutional>")) - len(org_preds)
			# if len(org_preds): miss_count[0] += 1

			# print(len(row["IS_"].split("<Institutional>")),len(embeddings))
			# print("Predicted:",len(set(row["IS_"].split("<Institutional>"))))# print("IS count: ",len(pred_out));
			# f.at[indx,"embeddings"] = final_embed
			# f.at[indx,"IS_count"] = len(pred_out)
			# if not indx%10000: 
			# 	# with open(outfile, 'w') as fout: json.dump(final_res, fout, indent=4)
			# 	print(indx,len(final_res[row['month']]))
			
		
		exit()