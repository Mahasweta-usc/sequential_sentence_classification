from typing import List
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
import argparse
import jsonlines
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["cuda_device"]="0"
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
MAX_LEN = 256
from tqdm import tqdm
import multiprocessing
import torch
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
	lines = [line.text for line in doc.sentences if len(line.text) > 1 ]
	return lines


def prune(url):
	global candidate
	temp = []
	for elem in candidate[url]:
		if any(len(elem) < len(cand) and set(elem).issubset(set(cand)) for cand in candidate[url]):
			temp.append(elem)
	for elem in temp: candidate[url].remove(elem)

def single_entries(url):
	global candidate
	for idx, elem in enumerate(candidate[url]):
		if len(elem) == 1:
			if MAX_LEN > 2*len(tokenizer.tokenize(elem[0])):
				candidate[url][idx].append(elem[0])
			else:
				str_len = int(len(elem[0])/2)
				candidate[url][idx] = [elem[0][:str_len]]
				candidate[url][idx].append(elem[0][str_len:])

def segmenter(url):
	global candidate, email_sent
	sum_ = 0;
	lim = int(0.9*MAX_LEN)
	candidate[url].append([])
	for idx, elem in enumerate(email_sent[url]):
		remain = lim - sum_
		sum_ += len(tokenizer.encode(elem))
		if sum_ > lim:
			retain = tokenizer.convert_tokens_to_string(tokenizer.tokenize(elem[:remain]))
			carryover = tokenizer.convert_tokens_to_string(tokenizer.tokenize(elem[remain:]))
			if not idx:
				email_sent[url][idx] = carryover
				email_sent[url].insert(idx,retain)
				candidate[url][-1].append(retain) 
				# return idx + 1
			return idx
		candidate[url][-1].append(elem)


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
	email_sent[url] = chunk["last_reply"]
	candidate[url] = []
	while True:
		pos = segmenter(url)
		win = min(len(email_sent[url]),int(len(candidate[url][-1])/2) + 1)
		if win > 0:
			for _ in range(win): email_sent[url].pop(0)
		else: break

	candidate[url] = [cand for cand in candidate[url] if cand]
	prune(url)
	single_entries(url)
	# if len(candidate[url]) > 20 or not len(candidate[url]): candidate[url] = candidate[url][:20]
	print(len(candidate[url]))
	json_results = email_to_json(url)
	if not current%100: print("{} emails segmented".format(current))
	return json_results


def process_text( f):
	n_cpu = multiprocessing.cpu_count() - 1

	pool = Pool(n_cpu)
	results = []
	print("Segmenting emails")
	for i ,row in f.iterrows():
		results.append(pool.apply_async(segment_text, args=(row,row["message_id"],i)))

	pool.close()
	pool.join()
	out = dict()
	for r in results: out.update(r.get()) #print(r.get()); 
	# print(outtable.shape[0],len(list(out.keys())))
	# outtable["last_reply"] = outtable["last_reply"].apply(tuple)
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
		outfile = filename.replace(".csv","_IS.csv")

		f = pd.read_csv(filename,lineterminator='\n');f.dropna(subset=["content","message_id"],inplace=True)
		f = f[f["folder"].isin(["dev","user","users","announce"])]
		print("No of entires: ",f.shape[0])
		cols = f.columns.tolist() + ['last_reply','IS_count','IS_']
		out = pd.DataFrame(columns = cols)
		row_count = 0
		print("Reading file")
		#comment for only segmentation and prediction
		f["last_reply"] = f["content"].apply(lambda x: sent_break(process_(EmailReplyParser.parse_reply(x.replace('.>','\n>'))))[:50])
		f["IS_count"] = [0]*f.shape[0]
		f["IS_"] = [""]*f.shape[0]

		json_data = process_text(f)
		print("Emails processed: ", len(list(json_data.keys())))
		print("Segmentation done. Starting predictions")
		for indx, row in tqdm(f.iterrows()):
			url = row["message_id"]
			sentences = json_data[url]["sentences"]
			labels = json_data[url]["labels"]
			predictions = []

			for sentence, label in zip(sentences,labels):
				instances = self._dataset_reader.text_to_instance(sentences=sentence,labels=label)
				output = self._model.forward_on_instances([instances])
				# print(output)
				idx = output[0]['action_probs'].argmax(axis=1).tolist()
				logits = [self._model.vocab.get_token_from_index(i, namespace='labels') for i in idx]
				binary_labels = [int(item.split("_")[0]) for item in logits]
				predictions += list(itertools.compress(sentence,binary_labels))

			pred_out = list(set(predictions))
			f.at[indx,"IS_"] = "<Institutional>".join(pred_out)
			f.at[indx,"IS_count"] = len(pred_out)

			if not indx%100: f.to_csv(outfile,index=False);print(f[f["IS_count"] > 0].shape[0])
		f.to_csv(outfile,index=False)
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
#     """
#     Predictor for the abstruct model
#     """
#     def predict_json(self, json_dict: JsonDict) -> JsonDict:
#         pred_labels = []
#         sentences = json_dict['sentences']
#         paper_id = json_dict['abstract_id']
#         try:
#           labels = json_dict['labels']
#         except:
#           labels = [["1"]*len(sent) for sent in sentences]
#         print(sentences,labels)
#         instance = self._dataset_reader.text_to_instance(sentences=sentences,labels=labels)
#         output = self._model.forward_on_instances([instance])
#         # print(output)
#         idx = output[0]['action_probs'].argmax(axis=1).tolist()
#         # print(idx)
#         labels = [self._model.vocab.get_token_from_index(i, namespace='labels') for i in idx]
#         # print(labels)
#         pred_labels.extend(labels)
#         assert len(pred_labels) == len(sentences)
#         preds = list(zip(sentences, pred_labels))


#         with jsonlines.open(file_path, mode='a') as writer:
#           json_dict["predictions"] = pred_labels
#           writer.write(json_dict)
#         return paper_id, preds


