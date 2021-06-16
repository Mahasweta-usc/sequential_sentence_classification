from typing import List
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
import argparse
import jsonlines
import os, sys
import random
import json
import numpy as np
import regex as re
from fuzzywuzzy import fuzz, process
import itertools
from itertools import combinations, compress
from nltk.tokenize import sent_tokenize,word_tokenize
import pandas as pd
import stanza
stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos')
from email_reply_parser import EmailReplyParser
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
MAX_LEN = 256
from tqdm import tqdm
import multiprocessing
import torch
from torch.multiprocessing import Pool, Manager, set_start_method
try:
	set_start_method('spawn', force=True)
except RuntimeError:
    pass
    


# mn = Manager()
candidate = dict()
email_sent = dict()
	



def process_(text):
  text = text.replace("\r\n"," ").replace("\n"," ")
  text = " ".join(re.sub('>',"",text).split())
  return text

def sent_break(text):
	doc = nlp(text)
	sentences = [line.text for line in doc.sentences]
	lines = [line for line in doc.sentences]
	verbs = []
	prev = 0
	for idx, line in enumerate(lines):
		tags = [word.upos for word in line.words if word.upos in ['AUX','VERB']]
		if tags:
			if idx - prev <= 1: verbs.append(" ".join(sentences[prev:idx+1]))
			else:
				for elem in sentences[prev:idx+1]: verbs.append(elem)
			prev = idx + 1
	for elem in sentences[prev:]: verbs.append(elem)
	return verbs


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
	email_sent[url] = sent_break(process_(chunk["last_reply"]))
	candidate[url] = []

	while True:
		pos = segmenter(url)
		if len(email_sent[url]) > 1: email_sent[url].pop(0)
		else: break

	candidate[url] = [cand for cand in candidate[url] if cand]
	prune(url)
	single_entries(url)
	if len(candidate[url]) > 500 or not len(candidate[url]): candidate[url] = []
	# print(len(candidate[url]))
	json_results = email_to_json(url)
	if not current%1000: print("{} emails segmented".format(current))
	return json_results


def process_text( filename):
	n_cpu = multiprocessing.cpu_count() - 1
	f = pd.read_csv(filename,lineterminator='\n');f.dropna(inplace=True)
	f = f[f.folder.isin(["dev","user","users","announce"])]
	cols = f.columns.tolist() + ['last_reply','IS_count','IS_']
	outtable = pd.DataFrame(columns = cols)
	row_count = 0
	print("Reading file")
	# f["last_reply"] = f["message"].apply(lambda x: sent_break(process_(EmailReplyParser.parse_reply(x.replace('.>','\n>')))))
	# f["IS_count"] = [0]*f.shape[0]
	# f["IS_"] = [""]*f.shape[0]
	for i,chunk in tqdm(f.iterrows()):
		email = EmailReplyParser.parse_reply(chunk["message"].replace('.>','\n>'))
		chunk["last_reply"] = email
		chunk['IS_count'] = 0
		chunk['IS_'] = ""
		outtable.loc[len(outtable.index)] = chunk

	pool = Pool(n_cpu)
	results = []
	print("Segmenting emails")
	for i ,row in outtable.iterrows():
		results.append(pool.apply_async(segment_text, args=(row,row["url"],i,nlp)))

	pool.close()
	pool.join()
	out = dict()
	for r in results: out.update(r.get()) #print(r.get()); 
	# print(outtable.shape[0],len(list(out.keys())))
	outtable["last_reply"] = outtable["last_reply"].apply(tuple)
	return out, outtable
 

@Predictor.register('SeqClassificationPredictor')
class SeqClassificationPredictor(Predictor):
	"""
	Predictor for the abstruct model
	""" 
	def predict_json(self, json_dict: JsonDict) -> JsonDict:
		print("Enter full file path: ")
		filename = input()
		outfile = filename.replace(".csv","_IS.csv")
		json_data,out = process_text(filename)
		print("Segmentation done. Starting predictions")
		for idx, row in tqdm(out.iterrows()):
			print(idx)
			url = row["url"]
			sentences = json_data[url]["sentences"]
			labels = json_data[url]["labels"]
			predictions = []
			torch.cuda.empty_cache()

			for sentence, label in zip(sentences,labels):
				instances = self._dataset_reader.text_to_instance(sentences=sentence,labels=label)
				output = self._model.forward_on_instances([instances])
				# print(output)
				idx = output[0]['action_probs'].argmax(axis=1).tolist()
				logits = [self._model.vocab.get_token_from_index(i, namespace='labels') for i in idx]
				binary_labels = [int(item.split("_")[0]) for item in logits]
				predictions += list(itertools.compress(sentence,binary_labels))

			pred_out = list(set(predictions))
			row["IS_"] = "<Institutional>".join(pred_out)
			row["IS_count"] = len(pred_out)

			# if predictions: print(pred_out)
			out.to_csv(outfile,index=False)

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


