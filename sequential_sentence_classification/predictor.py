# #For full email prediction

# from typing import List
# from overrides import overrides
# from allennlp.common.util import JsonDict, sanitize
# from allennlp.data import Instance
# from allennlp.predictors.predictor import Predictor
# import argparse
# import jsonlines
# import os, sys
# os.environ["CUDA_VISIBLE_DEVICES"]= '0'
# os.environ["cuda_device"]= '0'
# import random
# import json
# import numpy as np
# np.random.seed(0) 
# import regex as re
# from fuzzywuzzy import fuzz, process
# import itertools
# from itertools import combinations, compress
# import nltk
# nltk.download('punkt')
# from nltk.tokenize import sent_tokenize,word_tokenize
# import pandas as pd
# import stanza
# stanza.download('en')
# nlp = stanza.Pipeline(lang='en', processors='tokenize',use_gpu=True,tokenize_batch_size=4)
# from email_reply_parser import EmailReplyParser
# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# MAX_LEN = 256
# from tqdm import tqdm
# import multiprocessing
# import torch
# from ast import literal_eval
# from multiprocessing import Pool, Manager #, set_start_method
# # try:
# # 	set_start_method('spawn', force=True)
# # except RuntimeError:
# #     pass
    


# mn = Manager()
# candidate = mn.dict()
# email_sent = mn.dict()

# def process_(text):
#   text = text.replace("\r\n"," ").replace("\n"," ")
#   text = " ".join(re.sub('>',"",text, flags=re.IGNORECASE).split())
#   return text

# def sent_break(text):
# 	doc = nlp(text)
# 	lines = [line.text for line in doc.sentences if len(line.text) > 1]
# 	return lines

# def chunks(lst, n):
# 	for i in range(0, len(lst), n):
# 		yield lst[i:i + n]

# def single_entries(url):
#   global candidate
#   for idx, elem in enumerate(candidate[url]):
#     if len(elem) < 2:
#       elem = tokenizer.tokenize(elem[0])
#       length = MAX_LEN - len(elem) - 1
#       if MAX_LEN > 2*len(elem):
#         addage = tokenizer.convert_tokens_to_string(elem)
#         candidate[url][idx].append(addage)
#       else:
#         candidate[url][idx] = []
#         for subsent in chunks(elem,int(len(elem)/2) + 1):
#           addage = tokenizer.convert_tokens_to_string(subsent)
#           candidate[url][idx].append(addage)

# def decode(seg):
# 	# chunk = tokenizer.convert_ids_to_tokens(seg)
# 	text = tokenizer.convert_tokens_to_string(seg)
# 	return text 

# def segmenter(url):
#   global candidate;email_sent
#   sum_ = 0;
#   lim = int(0.9*MAX_LEN)
#   # print("before",candidate[url])
#   seg = []
#   for idx, elem in enumerate(email_sent[url]):
#     remain = lim - sum_
#     # print("after",candidate[url])
#     sum_ += len(tokenizer.encode(elem))
#     if sum_ > lim:
#       retain = tokenizer.convert_tokens_to_string(tokenizer.tokenize(elem[:remain]))
#       carryover = tokenizer.convert_tokens_to_string(tokenizer.tokenize(elem[remain:]))
#       if not idx:
#         # email_sent[url][idx] = carryover
#         # email_sent[url].insert(idx,retain)
#         seg.append(retain)
#         # return idx + 1
#       return seg
    
#     seg.append(elem)
#   return seg

# def email_to_json(url):
# 	global candidate
# 	json_data = []
# 	entry = {url:{}}
# 	entry[url]["sentences"] = candidate[url]
# 	entry[url]["labels"] = [["0"]*len(row) for row in candidate[url]]
# 	entry[url]["abstract_id"] = 0
# 	assert len(entry[url]["labels"]) == len(entry[url]["sentences"])
# 	return entry

# def prune(url):
# 	global candidate
# 	temp = []
# 	for elem in candidate[url]:
# 		if any(len(elem) < len(cand) and set(elem).issubset(set(cand)) for cand in candidate[url]):
# 			temp.append(elem)
# 	for elem in temp: candidate[url].remove(elem)

# def segment_text(chunk,url,current):
# 	global email_sent, candidate
# 	email_sent[url] = chunk["last_reply"][:50]
# 	candidate[url] = []
# 	while True:
# 		last_seg = segmenter(url)
# 		if last_seg: candidate[url] += [last_seg] #;print(len(candidate[url][-1]),len(email_sent[url]))
# 		win_ = int(0.5*len(last_seg))
# 		# if len(email_sent[url]) > 1: email_sent[url].pop(0)
# 		# else: break
# 		if win_ < len(email_sent[url]) :email_sent[url]=email_sent[url][win_+1:]
# 		else:break
# 	prune(url)
# 	single_entries(url)
# 	json_results = email_to_json(url)
# 	return json_results

# def segment_email(x):
# 	try:
# 		return sent_break(process_(EmailReplyParser.parse_reply(x.replace('.>','\n>'))))
# 	except:
# 		return sent_break(process_(x.replace('.>','\n>')))

# def process_text(f):
# 	n_cpu = int(multiprocessing.cpu_count()/3) # //2 #1 # multiprocessing.cpu_count() //2
# 	pool = Pool(n_cpu)
# 	results = []
# 	print("Segmenting emails")
# 	for i ,row in f.iterrows():
# 		# results.append(segment_text(row, row["message_id"], i))
# 		results.append(pool.apply_async(segment_text, args=(row,row["message_id"],i)))

# 	pool.close()
# 	pool.join()
# 	out = dict()
# 	for r in results:
# 		out.update(r.get())
# 	return out
 
 
# @Predictor.register('SeqClassificationPredictor')
# class SeqClassificationPredictor(Predictor):
# 	"""
# 	Predictor for the abstruct model
# 	""" 
# 	def predict_json(self, json_dict: JsonDict) -> JsonDict:
# 		print("Enter full file path: ")
# 		filename = os.environ["FILE_PREDS"];print(filename)
# 		f = pd.read_csv(filename)
# 		print("No of entires: ",f.shape[0])
# 		print("Reading file")
# 		#comment for only segmentation and prediction
# 		try: 
# 			_ = f['last_reply']
# 			f['last_reply'] = f.last_reply.apply(lambda x: literal_eval(str(x)))
# 		except:
# 			f["last_reply"] = f["body"].apply(lambda x: segment_email(x))

# 		f["IS"] = [""]*f.shape[0]
# 		print("Processing emails")

# 		json_data = process_text(f)
# 		print("Emails processed: ", len(list(json_data.keys())),len(list(candidate.keys())),len(list(email_sent.keys())))
# 		print("Segmentation done. Starting predictions")
# 		for indx, row in tqdm(f.iterrows()):
# 			url = row["message_id"]
# 			sentences = json_data[url]["sentences"]
# 			labels = json_data[url]["labels"]
# 			predictions = []

# 			for sentence, label in zip(sentences,labels):
# 				try:
# 					self._dataset_reader.predict = True
# 					instances = self._dataset_reader.text_to_instance(sentences=sentence)
# 					output = self._model.cuda().forward_on_instances([instances])
# 					# print(sentence)
# 					idx = output[0]['action_probs'].argmax(axis=1).tolist()
# 					logits = [self._model.vocab.get_token_from_index(i, namespace='labels') for i in idx]
# 					binary_labels = [int(item.split("_")[0]) for item in logits]
# 					predictions.extend(list(itertools.compress(sentence,binary_labels))); #print(np.shape(embeddings))
# 				except Exception as e: pass
			
# 			##store IS to csv
# 			if predictions: f.at[indx,'IS'] = "<IS>".join(set(predictions))
# 		f.to_csv(filename.replace(".csv","_IS.csv"))
# 		exit()


#For test/validation
from typing import List
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
import jsonlines

import os
file_path = os.environ["file_path"]

@Predictor.register('SeqClassificationPredictor')
class SeqClassificationPredictor(Predictor):
  """
  Predictor for the abstruct model
  """
  def predict_json(self, json_dict: JsonDict) -> JsonDict:
      pred_labels = []
      sentences = json_dict['sentences']
      paper_id = json_dict['abstract_id']
      try:
        labels = json_dict['labels']
      except:
        labels = [["1"]*len(sent) for sent in sentences]
      print(sentences,labels)
      self._dataset_reader.predict= True
      instance = self._dataset_reader.text_to_instance(sentences=sentences)
      output = self._model.cuda().forward_on_instances([instance])
      # print(output)
      idx = output[0]['action_probs'].argmax(axis=1).tolist()
      # print(idx)
      labels = [self._model.vocab.get_token_from_index(i, namespace='labels') for i in idx]
      # print(labels)
      pred_labels.extend(labels)
      assert len(pred_labels) == len(sentences)
      preds = list(zip(sentences, pred_labels))


      with jsonlines.open(file_path, mode='a') as writer:
        json_dict["predictions"] = pred_labels
        writer.write(json_dict)
      return paper_id, preds