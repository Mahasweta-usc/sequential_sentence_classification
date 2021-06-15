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
MAX_LEN = int(os.environ["SENT_MAX_LEN"]); print(MAX_LEN)
import multiprocessing
from multiprocessing import Pool, ThreadPool

n_cpu = multiprocessing.cpu_count() - 1

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
      if idx - prev <= 1: 
        print(sentences[prev:idx+1],"-----"," ".join(sentences[prev:idx+1]))
        verbs.append(" ".join(sentences[prev:idx+1]))
      else:
        for elem in sentences[prev:idx+1]: verbs.append(elem)
      prev = idx + 1

  for elem in sentences[prev:]: verbs.append(elem)
  return verbs



def segment_text(chunk):

  url = chunk["url"]
  email_sent = [sent_elem for sent_elem in sent_break(process_(chunk["last_reply"]))] 

  def prune():
    global candidate
    temp = []
    for elem in candidate:
        if any(len(elem) < len(cand) and set(elem).issubset(set(cand)) for cand in candidate):
            temp.append(elem)

    for elem in temp: candidate.remove(elem)

  def single_entries():
    global candidate
    for idx, elem in enumerate(candidate):
      if len(elem) == 1:
        if MAX_LEN > 2*len(tokenizer.tokenize(elem[0])):
          candidate[idx].append(elem[0])
        else:
          str_len = int(len(elem[0])/2)
          candidate[idx] = [elem[0][:str_len]]
          candidate[idx].append(elem[0][str_len:])

  def segmenter():
    global candidate, email_sent
    sum_ = 0;
    lim = int(0.9*MAX_LEN)
    candidate.append([])
    for idx, elem in enumerate(email_sent):
      remain = lim - sum_
      sum_ += len(tokenizer.encode(elem))
      if sum_ > lim:
        retain = tokenizer.convert_tokens_to_string(tokenizer.tokenize(elem[:remain]))
        carryover = tokenizer.convert_tokens_to_string(tokenizer.tokenize(elem[remain:]))
        if not idx:
          email_sent[idx] = carryover
          email_sent.insert(idx,retain)
          candidate[-1].append(retain) 
          # return idx + 1
        return idx
      
      candidate[-1].append(elem)


  def email_to_json():
    json_data = []
    entry = dict()
    entry[url]["sentences"] = candidate
    entry["labels"] = [["0"]*len(row) for row in candidate]
    entry["abstract_id"] = 0
    json_data.append(entry)



  candidate = []

  while True:
    pos = segmenter()
    if len(email_sent) > 1: email_sent.pop(0)
    else: break

  candidate = [cand for cand in candidate if cand]
  prune()
  single_entries()
  if len(candidate) > 500 or not len(candidate): return []
  email_to_json()
  return json_data

def process_text(filename):
  out_path = filename.replace('.csv','_IS.csv')
  f = pd.read_csv(filename,lineterminator='\n')

  cols = f.columns.tolist() + ['last_reply','IS_count','IS_']
  outtable = pd.DataFrame(columns = cols)
  row_count = 0

  for i,chunk in f.iterrows():
      email = EmailReplyParser.parse_reply(chunk["message"].replace('.>','\n>'))
      chunk["last_reply"] = email
      chunk['IS_count'] = 0
      chunk['IS_'] = ""
      outtable.loc[len(outtable.index)] = chunk

  pool = ThreadPool(n_cpu)
  results = []
  for row_no,row in outtable.iterrows():
      results.append(pool.apply_async(foo, args=(row)))

  pool.close()
  pool.join()
  out = [r.get() for r in results]
  return out, outtable
  


  






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



