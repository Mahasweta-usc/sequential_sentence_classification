from typing import List
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

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
nlp = stanza.Pipeline(lang='en', processors='tokenize')
from email_reply_parser import EmailReplyParser
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
MAX_LEN = os.environ["SENT_MAX_LEN"]; print(MAX_LEN)

def process_(text):
  text = text.replace("\r\n"," ").replace("\n"," ")
  text = " ".join(re.sub('>',"",text).split())
  return text

def sent_break(text):
    doc = nlp(text)
    lines = [line.text for line in doc.sentences]
    return lines

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
  global candidate
  global email_sent
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


def email_to_json(chunk,url):
    global json_dict, candidate
    for idx, row in enumerate(candidate):
        entry = dict()
        entry["sentences"] = row
        entry["labels"] = [0]*len(row)
        entry["abstract_id"] = 0
        entry["url"]
        json_dict.append(entry)


file_path = os.environ["file_path"]
output = os.environ["out_path"]

@Predictor.register('SeqClassificationPredictor')
class SeqClassificationPredictor(Predictor):
    """
    Predictor for the abstruct model
    """
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        json_dict, candidate = [], []
        filename = os.environ["file_path"]
        output = os.environ["out_path"]
        f = pd.read_csv(filename); f.dropna(inplace=True)
        cols = f.columns.tolist() + ['last_reply','IS_count','IS_']
        out = pd.DataFrame(columns=cols)

        for i,chunk in f.iterrows():
            folder = chunk["folder"]
            url = chunk["url"]
            predictions = []

            try: #dev_user += 1
              if folder.strip() not in ['dev','user','users','announce']: continue #'user','dev','users','announce'
              email = EmailReplyParser.parse_reply(chunk["message"].replace('.>','\n>'))
              email_sent = [sent_elem for sent_elem in sent_break(process_(email))]
              chunk["last_reply"] = email
              chunk['IS_count'] = ""
              chunk['IS_'] = ""
            except Exception as e:
              print(e)

            while True:
              pos = segmenter()
              if len(email_sent) > 1: email_sent.pop(0)
              else: break

            prune()
            single_entries()
            email_to_json(chunk,url)

            for elem in json_dict:
                instances = self._dataset_reader.text_to_instance(sentences=elem["sentences"],labels=elem["labels"])
                output = self._model.forward_on_instances([instances])
                # print(output)
                idx = output[0]['action_probs'].argmax(axis=1).tolist()
                
                predictions += itertools.compress(data["sentences"],data["predictions"])
                
            predictions = list(set(predictions))
            chunk["IS_"] = predictions
            chunk["IS_count"] = len(predictions)
            print(predictions)

            if predictions:
              IS_count += 1
              out.loc[len(out.index)] = chunk 
              out.drop_duplicates(subset=['url'], inplace=True)
              out.to_csv(output)









  