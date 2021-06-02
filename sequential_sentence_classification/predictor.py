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
MAX_LEN = int(os.environ["SENT_MAX_LEN"]); print(MAX_LEN)

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


def email_to_json():
    global json_data, candidate
    for idx, row in enumerate(candidate):
      entry = dict()
      entry["sentences"] = row
      entry["labels"] = ["0"]*len(row)
      entry["abstract_id"] = 0
      json_data.append(entry)



json_data, candidate, email_sent = [], [], []

@Predictor.register('SeqClassificationPredictor')
class SeqClassificationPredictor(Predictor):
    """
    Predictor for the abstruct model
    """
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        global json_data, candidate, email_sent
        filename = os.environ["file_path"]
        out_path = filename.replace('.csv','_IS.csv')
        try:
          f = pd.read_csv("/content/local.csv")
        except:
          f = pd.read_csv("/content/local.csv",lineterminator='\n')

        cols = f.columns.tolist() + ['last_reply','IS_count','IS_']

        try:
            out = pd.read_csv(out_path)
        except Exception as e:
          out = pd.DataFrame(columns = cols)

        row_count = 0
        for i,chunk in f.iterrows():
            row_count += 1
            json_data, candidate = [], []
            url = chunk["url"]
            predictions = []
            email = EmailReplyParser.parse_reply(chunk["message"].replace('.>','\n>'))
            email_sent = [sent_elem for sent_elem in sent_break(process_(email))]
            chunk["last_reply"] = email
            chunk['IS_count'] = ""
            chunk['IS_'] = ""

            while True:
              pos = segmenter()
              if len(email_sent) > 1: email_sent.pop(0)
              else: break

            candidate = [cand for cand in candidate if cand]
            prune()
            single_entries()
            email_to_json()

            if len(candidate) > 500 or not len(candidate): continue

            for elem in json_data:
              instances = self._dataset_reader.text_to_instance(sentences=elem["sentences"],labels=elem["labels"])
              output = self._model.forward_on_instances([instances])
              # print(output)
              idx = output[0]['action_probs'].argmax(axis=1).tolist()
              labels = [self._model.vocab.get_token_from_index(i, namespace='labels') for i in idx]
              binary_labels = [int(item.split("_")[0]) for item in labels]
              predictions += list(itertools.compress(elem["sentences"],binary_labels))
              # sys.stdout.write(" ".join(labels) + "\n")
                
            pred_out = list(set(predictions))
            chunk["IS_"] = "<Institutional>".join(pred_out)
            chunk["IS_count"] = len(pred_out)

            if predictions:
              out.loc[len(out.index)] = chunk 
              out.drop_duplicates(subset=["url"],inplace=True)
              out.to_csv(out_path,index=False)
            
            with open("/content/output.txt",'a+') as op:
              op.write(str(out.shape[0])+"/"+str(f.shape[0])+"\n") 

            if row_count >= f.shape[0] - 1 : exit()






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



  