#!/bin/bash

# export SEED=72270
export PYTORCH_SEED=`expr $SEED / 10`
export NUMPY_SEED=`expr $PYTORCH_SEED / 10`

# path to bert vocab and weights
# export BERT_VOCAB=bert-base-cased
# export BERT_WEIGHTS=bert-base-based

# path to dataset files
export TRAIN_PATH=data/CSAbstruct/train.jsonl
export DEV_PATH=data/CSAbstruct/dev.jsonl
export TEST_PATH=data/CSAbstruct/test.jsonl

# model
export USE_SEP=true  # true for our model. false for baseline
export WITH_CRF=false  # CRF only works for the baseline

# training params
export cuda_device=0
export BATCH_SIZE=2
export LR=2e-5
# export TRAINING_DATA_INSTANCES=1668
# export NUM_EPOCHS=2

# limit number of sentneces per examples, and number of words per sentence. This is dataset dependant
# export MAX_SENT_PER_EXAMPLE=20
# export SENT_MAX_LEN=80

# this is for the evaluation of the summarization dataset
export SCI_SUM=false
export USE_ABSTRACT_SCORES=false
export SCI_SUM_FAKE_SCORES=false  # use fake scores for testing

CONFIG_FILE=sequential_sentence_classification/config.jsonnet

python -m allennlp.run train $CONFIG_FILE  --include-package sequential_sentence_classification -s $SERIALIZATION_DIR "$@"
