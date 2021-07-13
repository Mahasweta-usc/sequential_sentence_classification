##clone repo and change into working directory 
##install libraries
pip install -r /content/sequential_sentence_classification/requirements.txt 
##downloading trained model
wget https://storage.googleapis.com/tempvaxx/model.tar.gz -P /content/sequential_sentence_classification/trained/
##for making predictions
allennlp predict  /content/sequential_sentence_classification/trained/model.tar.gz /content/sequential_sentence_classification/trained/test.jsonl --include-package sequential_sentence_classification  --predictor SeqClassificationPredictor  