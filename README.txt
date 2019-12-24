dir_to_data -  diraction to the folder with ner/pos data

#################################################### Part 1 ################################################################

python3 tagger2.py  dir_to_data/pos/train   dir_to_data/pos/dev dir_to_data/pos/test pos

python3 tagger2.py  dir_to_data/ner/train dir_to_data/ner/dev dir_to_data/ner/test ner

#################################################### Part 2 ################################################################

python3 top_k.py wordVectors.txt vocab.txt

#################################################### Part 3 ################################################################

python3 tagger2.py  /home/ella/Ex_2/pos/train dir_to_data/pos/dev dir_to_data/pos/test pos wordVectors.txt vocab.txt

python3 tagger2.py  /home/ella/Ex_2/ner/train dir_to_data/ner/dev dir_to_data/ner/test ner wordVectors.txt vocab.txt

#################################################### Part 4 ################################################################

With pre trained:

python3 tagger3.py  dir_to_data/pos/train dir_to_data/pos/dev dir_to_data/pos/test pos wordVectors.txt vocab.txt

python3 tagger3.py  dir_to_data/ner/train dir_to_data/ner/dev dir_to_data/ner/test ner wordVectors.txt vocab.txt

Without pre trained:

python3 tagger3.py  dir_to_data/pos/train dir_to_data/pos/dev dir_to_data/pos/test pos 

python3 tagger3.py  dir_to_data/ner/train dir_to_data/ner/dev dir_to_data/ner/test ner 

