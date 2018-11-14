mkdir data
wget https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json -O data/coqa-train.json
wget https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json -O data/coqa-dev.json
wget http://nlp.stanford.edu/data/glove.6B.zip -O data/glove.6B.zip
unzip data/glove.6B.zip -d data
