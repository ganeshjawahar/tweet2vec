## Improving Tweet Representations using Temporal and User Context

[![Join the chat at https://gitter.im/tweet2vec/](https://badges.gitter.im/abhshkdz/neural-vqa.svg)](https://gitter.im/tweet2vec/Lobby?utm_source=share-link&utm_medium=link&utm_campaign=share-link)

This repository contains the [Torch](http://torch.ch/) implementation of our [ECIR 2017 work](https://arxiv.org/abs/1612.06062). 

### Quick Start
Download the user profile attribute dataset from [here](http://cs.stanford.edu/~bdlijiwei/ACL_profile_full_data.zip)

Download the [Glove](http://nlp.stanford.edu/data/glove.twitter.27B.zip) word vectors trained on a super-large twitter corpus.

To train our model,

```
th main.lua
```

### Dependencies
* [Torch](http://torch.ch/)
* xlua
* tds
* optim
* nnx
* cutorch
* cunn
* cunnx

Packages (b) to (h) can be installed using:
```
luarocks install <package-name>
```

### Options

#### `th main.lua`

* `data_dir`: directory for accessing the user profile prediction data for an attribute (spouse or education or job) [data/spouse/]
* `glove_dir`: directory for accesssing the pre-trained glove word embeddings [data/]
* `pred_dir`: directory for storing the output (i.e., word, tweet and user embeddings) [predictions/]
* `to_lower`: should we change the case of word to lower case [1=yes (default), 0=no]
* `wdim`: dimensionality of word embeddings [200]
* `wwin`: size of the context window for word context model. add 1 for target word. [21]
* `twin`: size of the context window for tweet context model. add 1 for target tweet. [21]
* `min_freq`: words that occur less than <int> times will not be taken for training [5]
* `pad_tweet`: should we need to pad the tweet ? [1=yes (default), 0=no]
* `is_word_center_target`: should we model the center word as target. if marked 0, the last word will be considered as target. [0]
* `is_tweet_center_target`: should we model the center tweet as target. if marked 0, the last tweet will be considered as target. [1]
* `pre_train`: should we initialize word embeddings with pre-trained vectors? [1=yes (default), 0=no]
* `wc_mode`: how to get the hidden representation for the word context model? [1=concatenation, 2=sum (default), 3=average, 4=attention based average of the context embeddings]
* `tc_mode`: how to get the hidden representation for the tweet context model? [1=concatenation, 2=sum, 3=average, 4=attention based average (default) of the context embeddings]
* `tweet`: should we use the tweet based model too? [1=yes (default), 0=no]
* `user`: should we use the user based model too? [1=yes, 0=no (default)]
* `wpred`: what softmax to use for the final prediction in the word context model? [1=normal (time-consuming for large dataset), 2=hierarchical (default), 3=[brown](https://github.com/yoonkim/lstm-char-cnn) softmax]
* `tpred`: what softmax to use for the final prediction in the tweet context model? [1=normal (time-consuming for large dataset), 2=hierarchical (default), 3=[brown](https://github.com/yoonkim/lstm-char-cnn) softmax]
* `learning_rate`: learning rate for the gradient descent algorithm [0.001]
* `batch_size`: number of sequences to train on in parallel [128]
* `max_epochs`: number of full passes through the training data [25]

### Author
[Ganesh J](https://researchweb.iiit.ac.in/~ganesh.j/)

### Licence
MIT