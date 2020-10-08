# chatbot-seq2seq-tensorflow

A chatbot implemented in Python using tensorflow.  
The chatbot has been built based on a seq2seq model with attention mecanism and trained with the Cornell Dialogs corpuse that can be found here : [link](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

## Libraries used
  * Tensorflow 1.0.0
  * Numpy 1.18.5

## Training

To train the chatbot, go to the project's root directory and execute  `python training.py`, it may take a long long time to train (several hours even days depending on your computer ressources). At the end it will save the weights of the seq2seq model `chatbot_weights.ckpt` in the current folder. 

## Play with the chatbot

To interact with the chatbot, execute `python main.py`. You have to write your questions in the command prompt in order to get a response back from the chatbot. 

## Seq2seq Architecture 

![seq2seq model](https://miro.medium.com/max/3972/1*1JcHGUU7rFgtXC_mydUA_Q.jpeg)
![seq2seq in depth](https://miro.medium.com/max/691/1*5nvwJsH4EfONv_fdKNvobA.png)

## Attention Mecanism

![attention_mecanism](https://miro.medium.com/max/1200/1*1V221DO9QIafh4htkwVBYw.jpeg)
