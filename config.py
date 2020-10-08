## Preprocessing parameters ##
threshold = 20  # filter words that are below this threshold
max_length = 25  # maximum sequence length
voc_path = "./params"  # path where the question and answer vocabularies are saved in pickle format after preprocessing

## Setting the Hyperparamaters of the seq2seq model ##
params = {
    'epochs': 100,
    'batch_size': 64,
    'rnn_size': 512,
    'num_layers': 3,
    'encoding_embedding_size': 512,
    'decoding_embedding_size': 512,
    'learning_rate': 0.01,
    'learning_rate_decay': 0.9,
    'min_learning_rate': 0.0001,
    'keep_probability': 0.5
}

## Training ##
batch_index_check_training_loss = 100
early_stopping = 1000
checkpoint = "./chatbot_weights.ckpt"  # path of the trained weights



