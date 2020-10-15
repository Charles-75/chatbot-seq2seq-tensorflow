## Preprocessing parameters ##
threshold = 15  # filter words that are below this threshold
max_length = 20  # maximum sequence length
voc_path = "./params"  # path where the question and answer vocabularies are saved in pickle format after preprocessing

## Setting the Hyperparamaters of the seq2seq model ##
params = {
    'epochs': 100,
    'batch_size': 32,
    'rnn_size': 1024,
    'num_layers': 2,
    'encoding_embedding_size': 1024,
    'decoding_embedding_size': 1024,
    'learning_rate': 0.001,
    'learning_rate_decay': 0.9,
    'min_learning_rate': 0.0001,
    'keep_probability': 0.5
}

## Training ##
batch_index_check_training_loss = 100
early_stopping = 100
checkpoint = "./trained_weights.ckpt"  # path of the trained weights



