from preprocessing import convert_string2int
from chatbot_model import seq2seq_model, model_inputs
from config import params, max_length
from utils import load_voc
import tensorflow as tf
import numpy as np

if __name__ == "__main__":

    batch_size = params['batch_size']
    rnn_size = params['rnn_size']
    num_layers = params['num_layers']
    encoding_embedding_size = params['encoding_embedding_size']
    decoding_embedding_size = params['decoding_embedding_size']
    questionswords2int, answerswords2int, answersint2words = load_voc()

    # Loading the model inputs
    inputs, targets, lr, keep_prob = model_inputs()

    # Setting the sequence_length
    sequence_length = tf.placeholder_with_default(max_length, None, name='sequence_length')

    # Build the prediction tensorflow graph
    _, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                        targets,
                                        keep_prob,
                                        batch_size,
                                        sequence_length,
                                        len(answerswords2int),
                                        len(questionswords2int),
                                        encoding_embedding_size,
                                        decoding_embedding_size,
                                        rnn_size,
                                        num_layers,
                                        questionswords2int)

    # Load the weights and run the session
    checkpoint = "./chatbot_weights.ckpt"
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, checkpoint)

    # Setting up the chat
    while (True):
        question = input("You: ")
        if question == "Goodbye":
            break
        question = convert_string2int(question, questionswords2int)
        question = question + [questionswords2int['<PAD>']] * (20 - len(question))
        fake_batch = np.zeros((batch_size, 20))
        fake_batch[0] = question
        predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
        answer = ''
        for i in np.argmax(predicted_answer, 1):
            if answersint2words[i] == 'i':
                token = 'I'
            elif answersint2words[i] == '<EOS>':
                token = '.'
            elif answersint2words[i] == '<OUT>':
                token = 'out'
            else:
                token = ' ' + answersint2words[i]
            answer += token
            if token == '.':
                break
        print('ChatBot: ' + answer)
