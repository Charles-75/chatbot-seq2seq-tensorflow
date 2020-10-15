from config import threshold, max_length
from utils import save_voc
import tensorflow as tf
import numpy as np
import re


def get_data():
    lines = open("data/movie_lines.txt", "r", encoding='utf-8', errors='ignore').read().split('\n')
    conversations = open("data/movie_conversations.txt", "r", encoding='utf-8', errors='ignore').read().split('\n')

    # Creating a dictionary that maps each line and its id
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]

    # Creating a list of all the conversations
    conversations_ids = []
    for conversation in conversations[:-1]:  # we exclude the last row which is empty
        _conversation = conversation.split(' +++$+++ ')[-1]
        ids = _conversation[1:-1].replace("'", "").replace(" ", "")
        conversations_ids.append(ids.split(','))

    # Getting separately the questions and the answers
    questions = []
    answers = []
    for conversation in conversations_ids:
        for i in range(len(conversation) - 1):
            question_line = conversation[i]
            answers_line = conversation[i+1]
            questions.append(id2line[question_line])
            answers.append(id2line[answers_line])
    return questions, answers


# Doing a  first cleaning of the texts
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"n'", " not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"wouldn't", "would not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.?,]", "", text)
    return text


def preprocess_data(questions, answers):
    clean_questions = []
    for question in questions:
        clean_questions.append(clean_text(question))
    clean_answers = []
    for answer in answers:
        clean_answers.append(clean_text(answer))

    # Filtering out the questions and answers that are too short or too long
    short_questions = []
    short_answers = []
    for i, question in enumerate(clean_questions):
        if 2 <= len(question.split()) <= max_length:
            short_questions.append(question)
            short_answers.append(clean_answers[i])
    clean_questions = []
    clean_answers = []
    for i, answer in enumerate(short_answers):
        if 2 <= len(answer.split()) <= max_length:
            clean_answers.append(answer)
            clean_questions.append(short_questions[i])

    # Creating a dictionary that maps each word to its number of occurences
    word2count = {}
    for question in clean_questions:
        for word in question.split():
            word2count[word] = word2count.get(word, 0) + 1
    for answer in clean_answers:
        for word in answer.split():
            word2count[word] = word2count.get(word, 0) + 1

    # Creating two dictionaries that map the questions and answers words to a unique integer
    questionswords2int = {}
    word_number = 0
    for word, count in word2count.items():
        if count >= threshold:
            questionswords2int[word] = word_number
            word_number += 1
    answerswords2int = {}
    word_number = 0
    for word, count in word2count.items():
        if count >= threshold:
            answerswords2int[word] = word_number
            word_number += 1

    # Adding the last tokens to these two dictionaries
    tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
    for token in tokens:
        questionswords2int[token] = len(questionswords2int) + 1
    for token in tokens:
        answerswords2int[token] = len(answerswords2int) + 1

    # Creating the inverse dictionary of the answerswords2int dictionary
    answersint2words = {w_i: word for word, w_i in answerswords2int.items()}

    # Save vocabularies
    save_voc(questionswords2int, answerswords2int, answersint2words)

    # Adding the End of String token to the end of every answer
    for i in range(len(clean_answers)):
        clean_answers[i] += ' <EOS>'

    # Translating all questions and answers into integers and replacing all the words that were filtered out by <OUT>
    questions_to_int = []
    for question in clean_questions:
        question_ints = []
        for word in question.split():
            if word not in questionswords2int:
                question_ints.append(questionswords2int['<OUT>'])
            else:
                question_ints.append(questionswords2int[word])
        questions_to_int.append(question_ints)
    answers_to_int = []
    for answer in clean_answers:
        answer_ints = []
        for word in answer.split():
            if word not in answerswords2int:
                answer_ints.append(answerswords2int['<OUT>'])
            else:
                answer_ints.append(answerswords2int[word])
        answers_to_int.append(answer_ints)

    # Sorting questions and answers by the length of questions (it will speed up the training)
    sorted_questions = []
    sorted_answers = []
    for length in range(1, max_length+1):
        for i, question in enumerate(questions_to_int):
            if len(question) == length:
                sorted_questions.append(question)
                sorted_answers.append(answers_to_int[i])
    return sorted_questions, sorted_answers, questionswords2int, answerswords2int


# Converting the questions from strings to lists of encoding integers
def convert_string2int(question, word2int):
    question = clean_text(question)
    return [word2int.get(token, word2int['<OUT>']) for token in question.split()]


# Preprocessing the targets
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], strides=(1,1))
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets


# Padding the sequences with the <PAD> token
# Question: ['who', 'are', 'you', '<PAD>', '<PAD>', '<PAD>']
# Answer: ['<SOS>', 'i', 'am', 'a', 'bot', '<EOS>']
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]


# Splitting the data into batches of questions/answers and apply padding
def split_into_batches(questions, answers, batch_size, questionswords2int, answerswords2int):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index:start_index+batch_size]
        answers_in_batch = answers[start_index:start_index+batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerswords2int))
        yield padded_questions_in_batch, padded_answers_in_batch
