import numpy as np
import tqdm
import nltk
import pandas as pd


def tokenize_sentences(sentences, words_dict):
    tokenized_sentences = []
    for sentence in sentences:
        if hasattr(sentence, "decode"):
            sentence = sentence.decode("utf-8")
        tokens = nltk.tokenize.word_tokenize(sentence)
        result = []
        for word in tokens:
            word = word.lower()
            if word not in words_dict:
                words_dict[word] = len(words_dict)
            word_index = words_dict[word]
            result.append(word_index)
        tokenized_sentences.append(result)
    return tokenized_sentences, words_dict

def read_embedding_list(file_path):
    embedding_word_dict = {}
    embedding_list = []
    with open(file_path,encoding='utf-8') as f:
        for row in tqdm.tqdm(f.read().split("\n")[1:-1]):
            data = row.split(" ")
            word = data[0]
            embedding = np.array([float(num) for num in data[1:]])
            embedding_list.append(embedding)
            embedding_word_dict[word] = len(embedding_word_dict)

    embedding_list = np.array(embedding_list)
    return embedding_list, embedding_word_dict
# embedding_list, embedding_word_dict=read_embedding_list("test.vec")


def clear_embedding_list(embedding_list, embedding_word_dict, words_dict):
    cleared_embedding_list = []
    cleared_embedding_word_dict = {}

    for word in words_dict:
        if word not in embedding_word_dict:
            continue
        word_id = embedding_word_dict[word]
        row = embedding_list[word_id]
        cleared_embedding_list.append(row)
        cleared_embedding_word_dict[word] = len(cleared_embedding_word_dict)

    return cleared_embedding_list, cleared_embedding_word_dict


def convert_tokens_to_ids(tokenized_sentences, words_list, embedding_word_dict, sentences_length):
    words_train = []

    for sentence in tokenized_sentences:
        current_words = []
        for word_index in sentence:
            word = words_list[word_index]
            word_id = embedding_word_dict.get(word, len(embedding_word_dict) - 2)
            current_words.append(word_id)

        if len(current_words) >= sentences_length:
            current_words = current_words[:sentences_length]
        else:
            current_words += [len(embedding_word_dict) - 1] * (sentences_length - len(current_words))
        words_train.append(current_words)
    return words_train

UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
def load_data(train_file_path,test_file_path,test_label_path,seq_len,embedding_path):
    print("Loading data...")
    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)
    test_label=pd.read_csv(test_label_path)

    list_sentences_train = train_data["comment_text"].fillna(NAN_WORD).values
    _list_sentences_test = test_data["comment_text"].fillna(NAN_WORD).values
    Y_train = train_data[CLASSES].values
    _y_test=test_label[CLASSES].values

    list_sentences_test=[]
    Y_test=[]

    #remove test data whose label contains -1
    for i in range(len(_list_sentences_test)):
        if -1 in _y_test[i]:
            continue
        list_sentences_test.append(_list_sentences_test[i])
        Y_test.append(_y_test[i])


    print("Tokenizing sentences in train set...")
    tokenized_sentences_train, words_dict = tokenize_sentences(list_sentences_train, {})

    print("Tokenizing sentences in test set...")
    tokenized_sentences_test, words_dict = tokenize_sentences(list_sentences_test, words_dict)

    words_dict[UNKNOWN_WORD] = len(words_dict)

    print("Loading embeddings...")
    embedding_list, embedding_word_dict = read_embedding_list(embedding_path)
    embedding_size = len(embedding_list[0])

    print("Preparing data...")
    embedding_list, embedding_word_dict = clear_embedding_list(embedding_list, embedding_word_dict, words_dict)

    # UNK: [0 0 0 ... 0 0] -> vocab[-2]
    # END: [-1 -1 -1 ... -1 -1 -1] for padding sentence -> vocab[-1]
    embedding_word_dict[UNKNOWN_WORD] = len(embedding_word_dict)
    embedding_list.append([0.] * embedding_size)
    embedding_word_dict[END_WORD] = len(embedding_word_dict)
    embedding_list.append([-1.] * embedding_size)

    embedding_matrix = np.array(embedding_list)

    id_to_word = dict((id, word) for word, id in words_dict.items())
    train_list_of_token_ids = convert_tokens_to_ids(
        tokenized_sentences_train,
        id_to_word,
        embedding_word_dict,
        seq_len)
    test_list_of_token_ids = convert_tokens_to_ids(
        tokenized_sentences_test,
        id_to_word,
        embedding_word_dict,
        seq_len)
    X_train = np.array(train_list_of_token_ids)
    X_test = np.array(test_list_of_token_ids)
    index=int(0.8*len(X_train))
    train_data=X_train[:index]
    train_label=Y_train[:index]
    val_data=X_train[index:]
    val_label=Y_train[index:]
    test_data=X_test
    test_label=Y_test
    return train_data,train_label,val_data,val_label,test_data,test_label,embedding_matrix,embedding_word_dict

# X_train,Y_train,X_test,Y_test,embedding_matrix,embedding_word_dict=load_data('../data/train.csv','../data/test.csv','../data/test_labels.csv',500,'../data/crawl-300d-2M.vec')

# print(len(embedding_matrix))
# print(embedding_matrix[0:10])
# print(len(embedding_word_dict)) 143882
# # print(embedding_word_dict[0:10])
# print(len(X_train))
# print(X_train[0:10])
# print(len(Y_train))
# print(Y_train[0:10])
# print(len(X_test))
# print(X_test[0:10])
# print(len(Y_test))
# print(Y_test[0:10])

def batch_iter(x,y,batch_size):
    data_len=len(x)
    num_batch=int((data_len-1)/batch_size)+1

    for i in range(num_batch):
        start_id=i*batch_size
        end_id=min((i+1)*batch_size,data_len)
        if end_id-start_id<batch_size:
            break
        yield x[start_id:end_id],y[start_id:end_id]
