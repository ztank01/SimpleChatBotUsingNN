"""
@author: Ztank(19110492 - Trinh Cong Truong)
"""
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random


words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

#tiền xử lý
for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize từng từ
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #thêm documents vào kho ngữ liệu
        documents.append((w, intent['tag']))

        # Nếu tag chưa có trong class thì thêm vào
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize và lower từng từ và xóa trùng lặp
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sắp xếp classes
classes = sorted(list(set(classes)))
# documents = kết hợp giữa patterns và intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = tất cả từ, từ vựng
print (len(words), "unique lemmatized words", words)
#Lưu class words và classes
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# Tạo dữ liệu training và số hóa dữ liệu
training = []
#Tạo mảng rỗng cho output
output_empty = [0] * len(classes)
#Tập huấn luyện, bag of words cho mỗi câu
for doc in documents:
    # khởi tạo bag of words
    bag = []
    #Danh sách tokenized words cho mẫu, doc[0]=words, doc[1]=tag
    pattern_words = doc[0]
    # lemmatize từng từ - tạo từ cơ bản, cố tạo những từ liên quan
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word được tìm thấy trong current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    #classes.index(doc[1]) = tag, output_row để xác định tag của doc[0]
    output_row[classes.index(doc[1])] = 1
    #mỗi training sẽ có đc danh sách các từ và tag của nó
    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. Từ X(words) xác định Y(tag)

train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=500, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")
