"""
@author: Ztank(19110492 - Trinh Cong Truong)
"""
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from gtts import gTTS
import playsound
from datetime import datetime
import speech_recognition as sr

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
r = sr.Recognizer()

def speak(text):
    date_string = datetime.now().strftime("%d%m%Y%H%M%S")
    tts = gTTS(text=text, lang='en')
    filename = 'voice' +date_string + '.mp3'
    tts.save(filename)
    playsound.playsound(filename)
    
def clean_up_sentence(sentence):
    # tokenize mẫu - chia words vào array
    sentence_words = nltk.word_tokenize(sentence)
    # stem từng từ - tạo dạng rút gọn cho từ
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# trả về bag of words array: 0 hoặc 1 cho mỗi từ tồn tại trong câu

def bow(sentence, words, show_details=True):
    # tokenize mẫu
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # gán 1 nếu từ hiện tại là từ được lưu trong words tại vị trí này
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # lọc ra các dự đoán dưới ngưỡng
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


#Tạo giao diện người dùng với tkinter
import tkinter
from tkinter import *


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
    
        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
        
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
        speak(res)
        
def ConverSpeechToText():
    with sr.Microphone() as source:
        try:
            print("Adjusting...")
            r.adjust_for_ambient_noise(source, duration=1)
            print("Recording...")
            recorded_audio = r.listen(source, timeout=5)
            #Recorgnizing the Audio
            EntryBox.delete("0.0",END)
            EntryBox.insert("0.0","Recorgnizing.....")
            text = r.recognize_google(
                        recorded_audio, 
                        language="en"
                        )
            EntryBox.delete("0.0",END)
            EntryBox.insert("0.0",text)
        except:
            text = ""
            
base = Tk()
base.title("Chatbot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Tạo cửa sổ chat
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Tạo nút để gửi tin nhắn
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )
#Tạo núi để nhập bằng giọng nói
VoiceButton = Button(base, font=("Verdana",12,'bold'), text="Voice", width="12", height=5,
                    bd=0, bg="#ff0505", activebackground="#ab0000",fg='#ffffff',
                    command= ConverSpeechToText )

#Tạo nơi để nhập tin nhắn
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=45)
VoiceButton.place(x=6, y=446, height=45)

base.mainloop()
