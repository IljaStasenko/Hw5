import random
import json
import torch
import weather
import haiku

from aiogram import Router, F
from aiogram.dispatcher import router
from aiogram.types import Message
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weather_Await = False
haiku_Await = False


with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


def botRES(message: Message):
    global weather_Await
    global haiku_Await

    if not weather_Await and not haiku_Await:
        sentence = tokenize(message.text)
        print(message.text)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.8:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    if tag != 'Погода' and tag != 'Хоку':
                        return f"{random.choice(intent['responses'])}"
                    elif tag == 'Погода':
                        weather_Await = True
                        return f"Введите город"
                    else:
                        haiku_Await = True
                        return f"Введите ключевые слова на английском"
        else:
            return f"Не понял вопрос..."
    elif weather_Await:
        weather_Await = False
        return f"{weather.weatherA(message.text)}"
    else:
        haiku_Await = False
        return f"{haiku.retrNeuro(message.text)}"
