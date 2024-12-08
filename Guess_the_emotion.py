import joblib
import spacy
import numpy as np
import pickle

nlp = spacy.load('ru_core_news_sm')

def ProProcessing(text:str):
    """Преобразует фразу в список из 91 токена"""
    doc = nlp(text)
    # print(f'После токинезации, получаю {doc.__len__()} токенов')
    # # Удаляю Стоп-слова
    filtered_tokens = [token for token in doc if not token.is_stop]
    # print(f'После удаления стоп-слов, получаю {filtered_tokens.__len__()} токенов.\n\tПРИМЕР токенов {filtered_tokens[:10]}')
    # # Привожу к нормальной форме
    filtered_tokens_lemma = np.unique([token.lemma_ for token in filtered_tokens])
    # print(f'После приведения тоекнов к lemma и удаления дублей, осталось {len(filtered_tokens_lemma)} токенов.\n\tПРИМЕР токенов {filtered_tokens_lemma[:20]}')
    return nlp(' '.join(filtered_tokens_lemma)).vector

# Загрузка модели и словаря эмодзи из файлов
model = joblib.load('bestmodel.pkl')
with open('target.pkl', 'rb') as f:
    emotions_dict = pickle.load(f)


# text = 'Самый лучший товар'
text = input("Введите текст:")
X = [ProProcessing(text)]
predict = model.predict(X)[0]
emotion = next(key for key, value in emotions_dict.items() if value == predict)
print(f'Сообщение несет эмоцию "{emotion}"')

