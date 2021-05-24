import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer # to convert text to feature vectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import sys


vectorizer_file = "vectorizer.pickle"
fakeNewsModelFile =  "fakeNewsModel.pickle"

try:
    with open('stopwords.pickle', 'rb') as f:
        stopwords_file = pickle.load(f)
    # print('ok')
except:
    # print('nie ok')
    nltk.download('stopwords')
    with open('stopwords.pickle', 'wb') as f:
        pickle.dump(stopwords.words('english'), f)

    stopwords_file = stopwords.words('english')


def prepare_text(arr_of_text, vectorizer, fit=False):

    def preprocessor(content):
        return re.sub("[^a-zA-Z]",' ', content).lower().split()

    ps = PorterStemmer()

    def stemming(content):
        stemmed_content = [ps.stem(word) for word in preprocessor(content) if not word in stopwords_file]
        return ' '.join(stemmed_content)

    print('processing content')

    if isinstance(arr_of_text, str):
        arr_of_text = [stemming(arr_of_text)]
    else:
        arr_of_text = arr_of_text.apply(stemming)

    X = np.array(arr_of_text)
    print(X)

    if fit:
        # converting the textual data to numerical
        print('fitting vectorizer')
        vectorizer.fit(X)

    return vectorizer.transform(X)


# to use when machine is trained
try:
    with open(fakeNewsModelFile, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_file, 'rb') as f:
        vectorizer = pickle.load(f)

    print('loaded model')

    if len(sys.argv) > 1:
        articleFile = sys.argv[1]
    else:
        articleFile = 'article.txt'

    print(articleFile)
    try:
        with open(articleFile, 'r') as article:
            new_article = article.read()
    except IOError:
        print(f'Cannot open {articleFile}.')
        sys.exit(1)

    transformed_article = prepare_text(new_article, vectorizer)
    # print(transformed_article)
    article_prediction = model.predict(transformed_article)
    labels = ['reliable', 'unreliable']
    print()
    print('This article is', labels[article_prediction[0]])
except IOError:
    print('training')
    # load data to pandas
    news_dataset = pd.read_csv('./fake-news/train.csv')

    # count missing values
    # print(news_dataset.isnull().sum())

    # replacing null values with null
    news_dataset = news_dataset.fillna('')

    # merging author name and news columns
    # news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']
    # news_dataset['content'] = news_dataset['text']

    vectorizer = TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None)
    X = prepare_text(news_dataset['text'], vectorizer, True)

    with open(vectorizer_file, 'wb') as f:
        pickle.dump(vectorizer, f)

    predict = 'label'
    y = np.array(news_dataset[predict])

    #  train
    print('training model')
    acc_test = 0
    while acc_test <= 0.95:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        # prediction
        x_test_prediction = model.predict(X_test)
        acc_test = accuracy_score(x_test_prediction, y_test)
        print(acc_test)

    # save to file
    with open(fakeNewsModelFile, "wb") as f:
        pickle.dump(model, f)

    labels = ['reliable', 'unreliable']

    for x in range(100):
        print('Predicted:', labels[x_test_prediction[x]],  'Real:', labels[y_test[x]])

