import datetime
import pandas as pd

from datetime import timedelta as td
from sklearn.neighbors import KNeighborsRegressor
from sentence_transformers import SentenceTransformer


class NewsRegressor():
    def __init__(self):
        self.df = pd.read_excel('news.xlsx')[['date', 'news', 'proc', 'ticket']]
        self.vectorizer = SentenceTransformer('rubert-tiny2')

    def vectorize(self, text):
        return self.vectorizer.encode(text)

    def predict(self, date, ticket):
        """ по текущему времени и названию тикета получаем доступные новости 
            и по истории новостей до текущего времени определеяем влияние на тикеты
        """
        df = self.df
        result = 1.0
        timestamp = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        all_news = df[df['date']<timestamp]
        ticket_news = df[ (df['ticket'] == ticket) & (df['date']<timestamp+td(hours=3)) & (df['date']>timestamp-td(minutes=15)) ]['news'].unique()
        if len(all_news)>0 and len(ticket_news) > 0:
            regressor = KNeighborsRegressor()
            vectors = self.vectorizer.encode(all_news['news'].values.tolist())
            regressor.fit(vectors, all_news['proc'])
            predict = regressor.predict(self.vectorizer.encode(ticket_news))
            result = round(sum(predict)/len(predict), 3)
        return result


