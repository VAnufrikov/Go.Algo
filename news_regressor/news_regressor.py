import datetime
from datetime import timedelta as td

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsRegressor


class NewsRegressor:
    def __init__(self):
        self.df = pd.read_excel("news_regressor/news.xlsx")[
            ["date", "news", "proc", "ticket"]
        ]
        self.vectorizer = SentenceTransformer("cointegrated/rubert-tiny2")

    def vectorize(self, text):
        return self.vectorizer.encode(text)

    def predict(self, date, ticket):
        """По текущему времени и названию тикета получаем доступные новости
        и по истории новостей до текущего времени определяем влияние на тикеты
        """
        df = self.df

        result = 0
        timestamp = datetime.datetime.strptime(str(date)[:19], "%Y-%m-%dT%H:%M:%S")
        all_news = df[df["date"] < timestamp]
        ticket_news = df[
            (df["ticket"] == ticket)
            & (df["date"] < timestamp + td(hours=3))
            & (df["date"] > timestamp - td(minutes=15))
        ]["news"].unique()
        if len(all_news) > 0 and len(ticket_news) > 0:
            regressor = KNeighborsRegressor()
            vectors = self.vectorizer.encode(all_news["news"].values.tolist())
            regressor.fit(vectors, all_news["proc"])
            predict = regressor.predict(self.vectorizer.encode(ticket_news))
            result = round(sum(predict) / len(predict), 3)
        return result + 1
