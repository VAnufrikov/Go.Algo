from datetime import timedelta

import pandas as pd
from etna.datasets.tsdataset import TSDataset

from settings import DATE_START, DATE_END, LIMIT
from stock_portfolio.portfolio import Stocks


def run_agent():
    """Входом будет получение датасета за прошлые даты,
                выход решение о покупке или продаже """
    portfel = Stocks()
    agent = Agent()

    """
    Тут мы должны получать по каждому тикету 
    предикт на стоимость портфеля и далее анализировать какую часть портфеля
    мы можем купить по всем позициям
    """
    tiket = portfel.get_tiket()

    tradestats, orderstats, obstats = portfel.get_time_baket(tiket, DATE_START, DATE_END)

    inside = agent.predict(tradestats, orderstats, obstats)

    if inside == 1:
        agent.by()

    elif inside == -1:
        agent.sell()
    else:
        agent.do_nofing()


def etna_predict(param):
    print(param.head())

    return 0


def etna_train():
    """Производим обучение модели по подготовленым данным"""
    pass


def make_fake_datetime(df):
    """Создаем непрерывный временной ряд для etna"""
    start = df['trade_datetime'][0]

    list_datetime = []

    for i in range(len(df['trade_datetime'])):
        if i == 0:
            list_datetime.append(start)
        else:
            start = list_datetime[i - 1]
            list_datetime.append(start + timedelta(minutes=1))

    df['fake_datetime'] = list_datetime
    return df


def clean_df_for_etna(orderstats, tradestats, obstats):
    tradestats['segment'] = 'price'
    orderstats['segment'] = 'vol'
    obstats['segment'] = 'val'

    # Предиктим цену закрытия
    tradestats = tradestats[['trade_datetime', 'segment', 'pr_close']]
    tradestats = make_fake_datetime(tradestats)
    tradestats.columns = ['timestamp', 'segment', 'target']

    # Предиктим обьем продаж по позиции
    orderstats['vol_true_put'] = orderstats['put_vol_s'] - orderstats['cancel_vol_s']
    orderstats = orderstats[['trade_datetime', 'segment', 'vol_true_put']]
    orderstats = make_fake_datetime(orderstats)
    orderstats.columns = ['timestamp', 'segment', 'target']

    # Предиктим количество заявок которые доступно если все смогут и купить и продать
    # По сути смотрим тренд по заявкам
    obstats['val_true_trend'] = obstats['val_b'] - obstats['val_s']
    obstats = obstats[['trade_datetime', 'segment', 'val_true_trend']]
    obstats = make_fake_datetime(obstats)
    obstats.columns = ['timestamp', 'segment', 'target']

    df = pd.concat([
        tradestats, orderstats, obstats
    ], ignore_index=True).reset_index(drop=True)

    return TSDataset.to_dataset(df)


class Agent:
    """Класс, который описывает поведение агента в среде биржи со своим портфелем"""

    def __int__(self):
        """Инициализация агента и его портфеля"""
        self.limit = LIMIT

    def predict(self, tradestats, orderstats, obstats):
        """Тут мы прописываем логику обработки предикта etna:

               если будет рост отдаем 1
               если падение -1
               если так же то ничего не делаем
                """

        # Создаем норм таймлайн для 3х датасетов
        tradestats.loc[:, 'trade_datetime'] = pd.to_datetime(
            tradestats.tradedate.astype(str) + ' ' + tradestats.tradetime.astype(str))
        orderstats.loc[:, 'trade_datetime'] = pd.to_datetime(
            orderstats.tradedate.astype(str) + ' ' + orderstats.tradetime.astype(str))
        obstats.loc[:, 'trade_datetime'] = pd.to_datetime(
            obstats.tradedate.astype(str) + ' ' + obstats.tradetime.astype(str))

        # Сохраняем исходные trade_datetime для визуализации
        list_trade_datetime_tradestats = tradestats['trade_datetime'].to_list()
        list_trade_datetime_orderstats = orderstats['trade_datetime'].to_list()
        list_trade_datetime_obstats = obstats['trade_datetime'].to_list()

        clean_data_orderstats, clean_data_tradestats, clean_data_obstats = clean_df_for_etna(orderstats, tradestats,
                                                                                             obstats)

        predict = etna_predict(TSDataset.to_dataset(clean_data_orderstats, clean_data_tradestats, clean_data_obstats))

        """Тут нужно просписать логику сравнения текущей цены и цены в будующем
        
        если дороже покупаем:  1
        
        если ниже шортим: -1 
        
        если такая же возращем 0 
        
        """
        return 0

    def by(self):
        """Реализация выставление тикета в стакан на покупку"""
        pass

    def sell(self):
        """Реализация выставление тикета в стакан на продажу"""
        pass

    def do_nofing(self):
        """Не выставляем тикет и просто ждем, возвращаем действие ничего не делаем"""
        pass
