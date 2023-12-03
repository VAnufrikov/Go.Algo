import pandas as pd
from etna.datasets.tsdataset import TSDataset
from stock_portfolio.portfolio import Stocks
from settings import DATE_START, DATE_END, LIMIT

import etna


def run_agent():
    """Входом будет получение датасета за прошлые даты,
                выход решение о покупке или продаже """
    portfel = Stocks()
    agent = Agent()

    tiket = portfel.get_tiket()

    df_between_dates = portfel.get_time_baket(tiket, DATE_START, DATE_END)

    inside = agent.predict(df_between_dates)

    if inside == 1:
        agent.by()

    elif inside == -1:
        agent.sell()
    else:
        agent.do_nofing()


def etna_predict(param):
    print(param.head())

    return 0


def clean_df_for_etna(df):

    df = df[
        ['secid', 'trade_datetime', 'pr_open', 'pr_high', 'pr_low', 'vol', 'val', 'trades', 'trades_b', 'trades_s',
         'val_b',
         'val_s', 'vol_b', 'vol_s', 'pr_close']]

    df.columns = ['segment', 'timestamp', 'pr_open', 'pr_high', 'pr_low', 'vol', 'val', 'trades', 'trades_b',
                  'trades_s',
                  'val_b', 'val_s', 'vol_b', 'vol_s', 'target']


class Agent:
    """Класс, который описывает поведение агента в среде биржи со своим портфелем"""

    def __int__(self):
        """Инициализация агента и его портфеля"""
        self.limit = LIMIT

    def predict(self, df):
        df.loc[:, 'trade_datetime'] = pd.to_datetime(df.tradedate.astype(str) + ' ' + df.tradetime.astype(str))

        # Сохраняем исходные trade_datetime для визуализации#
        list_trade_datetime = df['trade_datetime'].to_list()

        clean_data_orderstats = clean_df_for_etna(df)

        predict = etna_predict(TSDataset.to_dataset(df))

        """Тут мы прописываем логику обработки предикта etna:
        
        если будет рост отдаем 1 
        если падение -1 
        если так же то ничего не делаем 
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
