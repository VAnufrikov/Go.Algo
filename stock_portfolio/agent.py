import etna
import random
import pandas as pd

from settings import Config, LIMIT
from datetime import datetime as dt
from etna.datasets.tsdataset import TSDataset
from stock_portfolio.portfolio import Stocks


def run_agent():
    """Входом будет получение датасета за прошлые даты,
                выход решение о покупке или продаже """
    portfel = Stocks()
    agent = Agent()

    tiket = portfel.get_tiket()

    df_between_dates = portfel.get_time_baket(tiket, Config.DATE_START, Config.DATE_END)

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

    def by(self, ticket, count, price):
        """Реализация выставление тикета в стакан на покупку"""
        take_profit, stop_loss = self.get_TP_SL(ticket)
        datetime = str(dt.now())
        str_line = f"{datetime}|{ticket}|{price}|{count}|{take_profit}|{stop_loss}\n"
        with open(Config.STOCKS_PATH, 'a') as fout:
            fout.write(str_line)

    def sell(self):
        """Реализация выставление тикета в стакан на продажу"""
        pass

    def do_nofing(self):
        """Не выставляем тикет и просто ждем, возвращаем действие ничего не делаем"""
        pass

    def get_prices(self, stocks: list) -> list[tuple]:
        """ Получить закупочную стоимость акций, которые необходимо купить
        Args:
            stocks: список акций, которые необходимо купить.
        Returns:
            prices_list: [("ticket", "price")] - список цен акций
        """
        prices_list = [(ticket, self.get_ticket_price(ticket)) for ticket in stocks]
        return prices_list

    def get_ticket_price(self, ticket: str) -> int:
        """ Получить закупочную стоимость конкретной акции
        Args:
            ticket: название акции
        Returns:
            price: цена акции
        """
        #TODO: изменить на получение реальной цены
        price = random.randint(100, 1000)
        return price

    def get_TP_SL(self, ticket: str) -> (float, float):
        """ Получить значения для TakeProfit и StopLoss
        Args:
            ticket: название акции
        Returns:
            take_profit: значение тейк профит
            stop_loss
        """
        #TODO: изменить на получение реальных значений
        take_profit = random.randint(100, 1000)
        stop_loss = random.randint(100, 1000)
        return take_profit, stop_loss

    def count_stocks_values(self, prices_list: list, limit: int) -> list[tuple]:
        """ Посчитать соотношение акций к покупке
        Args:
            prices_list: список акций с ценами
            limit: максимальная сумма стоимости акций
        Returns:
            stocks_count: [("ticket", "count")] - количество акций к покупке
        """
        max_price_for_one_bucket = limit/len(prices_list)
        stocks_count = [(ticket_info[0], max_price_for_one_bucket//ticket_info[1])
                         for ticket_info in prices_list]
        return stocks_count

    def fill_stock_portfolio(self, stocks: list, limit: int) -> None:
        """ Заполнить портфель равномерно акциями на максимальную сумму
        Args:
            stocks: названия акций для покупки
            limit: максимальная сумма стоимости акций
        Returns:
            None
        """
        prices_list = self.get_prices(stocks)
        stocks_count = self.count_stocks_values(prices_list, limit)
        for stock_info in stocks_count:
            ticket_name = stock_info[0]
            ticket_count = stock_info[1]
            self.by(ticket=ticket_name, count=ticket_count, price=stock_info[1])
