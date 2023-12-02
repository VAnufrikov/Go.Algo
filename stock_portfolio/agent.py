from stock_portfolio.portfolio import Stocks
from settings import DATE_START, DATE_END, LIMIT

import etna

class Agent:
    """Класс, который описывает поведение агента в среде биржи со своим портфелем"""

    def __int__(self):
        """Инициализация агента и его портфеля"""
        self.limit = LIMIT

    def predict(self):
        pass

    def by(self):
        """Реализация выставление тикета в стакан на покупку"""
        pass

    def sell(self):
        """Реализация выставление тикета в стакан на продажу"""
        pass

    def do_nofing(self):
        """Не выставляем тикет и просто ждем, возвращаем действие ничего не делаем"""
        pass

def run_agent():
    """Входом будет получение датасета за прошлые даты,
                выход решение о покупке или продаже """
    portfel = Stocks()
    agent = Agent()

    tiket = portfel.get_tiket()

    df_between_dates = portfel.get_time_baket(tiket, DATE_START, DATE_END)

    agent.sell()

    agent.by()

    agent.do_nofing()

