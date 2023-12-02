from stock_portfolio.portfolio import portfolio
from settings import LIMIT
from stock_portfolio.portfolio import portfolio


class Agent:
    """Класс, который описывает поведение агента в среде биржи со своим портфелем"""

    def __int__(self, limit=0):
        """Инициализация агента и его портфеля"""
        self.limit = LIMIT
        self.portfel = portfolio()

    def by(self):
        """Реализация выставление тикета в стакан на покупку"""
        # self.get_time_baket(tiket)
        pass
    def sell(self):
        """Реализация выставление тикета в стакан на продажу"""
        pass

    def do_nofing(self):
        """Не выставляем тикет и просто ждем, возвращаем действие ничего не делаем"""
        pass

    def think(self):
        """Входом будет получение датасета за прошлые даты,
                выход решение о покупке или продаже """
        tiket = self.portfel.get_tiket()

        data_between_dates = self.portfel.get_time_baket(tiket)


        self.sell()

        self.by()

        self.do_nofing()