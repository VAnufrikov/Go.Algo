from random import choice

from settings import SRCH_MOEX
from upload_data.Ranking import ranking
from upload_data.upload import read_data_stock, upload_data_from_moexalgo


class Stocks:
    """Создаем класс портфель в которой передаем по дефолту лимит портфеля"""

    def __init__(self):
        """Инициализация портфеля у агента"""
        self.listing = read_data_stock(SRCH_MOEX)  # Читаем все тикеты
        # TODO сделать алгоритм ранжирования по всем акциям согласно критериям анализа акций
        # и записать это в функцию ranking() #
        # выходом этой функции будет DF список тикетов для загрузки за даты moexalgo #
        self.ranking_listing = ranking(self.listing)


    def get_tiket(self):
        """Получаем рандомный тикет для фокусирования бота"""
        return choice(self.ranking_listing)

    def get_time_baket(self, tiket, start, end):
        """Получаем бакет по которому будем получать информацию о тикете"""

        data = upload_data_from_moexalgo(tiket, start, end)

        data_tiket = data[data['secid'] == tiket].sort_values(by='tradedate', ascending=True)

        return data_tiket
