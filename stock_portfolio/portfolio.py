from random import choice

class portfolio:
    """Создаем класс портфель в которой передаем по дефолту лимит портфеля"""
    def __init__(self, data=None):
        """Инициализация портфеля у агента"""
        self.data = data
        self.list_tikets = self.data['ticker'].unique().tolist()

    def get_tiket(self):
        """Получаем рандомный тикет для фокусирования бота"""
        return choice(self.list_tikets)

    def get_time_baket(self, tiket):
        """Получаем бакет по которому будем получать информацию о тикете"""
        data_tiket = self.data[self.data['secid'] == tiket].sort_values(by='tradedate', ascending=True)

        return data_tiket

    def get_price(self):
        """Получаем разброс цен на акции и возвращаем лучшую цену для покупки"""
        pass



