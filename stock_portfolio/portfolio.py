from random import choice


class portfolio:
    """Создаем класс портфель в которой передаем по дефолту лимит портфеля"""
    def __init__(self, limit=0, data=None):
        """Инициализация портфеля с дефолтной суммой портфеля"""
        self.limit = limit
        self.data = data
        self.list_tikets = self.data['ticker'].unique().tolist()

    def get_tiket(self):
        """Получаем рандомный тикет для фокусирования бота"""
        return choice(self.list_tikets)

    def get_time_baket(self):
        pass
    def get_price(self):
        """Получаем разброс цен на акции и возвращаем лучшую цену для покупки"""
        pass

    def by(self):
        tiket = self.get_tiket()
        """Реализация выставление тикета в стакан на покупку"""
        pass
    def sell(self):
        """Реализация выставление тикета в стакан на продажу"""
        pass

    def do_nofing(self):
        """Не выставляем тикет и просто ждем, возвращаем действие ничего не делаем"""
        pass
