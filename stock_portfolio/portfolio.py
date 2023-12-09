from random import choice

from settings import Config
from upload_data.Ranking import ranking
from upload_data.upload import read_data_stock


def get_tiket():
    listing = read_data_stock(Config.SRCH_MOEX)
    ranking_listing = ranking(listing)
    """Получаем рандомный тикет для фокусирования бота"""
    return choice(ranking_listing)

