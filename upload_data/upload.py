from datetime import timedelta
import pandas as pd
import time
from moexalgo import Ticker



def read_data_stock(srch):
    """Подготавливаем датасет со всеми компаниями ан московской бирже"""
    df = pd.read_csv('data/ListingSecurityList.csv', engine='python', encoding='cp1251', sep=';')

    df = df[['TRADE_CODE', 'EMITENT_FULL_NAME', 'INSTRUMENT_TYPE', 'LIST_SECTION', 'INSTRUMENT_CATEGORY', 'CURRENCY',
             'NOMINAL', 'ISSUE_AMOUNT']]

    # Берем только акции#
    df = df[(df['INSTRUMENT_TYPE'] == 'Акция обыкновенная') | (df['INSTRUMENT_TYPE'] == 'Акции иностранного эмитента')]

    return df.reset_index(drop=True)


def upload_data_from_moexalgo(TRADE_CODE, DATE_START, DATE_END):
    """Получаем данные по TRADE_CODE из moexalgo"""

    tiket = Ticker(TRADE_CODE)
    orderstats = tiket.orderstats(date=DATE_START, till_date=DATE_END)
    tradestats = tiket.tradestats(date=DATE_START, till_date=DATE_END)
    obstats = tiket.obstats(date=DATE_START, till_date=DATE_END)

    return tradestats, orderstats, obstats