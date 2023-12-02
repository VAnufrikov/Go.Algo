from datetime import date, timedelta

import pandas as pd
from moexalgo import Market, Ticker
import time

LIMIT = 10000
DATE_START = date(2023, 10, 1)
DATE_END = date(2023, 10, 10)


def read_data_stock(srch):
    """Подготавливаем датасет со всеми компаниями ан московской бирже"""
    df = pd.read_csv('data/ListingSecurityList.csv', engine='python', encoding='cp1251', sep=';')

    df = df[['TRADE_CODE', 'EMITENT_FULL_NAME', 'INSTRUMENT_TYPE', 'LIST_SECTION', 'INSTRUMENT_CATEGORY', 'CURRENCY',
             'NOMINAL', 'ISSUE_AMOUNT']]

    # Берем только акции#
    df = df[(df['INSTRUMENT_TYPE'] == 'Акция обыкновенная') | (df['INSTRUMENT_TYPE'] == 'Акции иностранного эмитента')]

    return df.reset_index(drop=True)

def perdelta(start, end):
    curr = start
    while curr < end:
        yield curr
        curr += timedelta(days=1)

def get_dates():
    list_dates = []

    for result in perdelta(DATE_START, DATE_END):
        list_dates.append(result)

    list_dates.append(DATE_END)

    return list_dates


def upload_data_from_moexalgo(TRADE_CODE):
    """Получаем данные по TRADE_CODE из moexalgo"""
    dates = get_dates()

    tradestats = pd.DataFrame()
    for date in dates:
        url = f'https://iss.moex.com/iss/datashop/algopack/eq/tradestats/{TRADE_CODE}.csv?from={date}&till={date}&iss.only=data'
        df = pd.read_csv(url, sep=';', skiprows=1)
        tradestats = pd.concat([tradestats, df])
        time.sleep(0.5)

    return tradestats #tradestats.to_csv(f'tradestats_{TRADE_CODE}.csv', index=None)



if __name__ == '__main__':
    listing = read_data_stock('data/ListingSecurityList.csv')

    df = upload_data_from_moexalgo(listing['TRADE_CODE'][0])
    print(df.head())

