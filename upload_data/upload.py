from datetime import timedelta
import pandas as pd
import requests
import time



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

def get_dates(DATE_START, DATE_END):
    list_dates = []

    for result in perdelta(DATE_START, DATE_END):
        list_dates.append(result)

    list_dates.append(DATE_END)

    return list_dates


def upload_data_from_moexalgo(TRADE_CODE, DATE_START, DATE_END):
    """Получаем данные по TRADE_CODE из moexalgo"""
    dates = get_dates(DATE_START, DATE_END)

    tradestats = pd.DataFrame()
    for date in dates:
        url = f'https://iss.moex.com/iss/datashop/algopack/eq/tradestats/{TRADE_CODE}.csv?from={date}&till={date}&iss.only=data'
        df = pd.read_csv(url, sep=';', skiprows=1)
        tradestats = pd.concat([tradestats, df])
        time.sleep(0.5)
    return tradestats

def get_features_from_financialmodelingprep(ticket, token):
    params = {
         'period': 'annual',
         'apikey': token#'USOBt2DLiRjcewpnT4gvsqvqIm3uiwAy',
         }

    response = requests.get(
        f'https://financialmodelingprep.com/api/v3/income-statement/{ticket}',
        params=params,

    )
    return response.json()[0]
