import requests
import pandas as pd
from moexalgo import Ticker


def read_data_stock(srch):
    """Подготавливаем датасет со всеми компаниями ан московской бирже"""
    df = pd.read_csv(srch, engine='python', encoding='cp1251', sep=';')

    df = df[['TRADE_CODE', 'EMITENT_FULL_NAME', 'INSTRUMENT_TYPE', 'LIST_SECTION', 'INSTRUMENT_CATEGORY', 'CURRENCY',
             'NOMINAL', 'ISSUE_AMOUNT']]

    # Берем только акции#
    df = df[(df['INSTRUMENT_TYPE'] == 'Акция обыкновенная') | (df['INSTRUMENT_TYPE'] == 'Акции иностранного эмитента')]

    return df.reset_index(drop=True)


def upload_data_from_moexalgo(TRADE_CODE, DATE_START, DATE_END):
    """Получаем данные по TRADE_CODE из moexalgo"""
    tiket = Ticker(TRADE_CODE)

    print('start upload_data_from_moexalgo')

    tradestats = pd.DataFrame(tiket.tradestats(date=DATE_START, till_date=DATE_END))
    print('tradestats ready')

    orderstats = pd.DataFrame(tiket.orderstats(date=DATE_START, till_date=DATE_END))
    print('orderstats ready')

    obstats = pd.DataFrame(tiket.obstats(date=DATE_START, till_date=DATE_END))
    print('obstats ready')

    obstats['ts'] = pd.to_datetime(obstats['ts'], format='%Y-%m-%d %H:%M')
    orderstats['ts'] = pd.to_datetime(orderstats['ts'], format='%Y-%m-%d %H:%M')
    tradestats['ts'] = pd.to_datetime(tradestats['ts'], format='%Y-%m-%d %H:%M')


    # df = filter_data_stock(pd.read_csv('data/obstats_2020.csv', sep=';'),TRADE_CODE, DATE_START, DATE_END)
    # df1 = filter_data_stock(pd.read_csv('data/obstats_2021.csv', sep=';'),TRADE_CODE, DATE_START, DATE_END)
    # df2 = filter_data_stock(pd.read_csv('data/obstats_2022.csv', sep=';'),TRADE_CODE, DATE_START, DATE_END)
    # df3 = filter_data_stock(pd.read_csv('data/obstats_2023.csv', sep=';'),TRADE_CODE, DATE_START, DATE_END)
    #
    # obstats = pd.concat([
    #     df, df1, df2, df3
    # ], ignore_index=True).reset_index(drop=True)
    obstats.rename(columns={'secid': 'ticker'}, inplace=True)

    #
    # df = filter_data_stock(pd.read_csv('data/orderstats_2020.csv', sep=';'),TRADE_CODE, DATE_START, DATE_END)
    # df1 = filter_data_stock(pd.read_csv('data/orderstats_2021.csv', sep=';'),TRADE_CODE, DATE_START, DATE_END)
    # df2 = filter_data_stock(pd.read_csv('data/orderstats_2022.csv', sep=';'),TRADE_CODE, DATE_START, DATE_END)
    # df3 = filter_data_stock(pd.read_csv('data/orderstats_2023.csv', sep=';'),TRADE_CODE, DATE_START, DATE_END)
    #
    # orderstats = pd.concat([
    #     df, df1, df2, df3
    # ], ignore_index=True).reset_index(drop=True)
    orderstats.rename(columns={'secid': 'ticker'}, inplace=True)

    #
    # df = filter_data_stock(pd.read_csv('data/tradestats_2021.csv', sep=';'),TRADE_CODE, DATE_START, DATE_END)
    # df1 = filter_data_stock(pd.read_csv('data/tradestats_2022.csv', sep=';'),TRADE_CODE, DATE_START, DATE_END)
    # df2 = filter_data_stock(pd.read_csv('data/tradestats_2020.csv', sep=';'),TRADE_CODE, DATE_START, DATE_END)
    # df3 = filter_data_stock(pd.read_csv('data/tradestats_2023.csv', sep=';'),TRADE_CODE, DATE_START, DATE_END)
    #
    # tradestats = pd.concat([
    #     df, df1, df2, df3
    # ], ignore_index=True).reset_index(drop=True)
    tradestats.rename(columns={'secid': 'ticker'}, inplace=True)


# <<<<<<< checker
# def upload_data_from_moexalgo(TRADE_CODE, DATE_START, DATE_END):
#     """Получаем данные по TRADE_CODE из moexalgo"""
#     dates = get_dates(DATE_START, DATE_END)

#     tradestats = pd.DataFrame()
#     for date in dates:
#         url = f'https://iss.moex.com/iss/datashop/algopack/eq/tradestats/{TRADE_CODE}.csv?from={date}&till={date}&iss.only=data'
#         df = pd.read_csv(url, sep=';', skiprows=1)
#         tradestats = pd.concat([tradestats, df])
#         time.sleep(0.5)
#     return tradestats
# =======

    return tradestats, orderstats, obstats

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
