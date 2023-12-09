from datetime import time, datetime

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

# def filter_data_stock(df,TRADE_CODE, DATE_START, DATE_END):
#
#     df['tradedate']= pd.to_datetime(df['tradedate'])
#     df = df[df['secid'] == TRADE_CODE]
#     df = df.loc[(df['tradedate'] >= datetime.strptime(str(DATE_START), '%Y-%m-%d'))]
#     df = df.loc[(df['tradedate'] <= datetime.strptime(str(DATE_END), '%Y-%m-%d'))]
#
#     return df

def upload_data_from_moexalgo(TRADE_CODE, DATE_START, DATE_END):
    """Получаем данные по TRADE_CODE из moexalgo"""
    tiket = Ticker(TRADE_CODE)
    tradestats = pd.DataFrame(tiket.tradestats(date=DATE_START, till_date=DATE_END))
    orderstats = pd.DataFrame(tiket.orderstats(date=DATE_START, till_date=DATE_END))
    obstats = pd.DataFrame(tiket.obstats(date=DATE_START, till_date=DATE_END))

    obstats['ts'] = pd.to_datetime(obstats['ts'], format='%Y-%m-%d %H:%M')
    orderstats['ts'] = pd.to_datetime(orderstats['ts'], format='%Y-%m-%d %H:%M')
    tradestats['ts'] = pd.to_datetime(tradestats['ts'], format='%Y-%m-%d %H:%M')

    # print('start upload_data_from_moexalgo')
    # df = filter_data_stock(pd.read_csv('data/obstats_2020.csv', sep=';'),TRADE_CODE, DATE_START, DATE_END)
    # df1 = filter_data_stock(pd.read_csv('data/obstats_2021.csv', sep=';'),TRADE_CODE, DATE_START, DATE_END)
    # df2 = filter_data_stock(pd.read_csv('data/obstats_2022.csv', sep=';'),TRADE_CODE, DATE_START, DATE_END)
    # df3 = filter_data_stock(pd.read_csv('data/obstats_2023.csv', sep=';'),TRADE_CODE, DATE_START, DATE_END)
    #
    # obstats = pd.concat([
    #     df, df1, df2, df3
    # ], ignore_index=True).reset_index(drop=True)
    obstats.rename(columns={'secid': 'ticker'}, inplace=True)
    # print('obstats ready')
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
    # print('orderstats ready')
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
    # print('tradestats ready')

    return tradestats, orderstats, obstats
