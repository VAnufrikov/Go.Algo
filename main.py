import pandas as pd
from moexalgo import Market, Ticker

LIMIT = 10000
DATE_START = '2021-01-01'
DATE_END = '2021-01-02'


def read_data_stock(srch):
    """Подготавливаем датасет со всеми компаниями ан московской бирже"""
    df = pd.read_csv('data/ListingSecurityList.csv', engine='python', encoding='cp1251', sep=';')

    df = df[['TRADE_CODE', 'EMITENT_FULL_NAME', 'INSTRUMENT_TYPE', 'LIST_SECTION', 'INSTRUMENT_CATEGORY', 'CURRENCY']]

    # Берем только акции#
    df = df[(df['INSTRUMENT_TYPE'] == 'Акция обыкновенная') | (df['INSTRUMENT_TYPE'] == 'Акции иностранного эмитента')]

    return df


if __name__ == '__main__':
    listing = read_data_stock('data/ListingSecurityList.csv')

    print(listing.head())
