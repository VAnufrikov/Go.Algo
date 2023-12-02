from datetime import date
from upload_data.upload import read_data_stock, upload_data_from_moexalgo
from upload_data.Ranking import ranking
from stock_portfolio.portfolio import portfolio


SRCH_MOEX = 'data/ListingSecurityList.csv'
LIMIT = 10000
DATE_START = date(2023, 10, 1)
DATE_END = date(2023, 10, 10)

if __name__ == '__main__':
    listing = read_data_stock(SRCH_MOEX)

    # TODO сделать алгоритм ранжирования по всем акциям согласно критериям анализа акций
    # и записать это в функцию ranking() #
    # выходом этой функции будет DF список тикетов для загрузки за даты moexalgo #
    ranking(DATE_START, DATE_END)

    code = listing['TRADE_CODE'][0]

    # Получили пока что 1 TRADE_CODE по которому мы будем учится
    df = upload_data_from_moexalgo(code, DATE_START, DATE_END)

    portfel = portfolio(LIMIT, df)

    print(df.head())

