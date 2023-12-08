from datetime import date

class Config:
    STOCKS_PATH = "stock_portfolio.csv"
    SRCH_MOEX = 'data/ListingSecurityList.csv'
    LIMIT = 10000
    DATE_START = date(2023, 10, 1)
    DATE_END = date(2023, 10, 10)
