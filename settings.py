from datetime import date

LIMIT = 10000
HORIZON = 104  # горизонт в бакетах (дефолт 5 минут)
DATE_START = date(2023, 1, 1)
DATE_END = date(2023, 7, 10)


class Config:
    STOCKS_PATH = "stock_portfolio.csv"
    SRCH_MOEX = "data/ListingSecurityList.csv"
    SQL_DATABASE_PATH = "stock_portfolio.db"
