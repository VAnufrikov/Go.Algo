from datetime import date
from upload_data.upload import read_data_stock, upload_data_from_moexalgo


SRCH_MOEX = 'data/ListingSecurityList.csv'
LIMIT = 10000
DATE_START = date(2023, 10, 1)
DATE_END = date(2023, 10, 10)

if __name__ == '__main__':
    listing = read_data_stock(SRCH_MOEX)

    # Получили пока что 1 TRADE_CODE по которому мы будем учится
    df = upload_data_from_moexalgo(listing['TRADE_CODE'][0], DATE_START, DATE_END)

    print(df.head())

