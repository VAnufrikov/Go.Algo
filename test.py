from datetime import date

DATE_START = date(2023, 10, 1)
DATE_END = date(2023, 10, 2)

from upload_data.upload import upload_data_from_moexalgo

tradestats, orderstats, obstats = upload_data_from_moexalgo('AFKS',DATE_START,DATE_END)


print(tradestats.info())
print(orderstats.info())
print(obstats.info())