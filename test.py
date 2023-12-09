from datetime import datetime

import pandas as pd
from etna.datasets import TSDataset

from moexalgo import Ticker
from settings import DATE_START, DATE_END

tiket = Ticker('ALRS')
tradestats = pd.DataFrame(tiket.tradestats(date=DATE_START, till_date=DATE_END))


tradestats['fake_datetime'] = pd.to_datetime(tradestats['ts'],format='%Y-%m-%d %H:%M')

tradestats.rename(columns={'secid': 'ticker'}, inplace=True)
# print(tradestats.info ())

df = tradestats[['fake_datetime','ticker','pr_close']].reset_index(drop=True)

df.columns = ['timestamp','segment','target']

etna_df = TSDataset.to_dataset(df)

print(etna_df.info())