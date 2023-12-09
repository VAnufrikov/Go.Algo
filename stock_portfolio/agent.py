
import etna
import random

import math
from datetime import timedelta

import pandas as pd

from settings import Config, LIMIT
from datetime import datetime as dt
from etna.datasets.tsdataset import TSDataset

from stock_portfolio.portfolio import Stocks

from etna.models import CatBoostMultiSegmentModel
from etna.pipeline import Pipeline
from etna.transforms import DensityOutliersTransform, TimeSeriesImputerTransform, LinearTrendTransform, TrendTransform, \
    LagTransform, DateFlagsTransform, FourierTransform, SegmentEncoderTransform, MeanTransform
from etna.analysis import plot_forecast
from etna.metrics import SMAPE

from settings import DATE_START, DATE_END, LIMIT, HORIZON
from stock_portfolio.portfolio import get_tiket
from upload_data.upload import upload_data_from_moexalgo
import numpy as np
from etna.analysis.utils import _prepare_axes
from etna.analysis.forecast.utils import _prepare_forecast_results, _select_quantiles

from typing import Dict, Sequence
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")



def run_agent():
    """Входом будет получение датасета за прошлые даты,
                выход решение о покупке или продаже """
    agent = Agent()
    """
    Тут мы должны получать по каждому тикету 
    предикт на стоимость портфеля и далее анализировать какую часть портфеля
    мы можем купить по всем позициям
    """
    
    tradestats, orderstats, obstats = upload_data_from_moexalgo(get_tiket(), DATE_START, DATE_END)

    inside = predict(tradestats, orderstats, obstats)

    if inside == 1:
        agent.by()

    elif inside == -1:
        agent.sell()
    else:
        agent.do_nofing()


def etna_predict(param):
    """Предсказываем наши временные ряды по 3м сегментам"""
    ts = TSDataset(param, freq="T")

    my_plot(ts, segments=["price"])
    my_plot(ts, segments=["vol"])
    my_plot(ts, segments=["val"])

    train_ts, test_ts = ts.train_test_split(test_size=HORIZON)

    transforms = [
        DensityOutliersTransform(in_column="target", distance_coef=3.0),
        TimeSeriesImputerTransform(in_column="target", strategy="forward_fill"),
        LinearTrendTransform(in_column="target"),
        TrendTransform(in_column="target", out_column="trend"),
        LagTransform(in_column="target", lags=list(range(HORIZON, 122)), out_column="target_lag"),
        DateFlagsTransform(week_number_in_month=True, out_column="date_flag"),
        FourierTransform(period=360.25, order=6, out_column="fourier"),
        SegmentEncoderTransform(),
        MeanTransform(in_column=f"target_lag_{HORIZON}", window=12, seasonality=7),
        MeanTransform(in_column=f"target_lag_{HORIZON}", window=7),
    ]

    pipeline, forecast_ts = etna_train(transforms, HORIZON, train_ts)

    # Сохраняем картинку
    my_plot_forecast(forecast_ts=forecast_ts, test_ts=test_ts, train_ts=train_ts, n_train_samples=50)

    smape = SMAPE()
    print(smape(y_true=test_ts, y_pred=forecast_ts))


def etna_train(transforms, HORIZON, train_ts):
    """Производим обучение модели по подготовленым данным"""

    model = CatBoostMultiSegmentModel()

    pipeline = Pipeline(model=model, transforms=transforms, horizon=HORIZON)

    pipeline.fit(train_ts)

    return pipeline, pipeline.forecast()


def make_fake_datetime(df):
    """Создаем непрерывный временной ряд для etna"""
    start = df['trade_datetime'][0]

    list_datetime = []

    for i in range(len(df['trade_datetime'])):
        if i == 0:
            list_datetime.append(start)
        else:
            start = list_datetime[i - 1]
            list_datetime.append(start + timedelta(minutes=1))

    df['fake_datetime'] = list_datetime
    return df



def clean_df_for_etna(orderstats, tradestats, obstats):
    tradestats['segment_p'] = 'price'
    orderstats['segment_vol'] = 'vol'
    obstats['segment_val'] = 'val'

    # Предиктим цену закрытия
    tradestats = tradestats[['trade_datetime', 'segment_p', 'pr_close']]
    tradestats = make_fake_datetime(tradestats)
    tradestats.pop('trade_datetime')
    tradestats = tradestats[['fake_datetime', 'segment_p', 'pr_close']]

    # Предиктим обьем продаж по позиции
    orderstats['vol_true_put'] = orderstats['put_vol_s'] - orderstats['cancel_vol_s']
    orderstats = orderstats[['trade_datetime', 'segment_vol', 'vol_true_put']]
    orderstats = make_fake_datetime(orderstats)
    orderstats.pop('trade_datetime')
    orderstats = orderstats[['fake_datetime', 'segment_vol', 'vol_true_put']]

    # Предиктим количество заявок которые доступно если все смогут и купить и продать
    # По сути смотрим тренд по заявкам
    obstats['val_true_trend'] = obstats['val_b'] - obstats['val_s']
    obstats = obstats[['trade_datetime', 'segment_val', 'val_true_trend']]
    obstats = make_fake_datetime(obstats)
    obstats.pop('trade_datetime')
    obstats = obstats[['fake_datetime', 'segment_val', 'val_true_trend']]

    # На случай если у нас не совпадают df.shape
    res = tradestats.merge(orderstats).merge(obstats)

    tradestats = res[['fake_datetime', 'segment_p', 'pr_close']]
    orderstats = res[['fake_datetime', 'segment_vol', 'vol_true_put']]
    obstats = res[['fake_datetime', 'segment_val', 'val_true_trend']]

    tradestats.columns = ['timestamp', 'segment', 'target']
    orderstats.columns = ['timestamp', 'segment', 'target']
    obstats.columns = ['timestamp', 'segment', 'target']

    df = pd.concat([
        tradestats, orderstats, obstats
    ], ignore_index=True).reset_index(drop=True)

    return df[['timestamp', 'segment', 'target']]


def predict(trade, order, obs):
    """Тут мы прописываем логику обработки предикта etna:
    если будет рост отдаем 1
    если падение -1
    если так же то ничего не делаем
    """
    # print(trade.head())

    trade.rename(columns={'ts': 'trade_datetime'}, inplace=True)
    order.rename(columns={'ts': 'trade_datetime'}, inplace=True)
    obs.rename(columns={'ts': 'trade_datetime'}, inplace=True)

    # Создаем норм таймлайн для 3х датасетов
    # trade['trade_datetime'] = pd.to_datetime(
    #     trade.tradedate.astype(str) + ' ' + trade.tradetime.astype(str))
    # order['trade_datetime'] = pd.to_datetime(
    #     order.tradedate.astype(str) + ' ' + order.tradetime.astype(str))
    # obs['trade_datetime'] = pd.to_datetime(
    #     obs.tradedate.astype(str) + ' ' + obs.tradetime.astype(str))

    # Сохраняем исходные trade_datetime для визуализации
    list_trade_datetime_tradestats = trade['trade_datetime'].to_list()
    list_trade_datetime_orderstats = order['trade_datetime'].to_list()
    list_trade_datetime_obstats = obs['trade_datetime'].to_list()

    df = clean_df_for_etna(order, trade, obs)

    # df.to_csv('df_before.csv', sep=';')
    #
    # df_1 = TSDataset.to_dataset(df)
    # ts = TSDataset(df_1, freq="T")
    #
    # ts.to_pandas(True)[['timestamp', 'segment', 'target']].to_csv('df_after.csv', sep=';')

    predict = etna_predict(TSDataset.to_dataset(df))

    """Тут нужно просписать логику сравнения текущей цены и цены в будующем

    если дороже покупаем:  1

    если ниже шортим: -1 

    если такая же возращем 0 

    """
    return 0


class Agent:
    """Класс, который описывает поведение агента в среде биржи со своим портфелем"""

    def __int__(self):
        """Инициализация агента и его портфеля"""
        self.limit = LIMIT

    def by(self, ticket, count, price):
        """Реализация выставление тикета в стакан на покупку"""
        take_profit, stop_loss = self.get_TP_SL(ticket)
        datetime = str(dt.now())
        str_line = f"{datetime}|{ticket}|{price}|{count}|{take_profit}|{stop_loss}\n"
        with open(Config.STOCKS_PATH, 'a') as fout:
            fout.write(str_line)

    def sell(self):
        """Реализация выставление тикета в стакан на продажу"""
        pass

    def do_nofing(self):
        """Не выставляем тикет и просто ждем, возвращаем действие ничего не делаем"""
        pass


    def get_prices(self, stocks: list) -> list[tuple]:
        """ Получить закупочную стоимость акций, которые необходимо купить
        Args:
            stocks: список акций, которые необходимо купить.
        Returns:
            prices_list: [("ticket", "price")] - список цен акций
        """
        prices_list = [(ticket, self.get_ticket_price(ticket)) for ticket in stocks]
        return prices_list

    def get_ticket_price(self, ticket: str) -> int:
        """ Получить закупочную стоимость конкретной акции
        Args:
            ticket: название акции
        Returns:
            price: цена акции
        """
        #TODO: изменить на получение реальной цены
        price = random.randint(100, 1000)
        return price

    def get_TP_SL(self, ticket: str) -> (float, float):
        """ Получить значения для TakeProfit и StopLoss
        Args:
            ticket: название акции
        Returns:
            take_profit: значение тейк профит
            stop_loss
        """
        #TODO: изменить на получение реальных значений
        take_profit = random.randint(100, 1000)
        stop_loss = random.randint(100, 1000)
        return take_profit, stop_loss

    def count_stocks_values(self, prices_list: list, limit: int) -> list[tuple]:
        """ Посчитать соотношение акций к покупке
        Args:
            prices_list: список акций с ценами
            limit: максимальная сумма стоимости акций
        Returns:
            stocks_count: [("ticket", "count")] - количество акций к покупке
        """
        max_price_for_one_bucket = limit/len(prices_list)
        stocks_count = [(ticket_info[0], max_price_for_one_bucket//ticket_info[1])
                         for ticket_info in prices_list]
        return stocks_count

    def fill_stock_portfolio(self, stocks: list, limit: int) -> None:
        """ Заполнить портфель равномерно акциями на максимальную сумму
        Args:
            stocks: названия акций для покупки
            limit: максимальная сумма стоимости акций
        Returns:
            None
        """
        prices_list = self.get_prices(stocks)
        stocks_count = self.count_stocks_values(prices_list, limit)
        for stock_info in stocks_count:
            ticket_name = stock_info[0]
            ticket_count = stock_info[1]
            self.by(ticket=ticket_name, count=ticket_count, price=stock_info[1])

def my_plot_forecast(
        forecast_ts: Union["TSDataset", List["TSDataset"], Dict[str, "TSDataset"]],
        test_ts: Optional["TSDataset"] = None,
        train_ts: Optional["TSDataset"] = None,
        segments: Optional[List[str]] = None,
        n_train_samples: Optional[int] = None,
        columns_num: int = 2,
        figsize: Tuple[int, int] = (10, 5),
        prediction_intervals: bool = False,
        quantiles: Optional[List[float]] = None,
):
    """
    Plot of prediction for forecast pipeline.

    Parameters
    ----------
    forecast_ts:
        there are several options:

        #. Forecasted TSDataset with timeseries data, single-forecast mode

        #. List of forecasted TSDatasets, multi-forecast mode

        #. Dictionary with forecasted TSDatasets, multi-forecast mode

    test_ts:
        TSDataset with timeseries data
    train_ts:
        TSDataset with timeseries data
    segments:
        segments to plot; if not given plot all the segments from ``forecast_df``
    n_train_samples:
        length of history of train to plot
    columns_num:
        number of graphics columns
    figsize:
        size of the figure per subplot with one segment in inches
    prediction_intervals:
        if True prediction intervals will be drawn
    quantiles:
        List of quantiles to draw, if isn't set then quantiles from a given dataset will be used.
        In multi-forecast mode, only quantiles present in each forecast will be used.

    Raises
    ------
    ValueError:
        if the format of ``forecast_ts`` is unknown
    """
    forecast_results = _prepare_forecast_results(forecast_ts)
    num_forecasts = len(forecast_results.keys())

    if segments is None:
        unique_segments = set()
        for forecast in forecast_results.values():
            unique_segments.update(forecast.segments)
        segments = list(unique_segments)

    _, ax = _prepare_axes(num_plots=len(segments), columns_num=columns_num, figsize=figsize)

    if prediction_intervals:
        quantiles = _select_quantiles(forecast_results, quantiles)

    if train_ts is not None:
        train_ts.df.sort_values(by="timestamp", inplace=True)
    if test_ts is not None:
        test_ts.df.sort_values(by="timestamp", inplace=True)

    for i, segment in enumerate(segments):
        if train_ts is not None:
            segment_train_df = train_ts[:, segment, :][segment]
        else:
            segment_train_df = pd.DataFrame(columns=["timestamp", "target", "segment"])

        if test_ts is not None:
            segment_test_df = test_ts[:, segment, :][segment]
        else:
            segment_test_df = pd.DataFrame(columns=["timestamp", "target", "segment"])

        if n_train_samples is None:
            plot_df = segment_train_df
        elif n_train_samples != 0:
            plot_df = segment_train_df[-n_train_samples:]
        else:
            plot_df = pd.DataFrame(columns=["timestamp", "target", "segment"])

        if (train_ts is not None) and (n_train_samples != 0):
            ax[i].plot(plot_df.index.values, plot_df.target.values, label="train")
        if test_ts is not None:
            ax[i].plot(segment_test_df.index.values, segment_test_df.target.values, color="purple", label="test")

        # plot forecast plot for each of given forecasts
        quantile_prefix = "target_"
        for forecast_name, forecast in forecast_results.items():
            legend_prefix = f"{forecast_name}: " if num_forecasts > 1 else ""

            segment_forecast_df = forecast[:, segment, :][segment].sort_values(by="timestamp")
            line = ax[i].plot(
                segment_forecast_df.index.values,
                segment_forecast_df.target.values,
                linewidth=1,
                label=f"{legend_prefix}forecast",
            )
            forecast_color = line[0].get_color()

            # draw prediction intervals from outer layers to inner ones
            if prediction_intervals and quantiles is not None:
                alpha = np.linspace(0, 1 / 2, len(quantiles) // 2 + 2)[1:-1]
                for quantile_idx in range(len(quantiles) // 2):
                    # define upper and lower border for this iteration
                    low_quantile = quantiles[quantile_idx]
                    high_quantile = quantiles[-quantile_idx - 1]
                    values_low = segment_forecast_df[f"{quantile_prefix}{low_quantile}"].values
                    values_high = segment_forecast_df[f"{quantile_prefix}{high_quantile}"].values
                    # if (low_quantile, high_quantile) is the smallest interval
                    if quantile_idx == len(quantiles) // 2 - 1:
                        ax[i].fill_between(
                            segment_forecast_df.index.values,
                            values_low,
                            values_high,
                            facecolor=forecast_color,
                            alpha=alpha[quantile_idx],
                            label=f"{legend_prefix}{low_quantile}-{high_quantile}",
                        )
                    # if there is some interval inside (low_quantile, high_quantile) we should plot around it
                    else:
                        low_next_quantile = quantiles[quantile_idx + 1]
                        high_prev_quantile = quantiles[-quantile_idx - 2]
                        values_next = segment_forecast_df[f"{quantile_prefix}{low_next_quantile}"].values
                        ax[i].fill_between(
                            segment_forecast_df.index.values,
                            values_low,
                            values_next,
                            facecolor=forecast_color,
                            alpha=alpha[quantile_idx],
                            label=f"{legend_prefix}{low_quantile}-{high_quantile}",
                        )
                        values_prev = segment_forecast_df[f"{quantile_prefix}{high_prev_quantile}"].values
                        ax[i].fill_between(
                            segment_forecast_df.index.values,
                            values_high,
                            values_prev,
                            facecolor=forecast_color,
                            alpha=alpha[quantile_idx],
                        )
                # when we can't find pair quantile, we plot it separately
                if len(quantiles) % 2 != 0:
                    remaining_quantile = quantiles[len(quantiles) // 2]
                    values = segment_forecast_df[f"{quantile_prefix}{remaining_quantile}"].values
                    ax[i].plot(
                        segment_forecast_df.index.values,
                        values,
                        "--",
                        color=forecast_color,
                        label=f"{legend_prefix}{remaining_quantile}",
                    )
        ax[i].set_title(segment)
        ax[i].tick_params("x", rotation=45)
        ax[i].legend(loc="upper left")

        _.savefig(f'png_traiding/{segment}.png')  # save the figure to file
        plt.close(_)


def my_plot(
        df,
        n_segments: int = 10,
        column: str = "target",
        segments: Optional[Sequence[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        seed: int = 1,
        figsize: Tuple[int, int] = (15, 7),
    ):
        """Plot of random or chosen segments.

        Parameters
        ----------
        n_segments:
            number of random segments to plot
        column:
            feature to plot
        segments:
            segments to plot
        seed:
            seed for local random state
        start:
            start plot from this timestamp
        end:
            end plot at this timestamp
        figsize:
            size of the figure per subplot with one segment in inches
        """
        if segments is None:
            segments = df['segment'].unique().tolist()
            k = min(n_segments, len(segments))
        else:
            k = len(segments)

        columns_num = min(2, k)
        rows_num = math.ceil(k / columns_num)
        start = df.index.min() if start is None else pd.Timestamp(start)
        end = df.index.max() if end is None else pd.Timestamp(end)

        figsize = (figsize[0] * columns_num, figsize[1] * rows_num)
        _, ax = plt.subplots(rows_num, columns_num, figsize=figsize, squeeze=False)
        ax = ax.ravel()
        rnd_state = np.random.RandomState(seed)
        for i, segment in enumerate(sorted(rnd_state.choice(segments, size=k, replace=False))):
            df_slice = df[start:end, segment, column]  # type: ignore
            ax[i].plot(df_slice.index, df_slice.values)
            ax[i].set_title(segment)
            ax[i].grid()

        _.savefig(f'png_traiding/plot_{segment}.png')  # save the figure to file
        plt.close(_)
