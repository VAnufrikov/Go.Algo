import itertools
from uuid import uuid4

from news_regressor import NewsRegressor


import etna
import random

import math
from datetime import timedelta

import pandas as pd
from matplotlib.lines import Line2D

from settings import Config, DATE_START, DATE_END, LIMIT
from datetime import datetime as dt
from etna.datasets.tsdataset import TSDataset

from sqlite.client import SQLiteClient

client = SQLiteClient(Config.SQL_DATABASE_PATH)
client.connect()

from etna.models import CatBoostMultiSegmentModel
from etna.pipeline import Pipeline
from etna.transforms import DensityOutliersTransform, TimeSeriesImputerTransform, LinearTrendTransform, TrendTransform, \
    LagTransform, DateFlagsTransform, FourierTransform, SegmentEncoderTransform, MeanTransform, HolidayTransform
from etna.metrics import SMAPE, MAE, MSE, smape

from upload_data.upload import upload_data_from_moexalgo
import numpy as np
from etna.analysis.utils import _prepare_axes
from etna.analysis.forecast.utils import _prepare_forecast_results, _select_quantiles, _validate_intersecting_segments

from typing import Dict, Sequence, Literal, Tuple, Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

from settings import Config
from upload_data.Ranking import ranking
from upload_data.upload import read_data_stock


def get_tiket():
    listing = read_data_stock(Config.SRCH_MOEX)

    ranking_listing = ranking(listing)
    """Получаем рандомный тикет для фокусирования бота"""
    return ranking_listing


def make_indicators(df: pd.DataFrame):
    """Simple Moving Average: https://site.financialmodelingprep.com/developer/docs/technical-intraday-sma
       Exponential Moving Average: https://site.financialmodelingprep.com/developer/docs/technical-intraday-ema
       Weighted Moving Average: https://site.financialmodelingprep.com/developer/docs/technical-intraday-wma
       Double EMA: https://site.financialmodelingprep.com/developer/docs/technical-intraday-dema

    """
    pr_high = df[['segment'] == 'pr_high'].reset_index(drop=True)
    pr_low = df[['segment'] == 'pr_low'].reset_index(drop=True)
    df = df[['segment'] == 'price'].reset_index(drop=True)

    ema = df['target'].ewm(span=50, adjust=False).mean()

    df['50-baket-EMA'] = ema
    df['200-baket-SMA'] = df['target'].rolling(window=200).mean()

    # Рассчет взвешенной скользящей средней (WMA)
    n = 50  # количество дней для взвешенной скользящей средней
    weights = pd.Series(range(1, n + 1))  # создание весов
    wma = df['target'].rolling(window=n).apply(lambda prices: (prices * weights).sum() / weights.sum(), raw=True)

    df['50-baket-WMA'] = wma

    # Считаем DEMA от ema индикатора
    dema = 2 * ema - ema.ewm(span=50, adjust=False).mean()
    df['50-baket-DEMA'] = 2 * ema - dema

    # Считаем TEMA от ema индикатора
    EMA1 = df['target'].ewm(span=50, adjust=False).mean()
    EMA2 = EMA1.ewm(span=50, adjust=False).mean()
    df['50-baket-TEMA'] = 3 * EMA1 - 3 * EMA2 + df['target'].ewm(span=50, adjust=False).mean()

    # williams
    high = pr_high['target'].rolling(window=14).max()
    low = pr_low['target'].rolling(window=14).min()
    df = df['target']
    williams = ((high - df) / (high - low)) * -100

    df['14-baket-WILLIAMS'] = williams

    # TODO Relative Strength Index
    # TODO Average Directional Index
    # TODO Standard Deviation

    return df


def get_min_max_support(df):
    """Расчет уровней поддежки по 52 бакетам"""

    shift = Config.HORIZON + 52

    df = df.iloc[-shift:].head(52)

    # make_indicators(df)
    # TODO нужно потом применить индикаторы для расчета уровней поддежки по 52 бакета

    max_support = df['target'].max()
    min_support = df['target'].min()

    return max_support, min_support


def run_agent(horizon, uuid):
        """Входом будет получение датасета за прошлые даты,
                    выход решение о покупке или продаже """
        agent = Agent(uuid)
        NR = NewsRegressor()
    """
    Тут мы должны получать по каждому тикету 
    предикт на стоимость портфеля и далее анализировать какую часть портфеля
    мы можем купить по всем позициям
    """

    # Пока не пройдемся по всем тикетам из листа тикетов
    for tiket in get_tiket():
        tradestats, orderstats, obstats = upload_data_from_moexalgo(tiket, DATE_START, DATE_END)

        inside = predict(tradestats, orderstats, obstats)

        df_price = inside[['segment'] == 'price'].reset_index(drop=True)

        last_price, date_time = agent.get_ticket_price(tiket, df_price)

        take_profit, stop_loss = agent.get_TP_SL(inside, last_price)

        predict_news = NR.predict(ticket=tiket, date=date_time)

        # # Решить мы будем покупать или продавать или ничего не делать
        #
        # # by
        # insert_order()
        # insert_stock()
        #
        # # sell
        # insert_order()
        # sell_stock()

        if inside == 1:
            agent.by()
            
        elif inside == -1:
            agent.sell()
        else:
            agent.do_nofing()
            
        new_horizon = horizon - 1
        if new_horizon == 1:
            agent.close_day()
        
        return new_horizon



def etna_predict(param, segment, horizon):
    """Предсказываем наши временные ряды по 3м сегментам"""
    ts = TSDataset(param, freq="T")

    train_ts, test_ts = ts.train_test_split(test_size=horizon)

    transforms = [
        DensityOutliersTransform(in_column="target", distance_coef=3.0),
        TimeSeriesImputerTransform(in_column="target", strategy="forward_fill"),
        LinearTrendTransform(in_column="target"),
        TrendTransform(in_column="target", out_column="trend"),
        LagTransform(in_column="target", lags=list(range(horizon, 122)), out_column="target_lag"),
        DateFlagsTransform(week_number_in_month=True, out_column="date_flag"),
        FourierTransform(period=360.25, order=6, out_column="fourier"),
        SegmentEncoderTransform(),
        MeanTransform(in_column=f"target_lag_{horizon}", window=12, seasonality=7),
        MeanTransform(in_column=f"target_lag_{horizon}", window=7),
        DateFlagsTransform(out_column="date_flags",
                           day_number_in_week=True,
                           day_number_in_month=True,
                           week_number_in_month=True,
                           is_weekend=True,
                           ),
        HolidayTransform(out_column="holiday", iso_code="RU"),
    ]

    pipeline, forecast_ts = etna_train(transforms, horizon, train_ts)

    # Сохраняем картинки
    save_plot_forecast(forecast_ts, test_ts, train_ts, pipeline, ts, segment)

    df = pd.concat([
        train_ts.to_pandas(True)[['timestamp', 'segment', 'target']],
        forecast_ts.to_pandas(True)[['timestamp', 'segment', 'target']]
    ], ignore_index=True).reset_index(drop=True)

    return df


def save_plot_forecast(forecast_ts, test_ts, train_ts, pipeline, ts, segment):
    """Сохраняем картинки для анализа обучения"""
    my_plot(ts, segments=[segment])

    my_plot_forecast(forecast_ts=forecast_ts, test_ts=test_ts, train_ts=train_ts, n_train_samples=50)

    # if segment == 'price':
    #     print(f'start_backtest {segment}')
    #     metrics_df, forecast_df, fold_info_df = pipeline.backtest(ts=ts, metrics=[MAE(), MSE(), SMAPE()], n_folds=5,
    #                                                           mode="expand", n_jobs=-1)
    #
    #     my_plot_backtest(forecast_df, ts)
    #     print(metrics_df.head(100))

    print(f'{segment} SMAPE = {SMAPE(y_true=test_ts, y_pred=forecast_ts)}')
    print(f'{segment} MAE = {MAE(y_true=test_ts, y_pred=forecast_ts)}')
    print(f'{segment} MSE = {MSE(y_true=test_ts, y_pred=forecast_ts)}')


def etna_train(transforms, horizon, train_ts):
    """Производим обучение модели по подготовленым данным"""

    model = CatBoostMultiSegmentModel()

    pipeline = Pipeline(model=model, transforms=transforms, horizon=horizon)

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
    tradestats['segment_pr_high'] = 'pr_high'
    tradestats['segment_pr_low'] = 'pr_low'
    orderstats['segment_vol'] = 'vol'
    obstats['segment_val'] = 'val'

    # Предиктим цену pr_high
    open = tradestats[['trade_datetime', 'segment_price_open', 'pr_high']]
    median_open = tradestats['pr_high'].median()

    open['pr_high'] = tradestats['pr_high'] - median_open

    open = make_fake_datetime(open)
    open.pop('trade_datetime')
    open = open[['fake_datetime', 'segment_price_open', 'pr_high']]

    # Предиктим цену pr_low
    pr_low = tradestats[['trade_datetime', 'segment_price_open', 'pr_low']]
    median_pr_low = tradestats['pr_low'].median()

    pr_low['pr_low'] = tradestats['pr_low'] - median_pr_low

    pr_low = make_fake_datetime(pr_low)
    pr_low.pop('trade_datetime')
    pr_low = pr_low[['fake_datetime', 'segment_price_open', 'pr_low']]

    # Предиктим цену закрытия
    tradestats = tradestats[['trade_datetime', 'segment_p', 'pr_close']]
    median_tradestats = tradestats['pr_close'].median()

    tradestats['pr_close'] = tradestats['pr_close'] - median_tradestats

    tradestats = make_fake_datetime(tradestats)
    tradestats.pop('trade_datetime')
    tradestats = tradestats[['fake_datetime', 'segment_p', 'pr_close']]

    # Предиктим обьем продаж по позиции
    orderstats['vol_true_put'] = orderstats['put_vol_s'] - orderstats['cancel_vol_s']
    median_orderstats = orderstats['vol_true_put'].median()

    orderstats['vol_true_put'] = orderstats['vol_true_put'] - median_orderstats

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
    res = tradestats.merge(orderstats).merge(obstats).merge(open).merge(pr_low)

    tradestats = res[['fake_datetime', 'segment_p', 'pr_close']]
    orderstats = res[['fake_datetime', 'segment_vol', 'vol_true_put']]
    obstats = res[['fake_datetime', 'segment_val', 'val_true_trend']]
    open = res[['fake_datetime', 'segment_price_open', 'pr_high']]
    pr_low = res[['fake_datetime', 'segment_price_open', 'pr_high']]

    tradestats.columns = ['timestamp', 'segment', 'target']
    orderstats.columns = ['timestamp', 'segment', 'target']
    obstats.columns = ['timestamp', 'segment', 'target']
    open.columns = ['timestamp', 'segment', 'target']
    pr_low.columns = ['timestamp', 'segment', 'target']

    df = pd.concat([
        tradestats, orderstats, obstats, open, pr_low
    ], ignore_index=True).reset_index(drop=True)

    return df[['timestamp', 'segment', 'target']], median_tradestats, median_orderstats, median_open, median_pr_low


def predict(trade, order, obs, horizon):
    """Тут мы прописываем логику обработки предикта etna:
    если будет рост отдаем 1
    если падение -1
    если так же то ничего не делаем
    """
    trade.rename(columns={'ts': 'trade_datetime'}, inplace=True)
    order.rename(columns={'ts': 'trade_datetime'}, inplace=True)
    obs.rename(columns={'ts': 'trade_datetime'}, inplace=True)

    # Сохраняем исходные trade_datetime для визуализации
    list_trade_datetime_tradestats = trade['trade_datetime'].to_list()

    df, median_tradestats, median_orderstats, median_open, median_pr_low = clean_df_for_etna(order, trade, obs)

    predict = pd.DataFrame(columns=['timestamp', 'segment', 'target', 'trade_datetime'])

    for segment in df['segment'].unique().tolist():
        df_temp = df[df['segment'] == segment]

        predict_temp = etna_predict(TSDataset.to_dataset(df_temp), segment, horizon)
        print(f'{segment} predict ready')


        predict_temp['trade_datetime'] = list_trade_datetime_tradestats

        if segment == 'tradestats':
            predict_temp['target'] = predict_temp['target'] + median_tradestats

        if segment == 'orderstats':
            predict_temp['target'] = predict_temp['target'] + median_orderstats

        if segment == 'pr_high':
            predict_temp['target'] = predict_temp['target'] + median_open

        if segment == 'pr_low':
            predict_temp['target'] = predict_temp['target'] + median_pr_low

        predict = pd.concat([
            predict,
            predict_temp
        ], ignore_index=True).reset_index(drop=True)

    print(predict.info())

    return predict


class Agent:
    """Класс, который описывает поведение агента в среде биржи со своим портфелем"""

    def __init__(self, uuid):
        """Инициализация агента и его портфеля"""
        self.limit = LIMIT
        self.profit = 0
        self.uuid = uuid


    def by(self, ticket, count, price):
        """Реализация выставление тикета в стакан на покупку"""
        take_profit, stop_loss = self.get_TP_SL(ticket)
        client.insert_order(bot_id=self.uuid, ticket=ticket, count=count, take_profit=take_profit, stop_loss=stop_loss)
        # TODO insert в портфель

    def sell(self, ticket_name, count):
        """Реализация выставление тикета в стакан на продажу"""
        # insert_order
        # TODO sell_stock в портфель
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

    def get_ticket_price(self, ticket: str, df) -> tuple[int, Any]:
        """ Получить закупочную стоимость конкретной акции
        Args:
            ticket: название акции
        Returns:
            price: цена акции
            :param ticket: Тикет по которому хотим узнать цену
            :param df: DataFrame - с ценой акции
        """
        shift = Config.HORIZON + 1

        df = df.iloc[-shift:].head(1)

        price = int(df['target'])

        date_time = df['trade_datetime']

        print(f'target price = {price}')

        return price, date_time

    def get_TP_SL(self, df, last_price):
        """ Получить значения для TakeProfit и StopLoss
        Args:
            :param last_price: последняя цена
            :param df: df цен на акцию
        Returns:
            take_profit: значение тейк профит
            stop_loss: значение стоплос
        """
        # TODO венуть функцию get_min_max_support для расчета уровней
        # max_support, min_support = get_min_max_support(df_price)

        df = df[['segment'] == 'price'].reset_index(drop=True)

        df = df.iloc[-Config.HORIZON:]

        max_predict = df['target'].max()

        # Вверх максимум летим на predict + 2%
        take_profit = df['target'].max() + max_predict - ((max_predict * 2) / 100)

        # Вниз максимум летим на -10 %
        stop_loss = last_price - ((last_price * 10) / 100)

        return take_profit, stop_loss

    def count_stocks_values(self, prices_list: list, limit: int) -> list[tuple]:
        """ Посчитать соотношение акций к покупке
        Args:
            prices_list: список акций с ценами
            limit: максимальная сумма стоимости акций
        Returns:
            stocks_count: [("ticket", "count")] - количество акций к покупке
        """
        max_price_for_one_bucket = limit / len(prices_list)
        stocks_count = [(ticket_info[0], max_price_for_one_bucket // ticket_info[1])
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

    def add_profit(self, profit: float) -> None:
        """ Добавить профит за раунд
        Args:
            profit: названия акций для покупки
        Returns:
            None
        """
        self.profit += profit

    def add_limit(self, sum: float) -> None:
        """ Добавить деньги за продажу
        Args:
            profit: названия акций для покупки
        Returns:
            None
        """
        self.limit += sum


    def close_day(self):
        """ в конце дня закрываем все сделки в портфеле.
        """
        stocks_in_profile = client.select_all_portfolio_stocks(bot_id=self.uuid)
        for stock in stocks_in_profile:
           count, price = client.select_stock_count_and_price_in_portfolio(bot_id=self.uuid, ticket=stock)
           client.sell_stock(bot_id=self.uuid, ticket=stock, count=count)
           result = count * price
           self.add_limit(result)


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


def my_plot_backtest(
        forecast_df: pd.DataFrame,
        ts: "TSDataset",
        segments: Optional[List[str]] = None,
        columns_num: int = 2,
        history_len: Union[int, Literal["all"]] = 0,
        figsize: Tuple[int, int] = (10, 5),
):
    """Plot targets and forecast for backtest pipeline.

    This function doesn't support intersecting folds.

    Parameters
    ----------
    forecast_df:
        forecasted dataframe with timeseries data
    ts:
        dataframe of timeseries that was used for backtest
    segments:
        segments to plot
    columns_num:
        number of subplots columns
    history_len:
        length of pre-backtest history to plot, if value is "all" then plot all the history
    figsize:
        size of the figure per subplot with one segment in inches

    Raises
    ------
    ValueError:
        if ``history_len`` is negative
    ValueError:
        if folds are intersecting
    """
    if history_len != "all" and history_len < 0:
        raise ValueError("Parameter history_len should be non-negative or 'all'")

    if segments is None:
        segments = sorted(ts.segments)

    fold_numbers = forecast_df[segments[0]]["fold_number"]
    _validate_intersecting_segments(fold_numbers)
    folds = sorted(set(fold_numbers))

    # prepare dataframes
    df = ts.df
    forecast_start = forecast_df.index.min()
    history_df = df[df.index < forecast_start]
    backtest_df = df[df.index >= forecast_start]

    # prepare colors
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_cycle = itertools.cycle(default_colors)
    lines_colors = {line_name: next(color_cycle) for line_name in ["history", "test", "forecast"]}

    _, ax = _prepare_axes(num_plots=len(segments), columns_num=columns_num, figsize=figsize)
    for i, segment in enumerate(segments):
        segment_backtest_df = backtest_df[segment]
        segment_history_df = history_df[segment]
        segment_forecast_df = forecast_df[segment]
        is_full_folds = set(segment_backtest_df.index) == set(segment_forecast_df.index)
        single_point_forecast = len(segment_backtest_df) == 1
        draw_only_lines = is_full_folds and not single_point_forecast

        # plot history
        if history_len == "all":
            plot_df = pd.concat((segment_history_df, segment_backtest_df))
        elif history_len > 0:
            plot_df = pd.concat((segment_history_df.tail(history_len), segment_backtest_df))
        else:
            plot_df = segment_backtest_df
        ax[i].plot(plot_df.index, plot_df.target, color=lines_colors["history"])

        for fold_number in folds:
            start_fold = fold_numbers[fold_numbers == fold_number].index.min()
            end_fold = fold_numbers[fold_numbers == fold_number].index.max()
            end_fold_exclusive = pd.date_range(start=end_fold, periods=2, freq=ts.freq)[1]

            # draw test
            backtest_df_slice_fold = segment_backtest_df[start_fold:end_fold_exclusive]
            ax[i].plot(backtest_df_slice_fold.index, backtest_df_slice_fold.target, color=lines_colors["test"])

            if draw_only_lines:
                # draw forecast
                forecast_df_slice_fold = segment_forecast_df[start_fold:end_fold_exclusive]
                ax[i].plot(forecast_df_slice_fold.index, forecast_df_slice_fold.target, color=lines_colors["forecast"])
            else:
                forecast_df_slice_fold = segment_forecast_df[start_fold:end_fold]
                backtest_df_slice_fold = backtest_df_slice_fold.loc[forecast_df_slice_fold.index]

                # draw points on test
                ax[i].scatter(backtest_df_slice_fold.index, backtest_df_slice_fold.target, color=lines_colors["test"])

                # draw forecast
                ax[i].scatter(
                    forecast_df_slice_fold.index, forecast_df_slice_fold.target, color=lines_colors["forecast"]
                )

            # draw borders of current fold
            opacity = 0.075 * ((fold_number + 1) % 2) + 0.075
            ax[i].axvspan(
                start_fold,
                end_fold_exclusive,
                alpha=opacity,
                color="skyblue",
            )

        # plot legend
        legend_handles = [
            Line2D([0], [0], marker="o", color=color, label=label) for label, color in lines_colors.items()
        ]
        ax[i].legend(handles=legend_handles)

        ax[i].set_title(segment)
        ax[i].tick_params("x", rotation=45)

    _.savefig(f'png_traiding/plot_backtest_{segment}.png')  # save the figure to file
    plt.close(_)
