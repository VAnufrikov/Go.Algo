import pandas as pd

from settings import Config
from sqlite.client import SQLiteClient
from datetime import datetime, timedelta
from stock_portfolio.agent import Agent


agent = Agent()
sql_client = SQLiteClient(Config.SQL_DATABASE_PATH)
sql_client.connect()


class Checker:
    """Класс, запускающий раунд торгов (каждые 5 минут)"""

    def __init__(self):
        self.file_path = Config.STOCKS_PATH
        self.last_checking = datetime.now() - timedelta(minutes=5)


    def timer(self):
        while True:
            if datetime.now() > self.last_checking + timedelta(minutes=5):
                self.self.last_checking = datetime.now() 
                self.start_checking()
    

    def start_checking(self):
        profit=0
        orders = sql_client.select_all_orders()
        if orders:
            for line in orders:
                ticket_name = line[1]
                _, buying_price = sql_client.select_stock_count_and_price_in_portfolio(ticket_name)
                count = line[2]
                take_profit = line[3]
                stop_loss = line[4]
                current_price = agent.get_ticket_price(ticket_name)
                if current_price>take_profit or current_price<stop_loss:
                    sum = agent.sell(ticket_name, count)
                    profit += count * (current_price - buying_price)

            agent.add_profit(profit)
            agent.add_free_sum(sum)
