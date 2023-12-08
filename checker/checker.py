import pandas as pd

from settings import Config
from datetime import datetime, timedelta
from stock_portfolio.agent import Agent


agent = Agent()



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
        df = pd.read_csv(self.file_path)
        new_counts = []
        for line in df.values():
            ticket_name = line[1]
            count = line[2]
            buying_price = line[3]
            take_profit = line[4]
            stop_loss = line[5]
            current_price = agent.get_ticket_price(ticket_name)
            if current_price>take_profit or current_price<stop_loss:
                agent.sell(ticket_name, count)
                new_counts.append(0)
                profit += current_price - buying_price
            else:
                new_counts.append(count)
        df['count'] = new_counts
        df.to_csv(self.file_path, index=False)
        agent.add_profit(profit)
