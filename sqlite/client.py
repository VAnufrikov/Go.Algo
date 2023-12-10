import sqlite3


class SQLiteClient:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_name)

    def close(self):
        self.conn.close()

    def execute(self, sql, params=None):
        if params is None:
            params = []
        cursor = self.conn.cursor()
        cursor.execute(sql, params)
        self.conn.commit()
        return cursor.fetchall()

    def create_stock_portfolio_table(self):
        create_table_query = """CREATE TABLE IF NOT EXISTS stock_portfolio (
            bot_id TEXT NOT NULL,
            ticket TEXT NOT NULL,
            count INTEGER NOT NULL,
            buying_price REAL NOT NULL)"""
        return self.execute(create_table_query)

    def create_orders_table(self):
        create_table_query = """CREATE TABLE IF NOT EXISTS orders (
            bot_id TEXT NOT NULL,
            order_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket TEXT NOT NULL,
            count INTEGER NOT NULL,
            take_profit REAL NOT NULL,
            stop_loss REAL NOT NULL)"""
        return self.execute(create_table_query)

    def insert_order(self, bot_id, ticket, count, take_profit, stop_loss):
        insert_query = f"""INSERT INTO orders (bot_id, ticket, count, take_profit, stop_loss)
            VALUES ('{bot_id}','{ticket}', {count}, {take_profit}, {stop_loss});"""
        return self.execute(insert_query)

    def insert_stock(self, bot_id, ticket, count, buying_price):
        current_count, current_price = self.select_stock_count_and_price_in_portfolio(bot_id=bot_id, ticket=ticket)
        new_mean_price = (current_price*current_count + count*buying_price)/(current_count+count)
        count += current_count
        remove_query = f"""DELETE FROM stock_portfolio WHERE ticket=="{ticket}" AND bot_id=="{bot_id}" """
        self.execute(remove_query)
        insert_query = f"""INSERT INTO stock_portfolio (bot_id, ticket, count, buying_price)
            VALUES ('{bot_id}', '{ticket}', {count}, {new_mean_price});"""
        return self.execute(insert_query)


    def sell_stock(self, bot_id, ticket, count):
        current_count, current_price = self.select_stock_count_and_price_in_portfolio(bot_id=bot_id, ticket=ticket)
        profit = current_price * count
        new_count = current_count - count
        remove_query = f"""DELETE FROM stock_portfolio WHERE ticket=="{ticket}" AND bot_id=="{bot_id}" """
        self.execute(remove_query)
        insert_query = f"""INSERT INTO stock_portfolio (bot_id, ticket, count, buying_price)
            VALUES ('{bot_id}', '{ticket}', {new_count}, {current_price});"""
        self.execute(insert_query)
        return profit

    def select_all_orders(self, bot_id):
        select_all_query= f"""SELECT * from orders WHERE bot_id=="{bot_id}" """
        return self.execute(select_all_query)

    def select_all_portfolio_stocks(self, bot_id):
        select_all_query= f"""SELECT * from stock_portfolio WHERE bot_id=="{bot_id}" """
        return self.execute(select_all_query)

    def select_stock_count_and_price_in_portfolio(self, bot_id, ticket):
        select_stock_count= f"""SELECT count, buying_price
            FROM stock_portfolio
            WHERE ticket=="{ticket}" AND bot_id=="{bot_id}" """
        count = self.execute(select_stock_count)
        if count:
            return count[0][0], count[0][1]
        else:
            return 0, 0

    def select_stock_count_in_orders(self, bot_id, ticket):
        select_stock_count= f"""SELECT count
            FROM orders
            WHERE ticket=="{ticket}" AND bot_id=="{bot_id}" """
        count = self.execute(select_stock_count)
        if count:
            return count[0][0]
        else:
            return 0

    def close_order(self, order_id, bot_id):
        remove_query = f"""DELETE FROM orders WHERE order_id={order_id} AND bot_id=="{bot_id}" """
        return self.execute(remove_query)
