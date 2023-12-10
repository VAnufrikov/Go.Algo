from stock_portfolio.agent import run_agent
from settings import HORIZON
from uuid import uuid4
import os


if __name__ == '__main__':
    """Запускаем бота и устанавливаем нужную библиотеку etna"""
    # os.system("pip install --upgrade pip'")
    # os.system("pip install etna'")
    # os.system("pip install 'etna[all]'")
    agent_id = str(uuid4())
    while HORIZON > 1:
        # запускаем агента и возвращаем уменьшенный HORIZON
        HORIZON = run_agent(HORIZON, agent_id)
