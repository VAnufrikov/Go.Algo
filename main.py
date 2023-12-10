import os
from uuid import uuid4

from settings import HORIZON
from stock_portfolio.agent_old import run_agent

if __name__ == "__main__":
    """Запускаем бота и устанавливаем нужную библиотеку etna"""
    # Нужно установить при первой загрузке
    # os.system("pip3 install --upgrade pip'")
    # os.system("pip3 install 'etna[torch]'")
    # os.system("pip3 install 'etna[auto]'")
    # os.system("pip3 install 'etna[statsforecast]'")
    # os.system("pip3 install 'etna[classification]'")
    # os.system("pip3 install 'etna[prophet]'")
    # os.system("pip3 install 'etna[all]'")

    agent_id = str(uuid4())
    while HORIZON > 1:
        # запускаем агента и возвращаем уменьшенный HORIZON
        HORIZON = run_agent(HORIZON, agent_id)
