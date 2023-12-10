python3 -m pip install black
python3 -m pip install isort
python3 -m black checker/checker.py
python3 -m black news_regressor/*.py
python3 -m black sqlite/*.py
python3 -m black stock_portfolio/*.py
python3 -m black upload_data/*.py
python3 -m black settings.py
python3 -m black main.py

python3 -m isort checker/checker.py
python3 -m isort news_regressor/*.py
python3 -m isort sqlite/*.py
python3 -m isort stock_portfolio/*.py
python3 -m isort upload_data/*.py
python3 -m isort settings.py
python3 -m isort main.py

python3 -m pip3 install --upgrade pip
python3 -m pip3 install news_regressor/requirements.txt
python3 -m pip3 install requirements.txt
python3 -m main.py
