install:
		pip install --upgrade pip &&\
			pip install -r requirements.txt

test:
		python -m pytest -vv --cov=data tests/test_data.py 

format:
		black *.py lib/*.py

lint:
		pylint --disable=R,C *.py lib/*.py tests/*.py

refactor: format lint

all: install lint test format