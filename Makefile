.PHONY: install data train test all

install:
	pip install -r requirements.txt -r requirements-dev.txt

data:
	python -m churn.data

train:
	python -m churn.pipeline

test:
	python -m pytest -q

all: data train test
