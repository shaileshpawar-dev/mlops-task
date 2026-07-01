.PHONY: install install-dev run test docker-build docker-run clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

run:
	python run.py --input data.csv --config config.yaml \
		--output metrics.json --log-file run.log

test:
	pytest tests/ -v

docker-build:
	docker build -t mlops-task .

docker-run:
	docker run --rm mlops-task

clean:
	rm -rf __pycache__ tests/__pycache__ .pytest_cache
