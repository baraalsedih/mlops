.PHONY: install train test lint docker-build docker-run clean

install:
	pip install -r requirements.txt

train:
	python src/train.py

test:
	pytest tests/ -v --tb=short

lint:
	python -m py_compile src/train.py src/predict.py
	@echo "Syntax OK"

docker-build:
	docker build -t mlops-demo:latest .

docker-run:
	docker run --rm -v $(PWD)/models:/app/models mlops-demo:latest

docker-test:
	docker run --rm -v $(PWD)/models:/app/models mlops-demo:latest \
		python -m pytest tests/ -v --tb=short

clean:
	rm -rf models/__pycache__ src/__pycache__ tests/__pycache__ .pytest_cache
