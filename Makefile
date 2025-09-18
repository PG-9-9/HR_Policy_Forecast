.PHONY: setup download rag_index train api docker-build docker-run

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')"

download:
	python scripts/download_data.py

rag_index:
	python rag/build_index.py

train:
	python forecasting/train_tft.py

api:
	uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

docker-build:
	docker build -t mm-hr-policy-forecast -f docker/Dockerfile .

docker-run:
	docker run -p 8080:8080 --env-file .env -v $(PWD)/data:/app/data mm-hr-policy-forecast
