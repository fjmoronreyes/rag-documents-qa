# Virtual environment setup with Poetry
.PHONY: poetry-init
poetry-init:
	rm -rf .rag-documents && \
	python3 -m venv .rag-documents && \
	. .rag-documents/bin/activate && \
	poetry config virtualenvs.create false && \
	poetry install --with dev && \
	echo "Virtual environment created in .rag-documents. To activate it, run:" && \
	echo "source .rag-documents/bin/activate"

.PHONY: poetry-init-win
poetry-init-win:
	rmdir /s /q .rag-documents || exit 0
	python -m venv .rag-documents
	poetry config virtualenvs.create false
	poetry install --with dev
	@echo Virtual environment created in .rag-documents. To activate it, run:
	@echo .\\.rag-documents\\Scripts\\activate

# Code formatting & linting with Ruff
.PHONY: format
format:
	ruff check . --fix
	ruff format .

# Run the application
.PHONY: index-documents
index-documents:
	python app/main.py

# Run evaluation
.PHONY: evaluate
evaluate:
	python app/launch_evaluation.py