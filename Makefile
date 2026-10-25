PYTHON ?= .venv/bin/python
PORT ?= 8020
export PYTHONDONTWRITEBYTECODE := 1
export PYTHONPATH := $(CURDIR)
.PHONY: serve check audit
serve:
	$(PYTHON) -m submission serve --port $(PORT)
check:
	$(PYTHON) -m submission check
audit:
	$(PYTHON) -m submission audit
