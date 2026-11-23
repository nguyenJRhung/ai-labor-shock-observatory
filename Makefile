.PHONY: probe etl eda causal verify-r business test app all

ifeq ($(OS),Windows_NT)
	VENV_BIN := .venv/Scripts
	EXE := .exe
else
	VENV_BIN := .venv/bin
	EXE :=
endif

PYTHON := $(VENV_BIN)/python$(EXE)
PYTEST := $(VENV_BIN)/pytest$(EXE)
STREAMLIT := $(VENV_BIN)/streamlit$(EXE)

probe:
	$(PYTHON) src/etl/probe_data_sources.py

etl:
	$(PYTHON) -m src.etl.fetch_ilo_exposure
	$(PYTHON) -m src.etl.fetch_ilostat_lmics
	$(PYTHON) -m src.etl.fetch_wdi_github_aei
	$(PYTHON) -m src.etl.build_duckdb

eda:
	$(PYTHON) -m src.etl.generate_eda_charts

causal:
	$(PYTHON) -m src.causal.robustness_checks

verify-r:
	Rscript src/causal/verify_in_r.R

business:
	$(PYTHON) -m src.business.scenario_simulator

test:
	$(PYTEST) -v

app:
	$(STREAMLIT) run app/streamlit_app.py

all: etl eda causal verify-r business test
