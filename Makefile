PYTHON := python3
DATASET := datasets/dataset_train.csv
TEST_DATASET := datasets/dataset_test.csv

HISTOGRAM := src/data_visualization/histogram.py
SCATTER := src/data_visualization/scatter_plot.py
PAIR := src/data_visualization/pair_plot.py

WEB_FRONTEND := web/frontend
WEB_BACKEND := web/backend/server.py

.PHONY: help describe describe-test histogram histogram-save scatter scatter-save pair pair-all pair-save install-web api web build-web clean

help:
	@printf "Available targets:\n"
	@printf "  make describe                 Run data analysis on DATASET\n"
	@printf "  make describe-test            Run data analysis on TEST_DATASET\n"
	@printf "  make histogram                Show auto-selected homogeneous course histogram\n"
	@printf "  make histogram COURSE=Potions Show one course histogram\n"
	@printf "  make histogram-save           Save histogram to OUT=histogram.png\n"
	@printf "  make scatter                  Show most similar feature scatter plot\n"
	@printf "  make scatter X=Astronomy Y='Defense Against the Dark Arts'\n"
	@printf "  make scatter-save             Save scatter plot to OUT=scatter_plot.png\n"
	@printf "  make pair                     Show selected pair plot\n"
	@printf "  make pair-all                 Show pair plot for all numeric features\n"
	@printf "  make pair-save                Save pair plot to OUT=pair_plot.png\n"
	@printf "  make install-web              Install React frontend dependencies\n"
	@printf "  make api                      Start Flask API on http://127.0.0.1:5000\n"
	@printf "  make web                      Start React app on http://127.0.0.1:5173\n"
	@printf "  make build-web                Build React frontend\n"
	@printf "  make clean                    Remove generated caches and outputs\n"

describe:
	$(PYTHON) describe.py $(DATASET)

describe-test:
	$(PYTHON) describe.py $(TEST_DATASET)

histogram:
ifdef COURSE
	$(PYTHON) $(HISTOGRAM) $(DATASET) "$(COURSE)"
else
	$(PYTHON) $(HISTOGRAM) $(DATASET)
endif

histogram-save:
ifdef COURSE
	$(PYTHON) $(HISTOGRAM) $(DATASET) "$(COURSE)" --no-show --save $(or $(OUT),histogram.png)
else
	$(PYTHON) $(HISTOGRAM) $(DATASET) --no-show --save $(or $(OUT),histogram.png)
endif

scatter:
ifdef X
	$(PYTHON) $(SCATTER) $(DATASET) "$(X)" "$(Y)"
else
	$(PYTHON) $(SCATTER) $(DATASET)
endif

scatter-save:
ifdef X
	$(PYTHON) $(SCATTER) $(DATASET) "$(X)" "$(Y)" --no-show --save $(or $(OUT),scatter_plot.png)
else
	$(PYTHON) $(SCATTER) $(DATASET) --no-show --save $(or $(OUT),scatter_plot.png)
endif

pair:
	$(PYTHON) $(PAIR) $(DATASET)

pair-all:
	$(PYTHON) $(PAIR) $(DATASET) --all

pair-save:
	$(PYTHON) $(PAIR) $(DATASET) --no-show --save $(or $(OUT),pair_plot.png)

install-web:
	npm install --prefix $(WEB_FRONTEND)

api:
	$(PYTHON) $(WEB_BACKEND)

web:
	npm run dev --prefix $(WEB_FRONTEND)

build-web:
	npm run build --prefix $(WEB_FRONTEND)

clean:
	find . -name '__pycache__' -type d -prune -exec rm -r {} +
	find . -name '*.pyc' -type f -delete
	rm -rf outputs
	rm -rf $(WEB_FRONTEND)/dist
