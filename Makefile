UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
PYTHON := python3
else
PYTHON := python
endif

DATASET := datasets/dataset_train.csv
TEST_DATASET := datasets/dataset_test.csv

HISTOGRAM := src/data_visualization/histogram.py
SCATTER := src/data_visualization/scatter_plot.py
PAIR := src/data_visualization/pair_plot.py
TRAIN_MODULE := src.logistic_regression.logreg_train
PREDICT_MODULE := src.logistic_regression.logreg_predict

WEB_FRONTEND := web/frontend
WEB_BACKEND := web/backend/server.py

.PHONY: help describe describe-test histogram histogram-save scatter scatter-save pair pair-all pair-save train train-batch train-minibatch train-stochastic predict install-web api web build-web clean

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
	@printf "  make train                    Train all optimizers; use Batch for predictions\n"
	@printf "  make train-batch              Train all optimizers; use Batch for predictions\n"
	@printf "  make train-minibatch          Train all optimizers; use Mini-batch for predictions\n"
	@printf "  make train-stochastic         Train all optimizers; use Stochastic for predictions\n"
	@printf "  make predict                  Generate houses.csv from TEST_DATASET\n"
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

train:
	$(PYTHON) -m $(TRAIN_MODULE) $(DATASET) --epochs 1600

train-batch:
	$(PYTHON) -m $(TRAIN_MODULE) $(DATASET) --epochs 1600

train-minibatch:
	$(PYTHON) -m $(TRAIN_MODULE) $(DATASET) --epochs 1600 --batch-size 32

train-stochastic:
	$(PYTHON) -m $(TRAIN_MODULE) $(DATASET) --epochs 1600 --batch-size 1

predict:
	$(PYTHON) -m $(PREDICT_MODULE) $(TEST_DATASET) models/weights.json --output houses.csv

install-web:
	cd $(WEB_FRONTEND) && npm install

api:
	$(PYTHON) $(WEB_BACKEND)

web:
	cd $(WEB_FRONTEND) && npm run dev

build-web:
	cd $(WEB_FRONTEND) && npm run build

clean:
	find . -name '__pycache__' -type d -prune -exec rm -r {} +
	find . -name '*.pyc' -type f -delete
	rm -rf outputs
	rm -rf $(WEB_FRONTEND)/dist
