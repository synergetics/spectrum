.DEFAULT_GOAL := help
SHELL := /bin/bash

setup: ## Install dependencies
	@pip install -r ./requirements.txt

test: ## Run tests (note: requires imports)
	@coverage run -m pytest -vv --log-cli-level=ERROR ./tests

publish: ## Publish to pypi
	@rm -rf dist build
	@python setup.py sdist bdist_wheel
	@twine upload dist/* --verbose

help: ## Dislay this help
	@IFS=$$'\n'; for line in `grep -h -E '^[a-zA-Z_#-]+:?.*?## .*$$' $(MAKEFILE_LIST)`; do if [ "$${line:0:2}" = "##" ]; then \
	echo $$line | awk 'BEGIN {FS = "## "}; {printf "\n\033[33m%s\033[0m\n", $$2}'; else \
	echo $$line | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'; fi; \
	done; unset IFS;
