all: help

######################
# HELP
######################

######################
# TESTING
######################

test tests:
	poetry run pytest

######################
# LINTING
######################

lint:
	poetry run mypy --config-file mypy.ini .

######################
# FORMATTING
######################

PYTHON_FILES=.
format:
	poetry run black $(PYTHON_FILES)

######################
# HELP
######################

help:
	@echo '----'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'