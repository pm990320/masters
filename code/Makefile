
.PHONY: main
main: evaluate

evaluate:
	KERAS_BACKEND=theano python main.py evaluate

lint:
	black -t py37 --exclude "nab/" --check .

format:
	black -t py37 --exclude "nab/" .
