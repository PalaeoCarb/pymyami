.PHONY: test genparams build upload distribute

test:
	python -m unittest

genapprox:
	python pymyami/parameters/gen_approx_coefs.py

build:
	python setup.py sdist bdist_wheel

upload:
	twine upload dist/pymyami-$$(python setup.py --version)*

distribute:
	make test
	make build
	make upload