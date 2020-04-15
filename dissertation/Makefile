all : compile open
.PHONY : all

clean :
	rm -rvf **/*.aux
	rm -rvf **/*.bbl
	rm -rvf **/*.blg
	rm -rvf **/*.log
	rm -rvf **/*.out

compile : clean
	xelatex --interaction=nonstopmode -etex dissertation.tex
	bibtex dissertation.aux
	xelatex --interaction=nonstopmode -etex dissertation.tex
	xelatex --interaction=nonstopmode -etex dissertation.tex

open :
	open dissertation.pdf
