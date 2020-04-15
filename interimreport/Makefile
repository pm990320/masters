all : compile open
.PHONY : all

clean :
	rm -rvf **/*.aux
	rm -rvf **/*.bbl
	rm -rvf **/*.blg
	rm -rvf **/*.log
	rm -rvf **/*.out

compile : clean
	xelatex --interaction=nonstopmode -etex MSci_project_2250066.tex
	bibtex MSci_project_2250066.aux
	xelatex --interaction=nonstopmode -etex MSci_project_2250066.tex
	xelatex --interaction=nonstopmode -etex MSci_project_2250066.tex

open :
	open MSci_project_2250066.pdf
