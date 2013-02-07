# Standard-Makefile für LaTeX

MAINTEX = $(wildcard *.tex)
MAIN = $(patsubst %.tex,%,$(MAINTEX))

BIBFILE = $(wildcard *.bib)


pdf:	$(MAIN).tex
	echo hello > $(MAIN).aux.old;
	touch $(MAIN).aux;
	set -e; until cmp -s $(patsubst %.tex,%,$<).aux $(patsubst %.tex,%,$<).aux.old; do \
		cp $(patsubst %.tex,%,$<).aux $(patsubst %.tex,%,$<).aux.old; \
		pdflatex $<; \
	done
	-rm -f $(MAIN).aux.old;

pdflatex: $(MAIN).tex
	pdflatex $<

latex: $(MAIN).tex
	latex $<

bibtex:	$(BIBFILE)
	bibtex $(MAIN)

index:	$(MAIN).idx
	makeindex -g -s settings.ist $<

all:	$(MAIN).tex pdflatex bibtex index 
	echo hello > $(MAIN).aux.old;
	touch $(MAIN).aux;
	set -e; until cmp -s $(patsubst %.tex,%,$<).aux $(patsubst %.tex,%,$<).aux.old; do \
		cp $(patsubst %.tex,%,$<).aux $(patsubst %.tex,%,$<).aux.old; \
		pdflatex $<; \
	done
	-rm -f $(MAIN).aux.old;

clean : 
	-rm -f *~ *.bbl *.blg *.log *.aux *.aux.old *.ind *.idx *.ilg *.toc *.lof *.lot *.lol *.out

