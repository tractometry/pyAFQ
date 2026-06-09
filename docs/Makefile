# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = pyAFQ
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

upload: html
		./upload-gh-pages.sh build/html/ pyAFQ tractometry

.PHONY: distclean realclean

# find artifacts from sphinx-gallery in examples folder
ARTIFACTS := $(strip \
			 	$(wildcard source/examples/*/*.gz) $(wildcard source/examples/**/*.gz) \
			 	$(wildcard source/examples/*/*.trk) $(wildcard source/examples/**/*.trk) \
			 	$(wildcard source/examples/*/*.npy) $(wildcard source/examples/**/*.npy) \
			 	$(wildcard source/examples/*/*.png) $(wildcard source/examples/**/*.png) \
			 	$(wildcard source/examples/*/*.mp4) $(wildcard source/examples/**/*.mp4) \
			  )

# leaves only the files that were in the distribution. deletes files created
# by sphinx-build within source
distclean: clean
	@echo Removing files created by sphinx-build
	rm -rf $(BUILDDIR)
	rm -f $(SOURCEDIR)/reference/config.rst
	rm -rf $(SOURCEDIR)/auto_examples/
	$(if $(ARTIFACTS),rm -f $(ARTIFACTS))

# delete more files than distclean. this would also remove the data files.
realclean: distclean
	@echo "WARNING: This will delete all data in your AFQ_data folder and .dipy directory."
	@echo -n "Proceed? [y/N] " && read ans && [ $${ans:-N} = y ]

	rm -rf $(HOME)/.dipy/
	rm -rf $(HOME)/AFQ_data/
