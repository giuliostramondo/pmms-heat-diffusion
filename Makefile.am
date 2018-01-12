SUBDIRS = . seq pth omp acc

#
# Data files
#
include data.mk

nodist_noinst_DATA =
if HAVE_CONVERT
nodist_noinst_DATA += areas_100x150.pgm plasma_100x150.pgm pat1_100x150.pgm	\
	pat2_100x150.pgm pat3_100x150.pgm uni_100x150.pgm	\
	gradient_100x150.pgm
endif

DISTCLEANFILES = *.pgm

.PHONY: data
data: $(nodist_noinst_DATA)


#
# Maintenance rules
#
CONFIGURE_DEPENDENCIES = $(srcdir)/VERSION
ACLOCAL_AMFLAGS = --force -I src
