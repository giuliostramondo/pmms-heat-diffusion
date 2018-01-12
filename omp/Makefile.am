noinst_PROGRAMS = heat msort 

heat_SOURCES = $(COMMON)/main.c compute.c

msort_SOURCES = merge.c
msort_LDADD = 

## Uncomment the following:
# noinst_PROGRAMS += matrix
# matrix_SOURCES = ...

#
# Maintenance rules
#
CONFIGURE_DEPENDENCIES = $(srcdir)/../VERSION
ACLOCAL_AMFLAGS = --force -I .
include $(srcdir)/../src/libcommon.mk


