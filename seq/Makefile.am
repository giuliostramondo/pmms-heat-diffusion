noinst_PROGRAMS = inputdemo outputdemo imgdemo \
	heat_pics \
	heat

inputdemo_SOURCES = main_inputdemo.c 
outputdemo_SOURCES = main_outputdemo.c
imgdemo_SOURCES = main_imgdemo.c

heat_SOURCES = $(COMMON)/main.c compute.c

heat_pics_SOURCES = $(COMMON)/main.c compute.c
heat_pics_CPPFLAGS = -DGEN_PICTURES

#
# Maintenance rules
#
CONFIGURE_DEPENDENCIES = $(srcdir)/../VERSION
ACLOCAL_AMFLAGS = --force
include $(srcdir)/../src/libcommon.mk


# to allow build even when the reference is
# not available:
DISTCLEANFILES = ref1.c ref2.c
BUILT_SOURCES = ref1.c ref2.c
ref1.c ref2.c:
	touch $@