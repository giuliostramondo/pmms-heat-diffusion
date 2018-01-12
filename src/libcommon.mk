noinst_LIBRARIES = libcommon.a

libcommon_a_SOURCES = \
	$(COMMON)/compute.h \
	$(COMMON)/fail.c \
	$(COMMON)/fail.h \
	$(COMMON)/input.c \
	$(COMMON)/input.h \
	$(COMMON)/output.c \
	$(COMMON)/img.c \
	$(COMMON)/output.h

LDADD = libcommon.a
