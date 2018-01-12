noinst_PROGRAMS = heat heat_acc example_acc example_seq example_sse matmul_seq matmul_sse matmul_acc matmul_acc_adv matmul_omp

CLEANFILES=core.*

COMMON=$(srcdir)/../src
COMMON_FILES=$(COMMON)/main.c $(COMMON)/fail.c $(COMMON)/input.c $(COMMON)/output.c $(COMMON)/img.c

CPPFLAGS= -I $(COMMON)
heat_SOURCES = $(COMMON_FILES) compute.c
heat_CPPFLAGS = -fast -Minfo

heat_acc_SOURCES = $(COMMON_FILES) compute.c
heat_acc_CPPFLAGS = -fast -Minfo -acc
heat_acc_LDFLAGS = -acc

example_acc_SOURCES = example.c
example_acc_CPPFLAGS = -fast -Minfo -acc
example_acc_LDFLAGS = -acc

example_seq_SOURCES = example.c
example_seq_CPPFLAGS = -Minfo 

example_sse_SOURCES = example.c
example_sse_CPPFLAGS = -fast -Minfo 

matmul_seq_SOURCES = matmul.c
matmul_seq_CPPFLAGS = -Minfo -DMATMUL_SEQ

matmul_sse_SOURCES = matmul.c
matmul_sse_CPPFLAGS = -fast -Minfo -DMATMUL_SEQ

matmul_acc_SOURCES = matmul.c
matmul_acc_CPPFLAGS = -fast -Minfo -acc -DMATMUL_ACC
matmul_acc_LDFLAGS = -acc

matmul_acc_adv_SOURCES = matmul.c
matmul_acc_adv_CPPFLAGS = -fast -Minfo -acc -DMATMUL_ACC_ADV
matmul_acc_adv_LDFLAGS = -acc

matmul_omp_SOURCES = matmul.c
matmul_omp_CPPFLAGS = -fast -Minfo -mp -DMATMUL_OMP
matmul_omp_LDFLAGS = -mp

## Uncomment the following:
# noinst_PROGRAMS += benchmark
# benchmark_SOURCES = ...

#
# Maintenance rules
#
CONFIGURE_DEPENDENCIES = $(srcdir)/../VERSION
ACLOCAL_AMFLAGS = --force -I .
#include $(srcdir)/../src/libcommon.mk


