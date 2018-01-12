AC_PREREQ([2.53])
AC_INIT([heat], m4_esyscmd([tr -d '\n' < ../VERSION]), [pcs2015@list.uva.nl])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([1.4 no-define foreign])

AC_PROG_CC([gcc-mp-4.3 gcc-mp-4.4 gcc cc])
m4_include([../src/common.m4])

AX_OPENMP

CFLAGS="$CFLAGS $OPENMP_CFLAGS"

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
