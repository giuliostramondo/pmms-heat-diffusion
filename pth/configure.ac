AC_PREREQ([2.53])
AC_INIT([heat], m4_esyscmd([tr -d '\n' < ../VERSION]), [pcs2015@list.uva.nl])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([1.4 no-define foreign])

m4_include([../src/common.m4])

AX_PTHREAD

LIBS="$PTHREAD_LIBS $LIBS"
CFLAGS="$CFLAGS $PTHREAD_CFLAGS"
CC="$PTHREAD_CC"

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
