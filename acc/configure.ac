AC_PREREQ([2.53])
AC_INIT([heat], m4_esyscmd([tr -d '\n' < ../VERSION]), [pcs2015@list.uva.nl])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([1.4 no-define foreign])

#m4_include([../src/common.m4])
AC_CONFIG_HEADERS([config.h])

AC_PROG_CC([pgcc gcc])
#CC="pgcc"
AM_CPPFLAGS=""
CPPFLAGS=""
AM_CFLAGS=""
CFLAGS=""

# here $MPICC does not contain C99-specific flags
# so we need to recheck.
unset ac_cv_prog_cc_c99
AC_PROG_CC_C99

AC_CONFIG_FILES([Makefile])
AC_OUTPUT

