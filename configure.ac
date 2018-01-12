AC_PREREQ([2.53])

AC_INIT([heat], m4_esyscmd([tr -d '\n' < VERSION]), [pcs2015@list.uva.nl])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([1.4 no-define foreign])

AC_CONFIG_SUBDIRS([seq pth omp acc])

AC_CHECK_PROG([CONVERT], [convert], [convert], [no])
AM_CONDITIONAL([HAVE_CONVERT], [test "x$CONVERT" != xno])

# to force reloading the version number
AUTOCONF="$AUTOCONF -f"

AC_CONFIG_FILES([Makefile])
AC_OUTPUT

