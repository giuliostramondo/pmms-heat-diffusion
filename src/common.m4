AC_CONFIG_HEADERS([config.h])

AC_PROG_RANLIB
AC_PROG_CC_C99
AC_C_CONST
AC_C_RESTRICT
AC_C_VOLATILE

AC_SEARCH_LIBS([fabs], [m])
AC_SEARCH_LIBS([fmin], [m])
AC_SEARCH_LIBS([fmax], [m])
AC_SEARCH_LIBS([sqrt], [m])

if test "x$GCC" = xyes; then
   CFLAGS="$CFLAGS -Wall"
fi

COMMON="\$(srcdir)/../src"
AC_SUBST([COMMON])

CPPFLAGS="$CPPFLAGS -I\$(COMMON)"

# Need to run with -f to catch up changes to VERSION:
AUTOCONF="$AUTOCONF -f"
