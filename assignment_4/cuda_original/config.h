/* config.h.  Generated from config.h.in by configure.  */
/* config.h.in.  Generated from configure.ac by autoheader.  */

/* Define if OpenMP is enabled */
#define HAVE_OPENMP 1

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "pcs2015@list.uva.nl"

/* Define to the full name of this package. */
#define PACKAGE_NAME "heat"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "heat 1.17"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "heat"

/* Define to the version of this package. */
#define PACKAGE_VERSION "1.17"

/* Define to empty if `const' does not conform to ANSI C. */
/* #undef const */

/* Define to the equivalent of the C99 'restrict' keyword, or to
   nothing if this is not supported.  Do not define if restrict is
   supported directly.  */
#define restrict __restrict
/* Work around a bug in Sun C++: it does not support _Restrict, even
   though the corresponding Sun C compiler does, which causes
   "#define restrict _Restrict" in the previous line.  Perhaps some future
   version of Sun C++ will work with _Restrict; if so, it'll probably
   define __RESTRICT, just as Sun C does.  */
#if defined __SUNPRO_CC && !defined __RESTRICT
# define _Restrict
#endif

/* Define to empty if the keyword `volatile' does not work. Warning: valid
   code using `volatile' can become incorrect without. Disable with care. */
/* #undef volatile */
