#include "acl/error_codes/rt_error_codes.h"
// rt.h location varies across CANN versions:
//   Some: {cann}/{target}/include/experiment/runtime/runtime/rt.h
//   8.5+: {cann}/{target}/pkg_inc/runtime/runtime/rt.h
// Include search has both {cann}/{target}/include and
// {cann}/{target}/pkg_inc/runtime, so try portable path first.
#if __has_include("runtime/rt.h")
#include "runtime/rt.h"
#elif __has_include("experiment/runtime/runtime/rt.h")
#include "experiment/runtime/runtime/rt.h"
#else
#error "Cannot find rt.h — check CANN installation"
#endif
#include <acl/acl.h>
