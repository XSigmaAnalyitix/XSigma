/*
 * Some parts of this implementation were inspired by code from VTK
 * (The Visualization Toolkit), distributed under a BSD-style license.
 * See LICENSE for details.
 */

#ifndef __wrapping_hints_h__
#define __wrapping_hints_h__

#ifdef __XSIGMA_WRAP__
#define XSIGMA_WRAP_HINTS_DEFINED
// Exclude a method or class from wrapping
#define XSIGMA_WRAPEXCLUDE [[xsigma::wrapexclude]]
// The return value points to a newly-created XSIGMA object.
#define XSIGMA_NEWINSTANCE [[xsigma::newinstance]]
// The parameter is a pointer to a zerocopy buffer.
#define XSIGMA_ZEROCOPY [[xsigma::zerocopy]]
// The parameter is a path on the filesystem.
#define XSIGMA_FILEPATH [[xsigma::filepath]]
// Set preconditions for a function
#define XSIGMA_EXPECTS(x) [[xsigma::expects(x)]]
// Set size hint for parameter or return value
#define XSIGMA_SIZEHINT(...) [[xsigma::sizehint(__VA_ARGS__)]]
// Opt-in a class for automatic code generation of (de)serializers.
#define XSIGMA_MARSHALAUTO [[xsigma::marshalauto]]
// Specifies that a class has hand written (de)serializers.
#define XSIGMA_MARSHALMANUAL [[xsigma::marshalmanual]]
// Excludes a function from the auto-generated (de)serialization process.
#define XSIGMA_MARSHALEXCLUDE(reason) [[xsigma::marshalexclude(reason)]]
// Enforces a function as the getter for `property`
#define XSIGMA_MARSHALGETTER(property) [[xsigma::marshalgetter(#property)]]
// Enforces a function as the setter for `property`
#define XSIGMA_MARSHALSETTER(property) [[xsigma::marshalsetter(#property)]]
#endif

#ifndef XSIGMA_WRAP_HINTS_DEFINED
#define XSIGMA_WRAPEXCLUDE
#define XSIGMA_NEWINSTANCE
#define XSIGMA_ZEROCOPY
#define XSIGMA_FILEPATH
#define XSIGMA_EXPECTS(x)
#define XSIGMA_SIZEHINT(...)
#define XSIGMA_MARSHALAUTO
#define XSIGMA_MARSHALMANUAL
#define XSIGMA_MARSHALEXCLUDE(reason)
#define XSIGMA_MARSHALGETTER(property)
#define XSIGMA_MARSHALSETTER(property)
#endif

#define XSIGMA_MARSHAL_EXCLUDE_REASON_IS_REDUNDANT "is redundant"
#define XSIGMA_MARSHAL_EXCLUDE_REASON_IS_INTERNAL "is internal"
#define XSIGMA_MARSHAL_EXCLUDE_REASON_NOT_SUPPORTED \
    "(de)serialization is not supported for this type of property"

#endif
// XSIGMA-HeaderTest-Exclude: wrapping_hints.h
