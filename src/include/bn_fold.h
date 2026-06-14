#ifndef hand_detector_bn_fold_h
#define hand_detector_bn_fold_h

#include "layers.h"

// Folds `bn`'s running statistics and affine params into `conv`'s
// weights and bias in-place. After this call, `bn` is mathematically
// the identity and should be removed from the inference network
// (do not call bn->forward again — its gamma/beta/running stats are
// now baked into conv and no longer represent a valid BN).
//
// Preconditions:
//  - conv->type == LAYER_CONV2D
//  - bn->type   == LAYER_BATCHNORM
//  - conv->out_channels == number of channels in bn (gamma/beta/etc.)
extern void fold_batchnorm_into_conv( _layer_t *conv, _layer_t *bn );

#endif // hand_detector_bn_fold_h
