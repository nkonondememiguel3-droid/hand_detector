#ifndef hand_detector_loss_h
#define hand_detector_loss_h

#include "ds_arena.h"
#include "tensor.h"
typedef struct
{
  float value;
  _tensor_t *gradients;
} _loss_t_;

// binary cross-entropy with logits, predicted = raw logits, target belongs to {0,1}.
extern _loss_t_ bce_with_logits( _ds_arena_t_ *a, _tensor_t *predicted, _tensor_t *target );

// smooth L1 (Huber) loss for box regression.
// mask may be Null (no masking) or same shape as predicted, values 0/1.
extern _loss_t_ smooth_l1( _ds_arena_t_ *a, _tensor_t *predicted, _tensor_t *target, _tensor_t *mask );

#endif // hand_detector_loss_h
