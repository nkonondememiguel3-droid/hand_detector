#ifndef hand_detector_layers_h
#define hand_detector_layers_h

#include "ds_arena.h"
#include "tensor.h"
#include <stdbool.h>

typedef struct _layer_ _layer_t;

typedef enum
{
  LAYER_CONV2D,
  LAYER_RELU,
  LAYER_BATCHNORM,
  LAYER_DENSE,
  LAYER_GAP
} _layer_type_t;

struct _layer_
{
  _layer_type_t type;
  const char *name;

  // learnable parameters
  _tensor_t *weights;
  _tensor_t *bias;

  // batch normalization specific.
  _tensor_t *gamma, *beta;
  _tensor_t *running_mean, *running_var;
  _tensor_t *batch_mean, *batch_var;
  float bn_momentum, bn_eps;
  bool training;

  // convolution specific confi
  int in_channels, out_channels;
  int kernel_size, stride, padding;

  // cache for backward pass
  _tensor_t *last_input;
  _tensor_t *last_output;
  _tensor_t *normalized;

  // Vtable
  _tensor_t *( *forward )( _layer_t *self, _ds_arena_t_ *batch_arena, _tensor_t *input );
  _tensor_t *( *backward )( _layer_t *self, _ds_arena_t_ *batch_arena, _tensor_t *gradient_out );
};

extern _layer_t *conv2d_create( _ds_arena_t_ *param_arena, int in_c, int out_c, int kernel_size, int stride, int padding );
extern _layer_t *relu_create( _ds_arena_t_ *param_arena );
extern _layer_t *batchnorm_create( _ds_arena_t_ *param_arena, int channels );
extern _layer_t *dense_create( _ds_arena_t_ *param_arena, int in_features, int out_features );
extern _layer_t *gap_create( _ds_arena_t_ *param_arena );

#endif // hand_detector_layers_h
