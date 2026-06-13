#ifndef hand_detector_tensor
#define hand_detector_tensor

#include "ds_arena.h"
#include <stdbool.h>

#define TENSOR_MAX_DIMS 8

typedef struct
{
  float *data;
  float *gradients; // NULL if requires_gradients == false.
  int ndim;         // the dimensions.
  int shape[TENSOR_MAX_DIMS];
  int strides[TENSOR_MAX_DIMS]; // how far we move in memory to reach the next
                                // element in each dimensions.
  int size;                     // the total number of elements.
  bool requires_gradients;
} _tensor_t;

extern _tensor_t *tensor_create( _ds_arena_t_ *a, int ndim, const int *shape, int requires_gradients );
extern _tensor_t *tensor_zeros( _ds_arena_t_ *a, int ndim, const int *shape );
extern _tensor_t *tensor_ones( _ds_arena_t_ *a, int ndim, const int *shape );
extern _tensor_t *tensor_random_normal( _ds_arena_t_ *a, int ndim, const int *shape, float mean, float std );

extern void tensor_zero_gradient( _tensor_t *t );
extern void tensor_clip_gradient( _tensor_t *t, float clip_value );

// helper macro for indexing into our tensors.
#define T4( t, n, c, h, w )                                                                                                                          \
  ( ( t )->data[( n ) * ( t )->strides[0] + ( c ) * ( t )->strides[1] + ( h ) * ( t )->strides[2] + ( w ) * ( t )->strides[3]] )

#define G4( t, n, c, h, w )                                                                                                                          \
  ( ( t )->gradients[( n ) * ( t )->strides[0] + ( c ) * ( t )->strides[1] + ( h ) * ( t )->strides[2] + ( w ) * ( t )->strides[3]] )

#endif // hand_detector_tensor
