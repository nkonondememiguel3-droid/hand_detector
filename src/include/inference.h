#ifndef hand_detector_inference_h
#define hand_detector_inference_h

#include "network.h"
#include "tensor.h"
#include <stdbool.h>

typedef struct
{
  float x, y;          // top-left corner, in frame pixels
  float width, height; // box size, in frame pixels
  float confidence;    // sigmoid-decoded, in [0,1]
  bool detected;       // confidence >= threshold
} _detection_t;

// Runs `net` forward on `input` (a preprocessed (1,3,H,W) tensor) and
// decodes the raw 5-element output into a Detection in frame-pixel
// coordinates. `frame_w`/`frame_h` are the dimensions of the ORIGINAL
// (pre-resize) frame the box should be expressed in. `threshold` sets
// `detected`.
extern _detection_t run_inference( _network_t *net, _ds_arena_t_ *batch_arena, _tensor_t *input, int frame_w, int frame_h, float threshold );

// Decodes a raw 5-element output array directly (no forward pass).
// Useful for testing the decode logic in isolation.
extern _detection_t decode_detection( const float raw[5], int frame_w, int frame_h, float threshold );

#endif // hand_detector_inference_h
