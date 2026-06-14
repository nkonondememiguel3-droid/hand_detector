#ifndef hand_detector_preprocess_h
#define hand_detector_preprocess_h

#include "ds_arena.h"
#include "tensor.h"

// Loads an image from disk into a fresh arena-allocated HWC uint8 buffer
// (forced to 3 channels). Sets *w, *h, *c. Returns NULL on failure.
extern unsigned char *load_image_raw( _ds_arena_t_ *a, const char *path, int *w, int *h, int *c );

// Resizes an HWC uint8 buffer to target_w x target_h (same channel count),
// returns a fresh arena-allocated buffer.
extern unsigned char *resize_image( _ds_arena_t_ *a, const unsigned char *data, int w, int h, int c, int target_w, int target_h );

// Converts an HWC uint8 buffer to a (1,c,h,w) CHW float32 tensor, values
// normalized to [0,1].
extern _tensor_t *hwc_uint8_to_chw_tensor( _ds_arena_t_ *a, const unsigned char *data, int w, int h, int c );

// Convenience: load + resize + convert in one call. Returns NULL on failure.
extern _tensor_t *load_and_preprocess( _ds_arena_t_ *a, const char *path, int target_w, int target_h );

#endif // hand_detector_preprocess_h
