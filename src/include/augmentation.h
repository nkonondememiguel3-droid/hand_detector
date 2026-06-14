#ifndef hand_detector_augmentation_h
#define hand_detector_augmentation_h

#include "dataloader.h"

extern void augmentation_horizontal_flip( unsigned char *data, int w, int h, int num_channels, _sample_t *sample );
extern void augmentation_birghtness( unsigned char *data, int size, float factor );
extern void augmentation_contrast( unsigned char *data, int size, float factor );
extern void augmenttion_gaussian_noise( unsigned char *data, int size, float sigma );

// Crops [x0,y0,x0+crop_w,y0+crop_h) from `data` (w x h x c) into a fresh
// arena-allocated buffer of size crop_w x crop_h x c. Updates `sample`'s
// box into the crop's coordinate frame, or sets has_hand=0 if the box
// falls entirely outside the crop. *w and *h are updated to crop_w/crop_h.
extern unsigned char *augmenttion_random_crop( _ds_arena_t_ *a, const unsigned char *data, int *w, int *h, int c, _sample_t *sample, int crop_w,
                                               int crop_h, int x0, int y0 );

#endif // hand_detector_augmentation_h
