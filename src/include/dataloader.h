#ifndef hand_detector_dataloader_h
#define hand_detector_dataloader_h

#include "ds_arena.h"
#include <stdbool.h>

#define SAMPLE_PATH_MAX 512

typedef struct
{
  char image_path[SAMPLE_PATH_MAX];
  float cx, cy, w, h;
  bool has_hand;
} _sample_t;

typedef struct
{
  _sample_t *samples;
  int count;
} _dataset_t;

extern _dataset_t *dataset_load( _ds_arena_t_ *a, const char *labels_dir, const char *image_dir );
extern _dataset_t *dataset_concat( _ds_arena_t_ *a, _dataset_t **datasets, int num_datasets );
extern void dataset_shuffle( _dataset_t *ds );
extern void dataset_split( _ds_arena_t_ *a, _dataset_t *full, _dataset_t **train, _dataset_t **val, float val_fraction );
extern void dataset_stats( _dataset_t *ds, int *positives, int *negatives );

#endif // hand_detector_dataloader_h
