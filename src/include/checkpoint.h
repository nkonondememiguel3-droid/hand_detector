#ifndef hand_detector_checkpoint_h
#define hand_detector_checkpoint_h

#include "network.h"
#include <stdbool.h>

// Saves all learnable tensors (and BN running stats) for every layer in
// `net` to `path`, in the order layers were added. Returns true on success.
extern bool checkpoint_save( _network_t *net, const char *path );

// Loads a checkpoint from `path` into `net`. Validates layer types and
// tensor shapes against the live network; on any mismatch, returns false
// and leaves `net`'s tensors UNMODIFIED (all-or-nothing).
extern bool checkpoint_load( _network_t *net, const char *path );

#endif // hand_detector_checkpoint_h
