#include "checkpoint.h"
#include "ds_arena.h"
#include "inference.h"
#include "layers.h"
#include "network.h"
#include "preprocess.h"
#include "tensor.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_camera.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INPUT_SIZE 96
#define CONF_THRESHOLD 0.5f
#define CHECKPOINT_PATH "../checkpoints/hand_detector.bin"

//
// Network architecture — MUST match build_network() in main.c
// exactly (layer types, order, shapes) or checkpoint_load will
// fail its shape-validation pass.
//
static _network_t *build_network( _ds_arena_t_ *param_arena )
{
  _network_t *net = network_create( param_arena );

  network_add( net, conv2d_create( param_arena, 3, 16, 3, 2, 1 ) );
  network_add( net, batchnorm_create( param_arena, 16 ) );
  network_add( net, relu_create( param_arena ) );

  network_add( net, conv2d_create( param_arena, 16, 32, 3, 2, 1 ) );
  network_add( net, batchnorm_create( param_arena, 32 ) );
  network_add( net, relu_create( param_arena ) );

  network_add( net, conv2d_create( param_arena, 32, 64, 3, 2, 1 ) );
  network_add( net, batchnorm_create( param_arena, 64 ) );
  network_add( net, relu_create( param_arena ) );

  network_add( net, conv2d_create( param_arena, 64, 128, 3, 2, 1 ) );
  network_add( net, batchnorm_create( param_arena, 128 ) );
  network_add( net, relu_create( param_arena ) );

  network_add( net, gap_create( param_arena ) );
  network_add( net, dense_create( param_arena, 128, 5 ) );

  return net;
}

//
// HWC uint8 (camera frame, possibly RGBA) -> CHW float32 [0,1],
// resized to INPUT_SIZE x INPUT_SIZE. Drops alpha if present.
//
static _tensor_t *frame_to_tensor( _ds_arena_t_ *batch_arena, const unsigned char *pixels, int frame_w, int frame_h, int src_channels )
{
  // If the source has 4 channels (RGBA/BGRA), strip to 3 (RGB) first,
  // since hwc_uint8_to_chw_tensor / resize_image assume 3-channel.
  const unsigned char *rgb_src = pixels;
  unsigned char *rgb_buf = NULL;

  if ( src_channels != 3 )
  {
    rgb_buf = ARENA_ARRAY( batch_arena, unsigned char, (size_t)frame_w *frame_h * 3 );
    for ( int i = 0; i < frame_w * frame_h; i++ )
      for ( int c = 0; c < 3; c++ ) rgb_buf[i * 3 + c] = pixels[i * src_channels + c];
    rgb_src = rgb_buf;
  }

  unsigned char *resized = resize_image( batch_arena, rgb_src, frame_w, frame_h, 3, INPUT_SIZE, INPUT_SIZE );
  return hwc_uint8_to_chw_tensor( batch_arena, resized, INPUT_SIZE, INPUT_SIZE, 3 );
}

//
// Draw a hollow rectangle (SDL3 has no built-in DrawRect outline
// for floats spanning arbitrary thickness, so draw 4 lines)
//
static void draw_box( SDL_Renderer *renderer, float x, float y, float w, float h )
{
  SDL_SetRenderDrawColor( renderer, 0, 255, 0, 255 ); // green
  SDL_FRect rect = { x, y, w, h };
  SDL_RenderRect( renderer, &rect );
}

int main( int argc, char **argv )
{
  const char *ckpt_path = argc >= 2 ? argv[1] : CHECKPOINT_PATH;

  // Load network + checkpoint
  _ds_arena_t_ param_arena = ds_arena_new( 0 );
  _network_t *net = build_network( &param_arena );

  if ( !checkpoint_load( net, ckpt_path ) )
  {
    fprintf( stderr, "Failed to load checkpoint: %s\n", ckpt_path );
    return 1;
  }
  printf( "Loaded checkpoint: %s\n", ckpt_path );

  network_set_training( net, 0 ); // eval mode -- BN uses running stats

  // SDL3 init
  if ( !SDL_Init( SDL_INIT_VIDEO | SDL_INIT_CAMERA ) )
  {
    fprintf( stderr, "SDL_Init failed: %s\n", SDL_GetError() );
    return 1;
  }

  // Open default camera
  int num_devices = 0;
  SDL_CameraID *devices = SDL_GetCameras( &num_devices );
  if ( !devices || num_devices == 0 )
  {
    fprintf( stderr, "No camera devices found: %s\n", SDL_GetError() );
    SDL_Quit();
    return 1;
  }

  SDL_CameraID cam_id = devices[0];
  SDL_free( devices );

  SDL_Camera *camera = SDL_OpenCamera( cam_id, NULL ); // NULL = let SDL pick a format
  if ( !camera )
  {
    fprintf( stderr, "SDL_OpenCamera failed: %s\n", SDL_GetError() );
    SDL_Quit();
    return 1;
  }

  // Wait for camera to report its actual format
  // SDL3 cameras may take a frame or two to start delivering.
  SDL_CameraSpec spec;
  while ( !SDL_GetCameraFormat( camera, &spec ) ) SDL_Delay( 10 );

  int frame_w = spec.width;
  int frame_h = spec.height;
  printf( "Camera format: %dx%d, SDL_PixelFormat=%d\n", frame_w, frame_h, spec.format );

  // Create window + renderer
  SDL_Window *window = NULL;
  SDL_Renderer *renderer = NULL;
  if ( !SDL_CreateWindowAndRenderer( "Hand Detector Test", frame_w, frame_h, 0, &window, &renderer ) )
  {
    fprintf( stderr, "SDL_CreateWindowAndRenderer failed: %s\n", SDL_GetError() );
    SDL_Quit();
    return 1;
  }

  // Texture to hold camera frames for display
  // We'll convert each captured surface to RGBA32 for the texture.
  SDL_Texture *frame_texture = SDL_CreateTexture( renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, frame_w, frame_h );
  if ( !frame_texture )
  {
    fprintf( stderr, "SDL_CreateTexture failed: %s\n", SDL_GetError() );
    SDL_Quit();
    return 1;
  }

  // Batch arena for per-frame inference (reset every frame)
  _ds_arena_t_ batch_arena = ds_arena_new( 0 );

  printf( "Running. Press ESC or close window to quit.\n" );

  bool running = true;
  while ( running )
  {
    // Event loop
    SDL_Event event;
    while ( SDL_PollEvent( &event ) )
    {
      if ( event.type == SDL_EVENT_QUIT ) running = false;
      if ( event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE ) running = false;
    }

    // Acquire camera frame
    Uint64 timestamp_ns = 0;
    SDL_Surface *frame = SDL_AcquireCameraFrame( camera, &timestamp_ns );

    if ( frame )
    {
      // Convert to RGBA32 for both display and our HWC buffer, so
      // we have one consistent format regardless of camera's native
      // pixel format (often YUYV/NV12 on Linux webcams).
      SDL_Surface *rgba = SDL_ConvertSurface( frame, SDL_PIXELFORMAT_RGBA32 );

      if ( rgba )
      {
        // Update display texture
        SDL_UpdateTexture( frame_texture, NULL, rgba->pixels, rgba->pitch );

        // Run inference on this frame
        _ds_arena_checkpoint_t_ cp = ds_arena_checkpoint( &batch_arena );

        // rgba->pixels is tightly packed RGBA (4 bytes/pixel) assuming
        // pitch == width*4. If pitch differs (alignment padding), we'd
        // need to handle row-by-row; most SDL software surfaces for
        // RGBA32 have pitch == width*4, but check just in case.
        unsigned char *src = (unsigned char *)rgba->pixels;
        unsigned char *packed = src;

        if ( rgba->pitch != frame_w * 4 )
        {
          packed = ARENA_ARRAY( &batch_arena, unsigned char, (size_t)frame_w *frame_h * 4 );
          for ( int row = 0; row < frame_h; row++ )
            memcpy( packed + (size_t)row * frame_w * 4, src + (size_t)row * rgba->pitch, (size_t)frame_w * 4 );
        }

        _tensor_t *input = frame_to_tensor( &batch_arena, packed, frame_w, frame_h, 4 );

        _detection_t det = run_inference( net, &batch_arena, input, frame_w, frame_h, CONF_THRESHOLD );

        printf( "conf=%.3f  detected=%d", det.confidence, det.detected );
        if ( det.detected ) printf( "  box=(%.0f,%.0f,%.0f,%.0f)", det.x, det.y, det.width, det.height );
        printf( "\n" );

        // Render
        SDL_RenderClear( renderer );
        SDL_RenderTexture( renderer, frame_texture, NULL, NULL );

        if ( det.detected ) draw_box( renderer, det.x, det.y, det.width, det.height );

        SDL_RenderPresent( renderer );

        ds_arena_reset_to( &batch_arena, cp );
        SDL_DestroySurface( rgba );
      }

      SDL_ReleaseCameraFrame( camera, frame );
    }
    else
    {
      // No frame ready yet — small delay to avoid busy-spinning
      SDL_Delay( 5 );
    }
  }

  // Cleanup
  ds_arena_destroy( &batch_arena );
  SDL_DestroyTexture( frame_texture );
  SDL_DestroyRenderer( renderer );
  SDL_DestroyWindow( window );
  SDL_CloseCamera( camera );
  SDL_Quit();

  ds_arena_destroy( &param_arena );

  return 0;
}
