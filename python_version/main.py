"""
TensorFlow hand detector -- architecture-matched to the C implementation.

Usage:
    python3 train_tf.py <egohands_unified> <oxford_unified> <coco_negatives_unified> \
        --epochs 30 --batch_size 32 --input_size 96 \
        --out checkpoints/hand_detector_tf.bin
"""

import os
import sys
import argparse
import struct
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


# Data loading
def load_dataset_paths(unified_dir):
    labels_dir = os.path.join(unified_dir, "labels")
    images_dir = os.path.join(unified_dir, "images")

    samples = []
    skipped = 0
    for fname in os.listdir(labels_dir):
        if not fname.endswith(".txt"):
            continue
        stem = fname[:-4]
        img_path = os.path.join(images_dir, stem + ".jpg")
        if not os.path.exists(img_path):
            continue

        # Validate the JPEG can actually be decoded
        try:
            with open(img_path, "rb") as f:
                data = f.read()
            tf.io.decode_jpeg(data)
        except Exception:
            skipped += 1
            continue

        with open(os.path.join(labels_dir, fname)) as f:
            line = f.readline().strip()

        parts = line.split()
        if len(parts) != 5:
            continue

        cls, cx, cy, w, h = parts
        has_hand = int(cls)
        samples.append((img_path, has_hand, float(cx),
                       float(cy), float(w), float(h)))

    if skipped:
        print(f"  skipped {skipped} unreadable images in {unified_dir}")

    return samples


def build_tf_dataset(samples, input_size, batch_size, training=True):
    paths = [s[0] for s in samples]
    has_hand = np.array([s[1] for s in samples], dtype=np.float32)
    boxes = np.array([[s[2], s[3], s[4], s[5]]
                     for s in samples], dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, has_hand, boxes))

    def load_and_preprocess(path, hh, box):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [input_size, input_size], method="bilinear")
        # [0,1], matches hwc_uint8_to_chw_tensor normalization
        img = tf.cast(img, tf.float32) / 255.0

        if training:
            # Horizontal flip (mirrors your augmentation_horizontal_flip)
            do_flip = tf.random.uniform([]) < 0.5
            img = tf.cond(
                do_flip, lambda: tf.image.flip_left_right(img), lambda: img)
            new_cx = tf.cond(
                tf.logical_and(do_flip, hh > 0.5),
                lambda: 1.0 - box[0],
                lambda: box[0],
            )
            box = tf.stack([new_cx, box[1], box[2], box[3]])

            # Brightness / contrast jitter (matches augmentation_birghtness/contrast ranges)
            # ~[0.8,1.2] factor equivalent
            img = tf.image.random_brightness(img, max_delta=0.2)
            img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
            img = tf.clip_by_value(img, 0.0, 1.0)

            # Gaussian noise (matches augmenttion_gaussian_noise sigma=0.02)
            noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.02)
            img = tf.clip_by_value(img + noise, 0.0, 1.0)

        return img, (hh, box)

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(buffer_size=2048)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# Model — architecture-matched to build_network() in main.c
def build_model(input_size):
    inputs = layers.Input(shape=(input_size, input_size, 3), name="image")

    x = inputs
    # Stage 1: in=3, out=16, k=3, s=2, p=1 -> "same"
    x = layers.Conv2D(16, 3, strides=2, padding="same",
                      use_bias=True, name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)

    # Stage 2: in=16, out=32
    x = layers.Conv2D(32, 3, strides=2, padding="same",
                      use_bias=True, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)

    # Stage 3: in=32, out=64
    x = layers.Conv2D(64, 3, strides=2, padding="same",
                      use_bias=True, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)

    # Stage 4: in=64, out=128
    x = layers.Conv2D(128, 3, strides=2, padding="same",
                      use_bias=True, name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.ReLU(name="relu4")(x)

    # GAP -> (N,128)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Dense head -> (N,5): [conf_logit, cx, cy, w, h]
    outputs = layers.Dense(5, name="head")(x)

    model = models.Model(inputs, outputs, name="hand_detector")
    return model


# Loss — matches train_one_sample's bce_with_logits + 5*smooth_l1(masked)
BOX_LOSS_WEIGHT = 5.0


def detector_loss(has_hand, box_target):
    """
    Returns a Keras-compatible loss function closing over (has_hand, box_target)
    is awkward with the functional API, so instead we use a custom training
    step. See train() below.
    """
    pass


def smooth_l1(pred, target, mask):
    diff = tf.abs(pred - target)
    quadratic = 0.5 * diff * diff
    linear = diff - 0.5
    loss = tf.where(diff < 1.0, quadratic, linear)
    loss = loss * mask
    # mean over all elements (matches your per-sample mean in smooth_l1)
    denom = tf.reduce_sum(mask) + 1e-8
    return tf.reduce_sum(loss) / denom


def bce_with_logits(logits, target):
    # mean BCE, matches bce_with_logits's per-sample value
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=target, logits=logits)
    return tf.reduce_mean(loss)


# Training loop (custom, to mirror combined-loss structure exactly)
def train(model, train_ds, val_ds, epochs, lr, checkpoint_path):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    @tf.function
    def train_step(images, has_hand, boxes):
        with tf.GradientTape() as tape:
            output = model(images, training=True)  # (N,5)
            conf_logits = output[:, 0]
            box_pred = output[:, 1:5]

            conf_loss = bce_with_logits(conf_logits, has_hand)

            # (N,1) -> broadcast to (N,4)
            mask = tf.expand_dims(has_hand, axis=1)
            mask = tf.tile(mask, [1, 4])
            box_loss = smooth_l1(box_pred, boxes, mask)

            total_loss = conf_loss + BOX_LOSS_WEIGHT * box_loss

        grads = tape.gradient(total_loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(
            grads, 5.0)  # matches GRAD_CLIP_NORM=5.0
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return total_loss, conf_loss, box_loss

    @tf.function
    def val_step(images, has_hand, boxes):
        output = model(images, training=False)
        conf_logits = output[:, 0]
        box_pred = output[:, 1:5]

        conf_loss = bce_with_logits(conf_logits, has_hand)
        mask = tf.tile(tf.expand_dims(has_hand, axis=1), [1, 4])
        box_loss = smooth_l1(box_pred, boxes, mask)
        total_loss = conf_loss + BOX_LOSS_WEIGHT * box_loss

        return total_loss

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        train_losses = []
        for step, (images, (has_hand, boxes)) in enumerate(train_ds):
            total_loss, conf_loss, box_loss = train_step(
                images, has_hand, boxes)
            train_losses.append(float(total_loss))

            if (step + 1) % 50 == 0:
                print(f"  step {step+1}  loss={float(total_loss):.4f}  "
                      f"conf={float(conf_loss):.4f}  box={float(box_loss):.4f}")

        val_losses = []
        for images, (has_hand, boxes) in val_ds:
            val_losses.append(float(val_step(images, has_hand, boxes)))

        print(
            f"  train_loss={np.mean(train_losses):.4f}  val_loss={np.mean(val_losses):.4f}")

        save_checkpoint_c_format(model, checkpoint_path)
        print(f"  checkpoint saved -> {checkpoint_path}")


#
# Checkpoint export — matches checkpoint.c's binary format exactly
#
#
# Format:
#   [MAGIC: "HDC1", 4 bytes]
#   [num_layers: int32]
#   for each layer:
#     [layer_type: int32]   (LAYER_CONV2D=0, LAYER_BATCHNORM=1, LAYER_RELU=2, LAYER_GAP=3, LAYER_DENSE=4 -- VERIFY against your layers.h enum!)
#     [param_count: int32]
#     for each param:
#       [ndim: int32]
#       [shape[ndim]: int32 each]
#       [data: float32 * product(shape)]
#
# Param order:
#   CONV2D/DENSE   -> weights, bias
#   BATCHNORM      -> gamma, beta, running_mean, running_var
#   RELU/GAP       -> (none)

# IMPORTANT: these integer values MUST match your _layer_type_t enum in
# layers.h exactly, in declaration order. Open layers.h and confirm —
# if your enum is declared in a different order, update this dict.
LAYER_TYPE = {
    "conv2d": 0,
    "batchnorm": 1,
    "relu": 2,
    "gap": 3,
    "dense": 4,
}


def write_tensor(f, arr):
    """Writes [ndim][shape...][float32 data...] for a numpy array."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    ndim = arr.ndim
    f.write(struct.pack("<i", ndim))
    for d in arr.shape:
        f.write(struct.pack("<i", d))
    f.write(arr.tobytes())


def save_checkpoint_c_format(model, path):
    """
    Walks the model in the SAME ORDER as build_network() in main.c:
      conv1, bn1, relu1, conv2, bn2, relu2, conv3, bn3, relu3,
      conv4, bn4, relu4, gap, head(dense)

    For Conv2D: TF weight shape is (kh, kw, in_c, out_c) -- must be
    transposed to (out_c, in_c, kh, kw) to match your conv2d_create's
    weight tensor shape { out_c, in_c, kernel_size, kernel_size }.

    For Dense: TF weight shape is (in_features, out_features) -- must be
    transposed to (out_features, in_features) to match dense_create's
    weight tensor shape { out_features, in_features }.

    For BatchNorm: TF's layer.weights order is [gamma, beta, moving_mean,
    moving_variance] -- this matches checkpoint.c's BATCHNORM order
    (gamma, beta, running_mean, running_var) directly, no reordering needed.
    """
    layer_sequence = [
        ("conv1", "conv2d"), ("bn1", "batchnorm"), ("relu1", "relu"),
        ("conv2", "conv2d"), ("bn2", "batchnorm"), ("relu2", "relu"),
        ("conv3", "conv2d"), ("bn3", "batchnorm"), ("relu3", "relu"),
        ("conv4", "conv2d"), ("bn4", "batchnorm"), ("relu4", "relu"),
        ("gap", "gap"),
        ("head", "dense"),
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        f.write(b"HDC1")
        f.write(struct.pack("<i", len(layer_sequence)))

        for layer_name, layer_type in layer_sequence:
            f.write(struct.pack("<i", LAYER_TYPE[layer_type]))

            if layer_type == "conv2d":
                layer = model.get_layer(layer_name)
                kernel, bias = layer.get_weights()  # kernel: (kh,kw,in_c,out_c), bias: (out_c,)

                # Transpose (kh,kw,in_c,out_c) -> (out_c,in_c,kh,kw)
                kernel_chw = np.transpose(kernel, (3, 2, 0, 1))

                f.write(struct.pack("<i", 2))  # param_count
                write_tensor(f, kernel_chw)
                write_tensor(f, bias)

            elif layer_type == "batchnorm":
                layer = model.get_layer(layer_name)
                gamma, beta, moving_mean, moving_var = layer.get_weights()

                f.write(struct.pack("<i", 4))  # param_count
                write_tensor(f, gamma)
                write_tensor(f, beta)
                write_tensor(f, moving_mean)
                write_tensor(f, moving_var)

            elif layer_type == "dense":
                layer = model.get_layer(layer_name)
                # kernel: (in_feat,out_feat), bias: (out_feat,)
                kernel, bias = layer.get_weights()

                # Transpose (in_feat,out_feat) -> (out_feat,in_feat)
                kernel_t = np.transpose(kernel, (1, 0))

                f.write(struct.pack("<i", 2))
                write_tensor(f, kernel_t)
                write_tensor(f, bias)

            else:  # relu, gap -- no params
                f.write(struct.pack("<i", 0))


# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("egohands_root")
    parser.add_argument("oxford_root")
    parser.add_argument("coco_root")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--input_size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str,
                        default="checkpoints/hand_detector_tf.bin")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    args = parser.parse_args()

    print("Loading dataset paths...")
    egohands = load_dataset_paths(args.egohands_root)
    oxford = load_dataset_paths(args.oxford_root)
    coco = load_dataset_paths(args.coco_root)

    print(f"EgoHands: {len(egohands)}")
    print(f"Oxford: {len(oxford)}")
    print(f"COCO negatives: {len(coco)}")

    all_samples = egohands + oxford + coco
    positives = sum(1 for s in all_samples if s[1] == 1)
    negatives = sum(1 for s in all_samples if s[1] == 0)
    print(
        f"Combined: {len(all_samples)}  Positives: {positives}  Negatives: {negatives}")

    rng = np.random.default_rng(42)
    rng.shuffle(all_samples)

    n_val = int(len(all_samples) * args.val_fraction)
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]
    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")

    train_ds = build_tf_dataset(
        train_samples, args.input_size, args.batch_size, training=True)
    val_ds = build_tf_dataset(
        val_samples, args.input_size, args.batch_size, training=False)

    model = build_model(args.input_size)
    model.summary()

    train(model, train_ds, val_ds, args.epochs, args.lr, args.out)


if __name__ == "__main__":
    main()
