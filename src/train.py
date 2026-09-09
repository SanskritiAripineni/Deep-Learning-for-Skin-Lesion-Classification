"""Train and evaluate a VGG19 skin-lesion classifier.

Expected data layout::

    data/
      train/<class name>/*.jpg
      validation/<class name>/*.jpg
      test/<class name>/*.jpg

Augmentation happens in memory; source images are never modified.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf


@dataclass(frozen=True)
class RunConfig:
    data_dir: str
    output_dir: str
    image_size: int
    batch_size: int
    epochs: int
    fine_tune_epochs: int
    learning_rate: float
    fine_tune_learning_rate: float
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a VGG19 classifier from class-labeled image folders."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def validate_split_directories(data_dir: Path) -> dict[str, Path]:
    validation_dir = data_dir / "validation"
    if not validation_dir.is_dir() and (data_dir / "val").is_dir():
        validation_dir = data_dir / "val"

    splits = {
        "train": data_dir / "train",
        "validation": validation_dir,
        "test": data_dir / "test",
    }
    missing = [name for name, path in splits.items() if not path.is_dir()]
    if missing:
        expected = ", ".join(str(path) for path in splits.values())
        raise FileNotFoundError(
            f"Missing dataset split(s): {', '.join(missing)}. Expected: {expected}"
        )
    return splits


def load_datasets(
    splits: dict[str, Path], image_size: int, batch_size: int, seed: int
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    common = {
        "image_size": (image_size, image_size),
        "batch_size": batch_size,
        "label_mode": "categorical",
    }
    train = tf.keras.utils.image_dataset_from_directory(
        splits["train"], shuffle=True, seed=seed, **common
    )
    class_names = list(train.class_names)
    validation = tf.keras.utils.image_dataset_from_directory(
        splits["validation"], shuffle=False, **common
    )
    test = tf.keras.utils.image_dataset_from_directory(
        splits["test"], shuffle=False, **common
    )

    for split_name, dataset in (("validation", validation), ("test", test)):
        if list(dataset.class_names) != class_names:
            raise ValueError(
                f"{split_name} class folders do not match training classes: {class_names}"
            )

    autotune = tf.data.AUTOTUNE
    return (
        train.prefetch(autotune),
        validation.prefetch(autotune),
        test.prefetch(autotune),
        class_names,
    )


def build_model(image_size: int, class_count: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.20),
        ],
        name="augmentation",
    )
    backbone = tf.keras.applications.VGG19(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size, image_size, 3),
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    x = augmentation(inputs)
    x = tf.keras.applications.vgg19.preprocess_input(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(
        class_count, activation="softmax", name="class_probabilities"
    )(x)
    return tf.keras.Model(inputs, outputs, name="dr_derma_vgg19"), backbone


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def make_callbacks(output_dir: Path) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            output_dir / "best_model.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7
        ),
    ]


def merge_history(*histories: tf.keras.callbacks.History) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for history in histories:
        for metric, values in history.history.items():
            merged.setdefault(metric, []).extend(float(value) for value in values)
    return merged


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = RunConfig(
        data_dir=str(args.data_dir),
        output_dir=str(args.output_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        learning_rate=args.learning_rate,
        fine_tune_learning_rate=args.fine_tune_learning_rate,
        seed=args.seed,
    )

    tf.keras.utils.set_random_seed(config.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = validate_split_directories(args.data_dir)
    train, validation, test, class_names = load_datasets(
        splits, config.image_size, config.batch_size, config.seed
    )
    write_json(args.output_dir / "config.json", asdict(config))
    write_json(args.output_dir / "class_names.json", class_names)

    model, backbone = build_model(config.image_size, len(class_names))
    compile_model(model, config.learning_rate)
    feature_history = model.fit(
        train,
        validation_data=validation,
        epochs=config.epochs,
        callbacks=make_callbacks(args.output_dir),
    )

    histories = [feature_history]
    if config.fine_tune_epochs > 0:
        backbone.trainable = True
        for layer in backbone.layers:
            layer.trainable = layer.name.startswith("block5_")
        compile_model(model, config.fine_tune_learning_rate)
        fine_tune_history = model.fit(
            train,
            validation_data=validation,
            epochs=config.fine_tune_epochs,
            callbacks=make_callbacks(args.output_dir),
        )
        histories.append(fine_tune_history)

    metrics = model.evaluate(test, return_dict=True)
    serializable_metrics = {name: float(value) for name, value in metrics.items()}
    write_json(args.output_dir / "history.json", merge_history(*histories))
    write_json(args.output_dir / "test_metrics.json", serializable_metrics)
    model.save(args.output_dir / "final_model.keras")
    print(json.dumps(serializable_metrics, indent=2))


if __name__ == "__main__":
    main()
