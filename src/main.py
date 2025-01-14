import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import argparse
import os

from utils import (
    build_transformer_model,
    WeightedMaskedBinaryCrossentropy,
    categorical_accuracy,
    plot_roc_auc,
    calculate_accuracy
)

def main(args):
    X = np.load(args.input_X)
    y = np.load(args.input_y)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=13)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=13)

    y_train_flat = y_train[y_train != -1].flatten()
    classes = np.unique(y_train_flat)
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train_flat
    )
    class_weight_dict = dict(zip(classes, class_weights_array))
    print("Class weights:", class_weight_dict)

    input_shape = (X_train.shape[1], X_train.shape[2]) 
    num_classes = y_train.shape[2]                     

    model = build_transformer_model(
        input_shape=input_shape,
        num_classes=num_classes,
        head_size=512,
        num_heads=4,
        ff_dim=744,
        num_transformer_blocks=4,
        mlp_units=[312, 64, 48],
        dropout=0.35,
        l2_reg=1e-5,
    )

    model.compile(
        optimizer=Adam(learning_rate=5e-6),
        loss=WeightedMaskedBinaryCrossentropy(class_weight_dict=class_weight_dict, mask_value=-1),
        metrics=[categorical_accuracy]
    )

    model.summary()

    callbacks = [
        EarlyStopping(patience=8, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=1e-8, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=45,
        batch_size=12,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    roc_aucs_test = plot_roc_auc(y_test, y_pred_test)
    print("Average ROC AUC across valid timesteps (Test Data):",
          np.mean(roc_aucs_test) if roc_aucs_test else 'No valid data')

    roc_aucs_train = plot_roc_auc(y_train, y_pred_train)
    print("Average ROC AUC across valid timesteps (Train Data):",
          np.mean(roc_aucs_train) if roc_aucs_train else 'No valid data')

    test_accuracy = calculate_accuracy(y_test, y_pred_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    train_accuracy = calculate_accuracy(y_train, y_pred_train)
    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer Model")
    parser.add_argument('--input_X', type=str, default='data/X.npy', help='Path to X.npy file')
    parser.add_argument('--input_y', type=str, default='data/y.npy', help='Path to y.npy file')
    parser.add_argument('--output_dir', type=str, default='results/', help='Directory to save outputs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)