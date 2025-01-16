import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate, Layer
)
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.keras import backend as K
from sklearn.utils import class_weight

X = np.load('data/X.npy')
y = np.load('data/y.npy')

X = X[:500]
y = y[:500]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=9)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=9)

y_train_flat = y_train[y_train != -1].flatten()

classes = np.unique(y_train_flat)
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train_flat
)

class_weight_dict = dict(zip(classes, class_weights))
print("Class weights:", class_weight_dict)

class LearnablePositionalEncoding(Layer):
    def __init__(self, max_len, d_model):
        super(LearnablePositionalEncoding, self).__init__()
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, max_len, d_model),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=True,
        )

    def call(self, inputs):
        return inputs + self.pos_embedding[:, :tf.shape(inputs)[1], :]

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):

    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(inputs + x)


    x_ff = Dense(ff_dim, activation='relu', kernel_regularizer=l2(1e-3))(x)
    x_ff = Dense(inputs.shape[-1], kernel_regularizer=l2(1e-3))(x_ff)
    x_ff = Dropout(dropout)(x_ff)
    x = LayerNormalization(epsilon=1e-6)(x + x_ff)
    return x

def build_transformer_model(
    input_shape,
    num_classes,
    head_size=64,
    num_heads=4,
    ff_dim=128,
    num_transformer_blocks=4,
    mlp_units=[128, 64],
    dropout=0.2,
    l2_reg=1e-3,
):
    inputs = Input(shape=input_shape)
    x = LearnablePositionalEncoding(max_len=input_shape[0], d_model=input_shape[1])(inputs)


    for _ in range(num_transformer_blocks):
        x = transformer_encoder(
            x,
            head_size=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )


    for units in mlp_units:
        x = Dense(
            units,
            activation='relu',
            kernel_regularizer=l2(l2_reg)
        )(x)
        x = Dropout(dropout)(x)

    outputs = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model

class WeightedMaskedBinaryCrossentropy(tf.keras.losses.Loss):
    def __init__(self, class_weight_dict, mask_value=-1, **kwargs):
        super().__init__(**kwargs)
        self.class_weight_dict = class_weight_dict
        self.mask_value = mask_value

    def call(self, y_true, y_pred):

        weight_for_class_0 = tf.constant(self.class_weight_dict[0], dtype=tf.float32)
        weight_for_class_1 = tf.constant(self.class_weight_dict[1], dtype=tf.float32)


        mask = tf.cast(tf.not_equal(y_true, self.mask_value), tf.float32)


        weights = tf.where(tf.equal(y_true, 1), weight_for_class_1, weight_for_class_0)
        weights *= mask


        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)


        weighted_bce = bce * weights


        loss = tf.reduce_sum(weighted_bce) / tf.reduce_sum(weights)

        return loss

def categorical_accuracy(y_true, y_pred):
    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
    correct = tf.cast(tf.equal(y_true, y_pred_binary), tf.float32)
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    correct *= mask
    return tf.reduce_sum(correct) / tf.reduce_sum(mask)

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
    epochs=40,
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

def plot_roc_auc(y_true, y_pred, mask_value=-1):
    roc_aucs = []
    timesteps = y_true.shape[1]
    plt.figure(figsize=(15, 10))

    for i in range(timesteps):
        true_values = y_true[:, i, :].ravel()
        predicted_values = y_pred[:, i, :].ravel()

        valid_indices = true_values != mask_value
        true_values = true_values[valid_indices]
        predicted_values = predicted_values[valid_indices]

        if len(np.unique(true_values)) > 1:
            fpr, tpr, _ = roc_curve(true_values, predicted_values)
            roc_auc = roc_auc_score(true_values, predicted_values)
            roc_aucs.append(roc_auc)
            plt.plot(fpr, tpr, label=f'Timestep {i+1} (AUC = {roc_auc:.2f})')
        else:
            print(f'Skipped timestep {i+1} due to insufficient class variation.')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve for Each Timestep')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    return roc_aucs

y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

roc_aucs_test = plot_roc_auc(y_test, y_pred_test)
print("Average ROC AUC across valid timesteps (Test Data):",
      np.mean(roc_aucs_test) if roc_aucs_test else 'No valid data')

roc_aucs_train = plot_roc_auc(y_train, y_pred_train)
print("Average ROC AUC across valid timesteps (Train Data):",
      np.mean(roc_aucs_train) if roc_aucs_train else 'No valid data')

def calculate_accuracy(y_true, y_pred, mask_value=-1):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    mask = y_true_flat != mask_value
    y_true_filtered = y_true_flat[mask]
    y_pred_filtered = y_pred_flat[mask]
    y_pred_binary = (y_pred_filtered > 0.5).astype(int)
    correct = (y_true_filtered == y_pred_binary).sum()
    total = y_true_filtered.shape[0]
    accuracy = correct / total if total > 0 else 0
    return accuracy

test_accuracy = calculate_accuracy(y_test, y_pred_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

train_accuracy = calculate_accuracy(y_train, y_pred_train)
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

