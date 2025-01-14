import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, MultiHeadAttention, Dropout, LayerNormalization, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

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
    plt.legend()
    plt.show()
    return roc_aucs

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