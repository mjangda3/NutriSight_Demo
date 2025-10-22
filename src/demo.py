import os
os.environ["PYTHONHASHSEED"] = "1337"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  

import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Layer
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils import class_weight

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from tensorflow.keras.utils import set_random_seed
set_random_seed(SEED)

# 1) Synthetic data 

N = 500       
T = 24        
F = 32        
O = 6         
MASK_VALUE = -1.0

rng = np.random.default_rng(SEED)

X = rng.normal(size=(N, T, F)).astype("float32")
X[:, 1:, :] += 0.3 * X[:, :-1, :]

risk = rng.normal(0.0, 1.0, size=(N, 1))
out_bias = rng.uniform(-0.6, 0.8, size=(O,))
time_trend = np.linspace(-0.2, 0.2, T)[None, :, None]
logits = 0.9 * risk[:, None, :] + out_bias[None, None, :] + time_trend
p = 1.0 / (1.0 + np.exp(-logits))
y = (rng.uniform(size=(N, T, O)) < p).astype("float32")

mask_idx = rng.uniform(size=(N, T, O)) < 0.10
y[mask_idx] = MASK_VALUE

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)

# 2) Class weights from non-masked training labels

y_train_flat = y_train[y_train != MASK_VALUE].flatten().astype(int)
classes = np.unique(y_train_flat)
cw = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train_flat)
class_weight_dict = dict(zip(classes, cw))
print("Class weights:", class_weight_dict)

# 3) Model

class LearnablePositionalEncoding(Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, max_len, d_model),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=True,
        )
    def call(self, inputs):
        return inputs + self.pos_embedding[:, :tf.shape(inputs)[1], :]

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.0, causal=True):
    """
    Single encoder block with residual MHA + FFN.
    If causal=True, each timestep can only attend to itself and the past.
    """
    attn_out = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs, use_causal_mask=causal)

    x = Dropout(dropout)(attn_out)
    x = LayerNormalization(epsilon=1e-6)(inputs + x)

    x_ff = Dense(ff_dim, activation='relu', kernel_regularizer=l2(1e-3))(x)
    x_ff = Dense(inputs.shape[-1], kernel_regularizer=l2(1e-3))(x_ff)
    x_ff = Dropout(dropout)(x_ff)
    x = LayerNormalization(epsilon=1e-6)(x + x_ff)
    return x

def build_transformer_model(
    input_shape,
    num_classes,
    head_size=24,
    num_heads=1,
    ff_dim=24,
    num_transformer_blocks=1,
    mlp_units=(24,),
    dropout=0.01,
    l2_reg=1e-6,
    causal=True,  
):
    if isinstance(mlp_units, int):
        mlp_units = (mlp_units,)

    inputs = Input(shape=input_shape)
    x = LearnablePositionalEncoding(max_len=input_shape[0], d_model=input_shape[1])(inputs)

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(
            x,
            head_size=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            causal=causal,  
        )

    for units in mlp_units:
        x = Dense(units, activation='relu', kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(dropout)(x)

    outputs = Dense(num_classes, activation='sigmoid')(x)
    return Model(inputs, outputs)

class WeightedMaskedBinaryCrossentropy(tf.keras.losses.Loss):
    def __init__(self, class_weight_dict, mask_value=MASK_VALUE, **kwargs):
        super().__init__(**kwargs)
        self.class_weight_dict = {0: float(class_weight_dict.get(0, 1.0)),
                                  1: float(class_weight_dict.get(1, 1.0))}
        self.mask_value = float(mask_value)
    def call(self, y_true, y_pred):
        w0 = tf.constant(self.class_weight_dict[0], dtype=tf.float32)
        w1 = tf.constant(self.class_weight_dict[1], dtype=tf.float32)
        mask = tf.cast(tf.not_equal(y_true, self.mask_value), tf.float32)
        weights = tf.where(tf.equal(y_true, 1.0), w1, w0) * mask
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        loss = tf.reduce_sum(bce * weights) / tf.maximum(tf.reduce_sum(weights), 1.0)
        return loss

def categorical_accuracy(y_true, y_pred):
    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
    correct = tf.cast(tf.equal(y_true, y_pred_binary), tf.float32)
    mask = tf.cast(tf.not_equal(y_true, MASK_VALUE), tf.float32)
    correct *= mask
    return tf.reduce_sum(correct) / tf.maximum(tf.reduce_sum(mask), 1.0)

input_shape = (X_train.shape[1], X_train.shape[2]) 
num_classes = y_train.shape[2]                      

model = build_transformer_model(
    input_shape=input_shape,
    num_classes=num_classes,
    head_size=24,
    num_heads=1,
    ff_dim=24,
    num_transformer_blocks=1,
    mlp_units=(24,),
    dropout=0.01,
    l2_reg=1e-6,
    causal=True, 
)

model.compile(
    optimizer=Adam(learning_rate=5e-6),
    loss=WeightedMaskedBinaryCrossentropy(class_weight_dict=class_weight_dict, mask_value=MASK_VALUE),
    metrics=[categorical_accuracy]
)

# Optional: hide internals if you want
# model.summary = lambda *args, **kwargs: print("Model summary hidden.")
model.summary()

# 4) Train

callbacks = [
    EarlyStopping(patience=8, verbose=1, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=1e-8, verbose=1)
]

history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=48,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# 5) Plots + metrics

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

def plot_roc_auc(y_true, y_pred, mask_value=MASK_VALUE):
    roc_aucs = []
    timesteps = y_true.shape[1]
    plt.figure(figsize=(15, 10))
    for i in range(timesteps):
        true_values = y_true[:, i, :].ravel()
        predicted_values = y_pred[:, i, :].ravel()
        valid = true_values != mask_value
        true_values = true_values[valid]
        predicted_values = predicted_values[valid]
        if len(np.unique(true_values)) > 1:
            fpr, tpr, _ = roc_curve(true_values, predicted_values)
            auc = roc_auc_score(true_values, predicted_values)
            roc_aucs.append(auc)
            plt.plot(fpr, tpr, label=f'Timestep {i+1} (AUC = {auc:.2f})')
        else:
            print(f'Skipped timestep {i+1} due to insufficient class variation.')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve for Each Timestep')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend(ncol=2); plt.tight_layout(); plt.show()
    return roc_aucs

y_pred_test = model.predict(X_test, verbose=0)
y_pred_train = model.predict(X_train, verbose=0)

roc_aucs_test = plot_roc_auc(y_test, y_pred_test)
print("Average ROC AUC across valid timesteps (Test Data):",
      np.mean(roc_aucs_test) if roc_aucs_test else 'No valid data')

roc_aucs_train = plot_roc_auc(y_train, y_pred_train)
print("Average ROC AUC across valid timesteps (Train Data):",
      np.mean(roc_aucs_train) if roc_aucs_train else 'No valid data')

def calculate_accuracy(y_true, y_pred, mask_value=MASK_VALUE):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    valid = y_true_flat != mask_value
    yt = y_true_flat[valid]
    yp = y_pred_flat[valid]
    ypb = (yp > 0.5).astype(int)
    return (yt == ypb).mean() if yt.size else 0.0

print(f"Test Accuracy:  {calculate_accuracy(y_test,  y_pred_test)*100:.2f}%")
print(f"Train Accuracy: {calculate_accuracy(y_train, y_pred_train)*100:.2f}%")
