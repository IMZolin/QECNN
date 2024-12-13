import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer, Conv2D, Input, Add, GlobalAveragePooling2D, Reshape, Dropout, DepthwiseConv2D, Multiply, Resizing, BatchNormalization, Activation, Lambda, Dense, GroupNormalization, SeparableConv2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
from image_utils import LoadImagesFromFolder, psnr

from tensorflow.keras.mixed_precision import Policy
Policy('mixed_float16')

def plot_training_history(history):
    """Plot training and validation loss and metrics."""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss during Training')
    if 'psnr' in history.history and 'val_psnr' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['psnr'], label='Train PSNR')
        plt.plot(history.history['val_psnr'], label='Validation PSNR')
        plt.xlabel('Epochs')
        plt.ylabel('PSNR (dB)')
        plt.legend()
        plt.title('PSNR during Training')
    plt.tight_layout()
    plt.show()

# Squeeze-Excite Block with improvements
def squeeze_excite_block(input_tensor, ratio=16):
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, filters))(se)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    return Multiply()([input_tensor, se])

# Attention Block combining spatial and channel attention
def attention_block(input_tensor):
    x = squeeze_excite_block(input_tensor)
    x = Conv2D(1, kernel_size=7, padding="same", activation='sigmoid')(x)
    return Multiply()([input_tensor, x])

# Multi-scale feature extraction
def multi_scale_block(input_tensor, filters):
    x1 = Conv2D(filters, kernel_size=3, padding="same", activation='relu')(input_tensor)
    x2 = Conv2D(filters, kernel_size=5, padding="same", activation='relu')(input_tensor)
    x3 = Conv2D(filters, kernel_size=7, padding="same", activation='relu')(input_tensor)
    return Add()([x1, x2, x3])

# Enhanced Residual Block with Group Normalization
def residual_block(input_tensor, filters, kernel_size=3, dilation_rate=1, use_dropout=False):
    residual = SeparableConv2D(filters, (3, 3), padding='same', activation='relu')(input_tensor)
    residual = BatchNormalization()(residual)
    residual = SeparableConv2D(filters, (3, 3), padding='same')(residual)
    x = Add()([input_tensor, residual])
    x = Activation('relu')(x)

    if use_dropout:
        x = Dropout(0.3)(x)
    x = Conv2D(filters, kernel_size, padding="same", dilation_rate=dilation_rate, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Add()([input_tensor, x])
    return Activation('relu')(x)

# Modified QECNN Model
def ImprovedEnhancerModel(fw, fh):
    input_tensor = Input(shape=(fh, fw, 3))

    x = Conv2D(64, kernel_size=3, padding="same", activation='relu')(input_tensor)
    
    # Residual + Attention Blocks
    for _ in range(5):
        x = residual_block(x, filters=64, kernel_size=3, dilation_rate=2)
        x = residual_block(x, filters=64, kernel_size=3, dilation_rate=2)
        x = attention_block(x)
    
    # Final Convolution
    x = Conv2D(3, kernel_size=3, padding="same", activation='sigmoid')(x)
    model = Model(inputs=input_tensor, outputs=x)
    return model

def TrainImageImprovedEnhancementModel(num_epochs, batch_size, lr, folderRaw, folderComp, folderRawVal, folderCompVal, patchsize):
    print('Loading raw train images...')
    Xraw = LoadImagesFromFolder(folderRaw, patchsize) / 255.0
    print('Loading compressed train images...')
    Xcomp = LoadImagesFromFolder(folderComp, patchsize) / 255.0

    print('Loading raw validation images...')
    XrawVal = LoadImagesFromFolder(folderRawVal, patchsize) / 255.0
    print('Loading compressed validation images...')
    XcompVal = LoadImagesFromFolder(folderCompVal, patchsize) / 255.0

    enhancer = ImprovedEnhancerModel(patchsize, patchsize)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=10000,
        decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=1.0)
    enhancer.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[psnr])

    # Model checkpoint callback
    checkpoint_filepath = 'best_improved_model.weights.h5'
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,  
        mode='min',
        verbose=1
    )

    # Training the model
    print('Training improved model...')
    history_improved = enhancer.fit(
        Xcomp, Xraw, epochs=num_epochs, batch_size=batch_size, verbose=1,
        validation_data=(XcompVal, XrawVal),
        callbacks=[checkpoint_callback]
    )

    # Save final model weights
    try:
        enhancer.save_weights('improved_enhancer.weights.h5')
        print("Weights saved successfully.")
    except Exception as e:
        print(f"Error saving weights: {e}")
    plot_training_history(history_improved)
    return enhancer
