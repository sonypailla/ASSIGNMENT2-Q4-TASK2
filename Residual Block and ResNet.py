# Define a Residual Block
def residual_block(input_tensor, filters=64):
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(input_tensor)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.Add()([x, input_tensor])  # Skip connection
    x = layers.Activation('relu')(x)  # Apply activation after addition
    return x

# Define a simple ResNet-like model
def resnet_like():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same', activation='relu')(inputs)
    
    x = residual_block(x)
    x = residual_block(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

# Create and print ResNet model summary
resnet_model = resnet_like()
resnet_model.summary()
