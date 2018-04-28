from keras.preprocessing.image import ImageDataGenerator


# Runtime data augmentation
def get_train_test_augmented(X_train, Y_train, X_valid, Y_valid, batch_size=32, seed=1001):
    # Image data generator distortion options
    data_gen_args = dict(rotation_range=180.,
                         zoom_range=[0.55, 1.45],
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='reflect')
    


    # Train data, provide the same seed and keyword arguments to the fit and flow methods
    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)
    X_datagen.fit(X_train, augment=True, seed=seed)
    Y_datagen.fit(Y_train, augment=True, seed=seed)
    X_train_augmented = X_datagen.flow(X_train, batch_size=batch_size, shuffle=True, seed=seed)
    Y_train_augmented = Y_datagen.flow(Y_train, batch_size=batch_size, shuffle=True, seed=seed)
     
    
    # Validation data, no data augmentation, but create a generator anyway
    X_datagen_val = ImageDataGenerator()
    Y_datagen_val = ImageDataGenerator()
    X_datagen_val.fit(X_valid, augment=True, seed=seed)
    Y_datagen_val.fit(Y_valid, augment=True, seed=seed)
    X_valid_augmented = X_datagen_val.flow(X_valid, batch_size=batch_size, shuffle=True, seed=seed)
    Y_valid_augmented = Y_datagen_val.flow(Y_valid, batch_size=batch_size, shuffle=True, seed=seed)
    
    
    # combine generators into one which yields image and masks
    train_generator = zip(X_train_augmented, Y_train_augmented)
    valid_generator = zip(X_valid_augmented, Y_valid_augmented)
    
    return train_generator, valid_generator