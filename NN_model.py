import pandas as pd
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.losses import binary_crossentropy
from keras.models import Sequential
from keras.optimizer_v2.adam import Adam
from sklearn.model_selection import train_test_split


def get_data():
    return pd.read_csv('train_data.csv'), pd.read_csv('test_data.csv')


def get_NN_model():
    model = Sequential()
    model.add(Dense(100, activation="relu", input_shape=(30,)))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=binary_crossentropy,
                  metrics=['accuracy'])
    return model


def train_model(model):
    epochs = 100
    batch = 128

    train, test = get_data()
    X_train, y_train = train.drop(['Unnamed: 0', 'id', 'label'], axis=1), train['label']
    X_test, y_test = test.drop(['Unnamed: 0', 'id', 'label'], axis=1), test['label']
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    keras_callbacks = [
        ModelCheckpoint('./output/NN_model', save_weights_only=True, monitor='val_loss', mode='min',
                        save_best_only=True),
        EarlyStopping(monitor="val_loss", mode="min", patience=50, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", mode="min", patience=10, verbose=1),
    ]
    history = model.fit(
        X_train, y_train,
        batch_size=batch,
        epochs=epochs,
        shuffle=True,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=keras_callbacks
    )
    score = model.evaluate(X_test, y_test, verbose=0)
    print(score)

    model.save_weights('./output/NN_model')
