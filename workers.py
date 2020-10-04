from tensorflow.keras import layers, models


def yoon_cnn():
    sequence_input = layers.Input(shape=(25, 300), dtype='float32')

    filter_sizes = [1, 2, 3, 4, 5]

    convs = []
    for filter_size in filter_sizes:
        l_conv = layers.Conv1D(filters=200,
                               kernel_size=filter_size,
                               activation='relu')(sequence_input)
        l_pool = layers.GlobalMaxPooling1D()(l_conv)
        convs.append(l_pool)
    l_merge = layers.concatenate(convs, axis=1)
    x = layers.Dense(10, activation='relu')(l_merge)
    preds = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    return model


def cnn_worker(data):
    train_index, test_index, X, y = data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = yoon_cnn()
    model.fit(X_train, y_train)

    return model.evaluate(X_test, y_test)
