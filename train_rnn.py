import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# -----------------------------
# 1. Parameters
# -----------------------------
vocab_size = 10000
maxlen = 100
epochs = 5
batch_size = 64

# -----------------------------
# 2. Load IMDb dataset
# -----------------------------
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# -----------------------------
# 3. Pad sequences
# -----------------------------
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# -----------------------------
# 4. Build Simple RNN model
# -----------------------------
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    SimpleRNN(128),
    Dense(1, activation="sigmoid")
])

# Force build (important for Python 3.13)
model.build(input_shape=(None, maxlen))

# -----------------------------
# 5. Compile model
# -----------------------------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# 6. Train model
# -----------------------------
history = model.fit(
    x_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2
)

# -----------------------------
# 7. Evaluate model
# -----------------------------
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# -----------------------------
# 8. Save model
# -----------------------------
model.save("imdb_simple_rnn.h5")
print("Model saved as imdb_simple_rnn.h5")
