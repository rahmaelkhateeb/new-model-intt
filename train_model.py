import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import librosa
from audio_processor import AudioEmotionProcessor

print("🚀 Training HighAccuracy Emotion Detection Model... - train_model.py:10")
print("📊 Using RAVDESS Dataset - train_model.py:11")

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

processor = AudioEmotionProcessor()

# Check if RAVDESS exists
ravdess_path = "data/ravdess_processed"
if not os.path.exists(ravdess_path):
    print("❌ RAVDESS dataset not found! - train_model.py:22")
    print("📥 Download from: https://zenodo.org/record/1188976 - train_model.py:23")
    print("📁 Extract to: data/ravdess_processed/ - train_model.py:24")
    exit(1)

print("🔄 Processing RAVDESS dataset... - train_model.py:27")
X, y = processor.prepare_ravdess_dataset(ravdess_path)
print(f"✅ Dataset ready: {X.shape[0]} samples - train_model.py:29")

# Encode labels
y_encoded = processor.le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Build advanced model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(len(processor.emotions), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy'),
    tf.keras.callbacks.ReduceLROnPlateau(patience=7, factor=0.5),
    tf.keras.callbacks.ModelCheckpoint('models/emotion_model.h5', save_best_only=True)
]

print("🎯 Training model... - train_model.py:69")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=200,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Final evaluation
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"🎉 FINAL TEST ACCURACY: {test_acc:.4f} ({test_acc*100:.2f}%) - train_model.py:81")

# Save everything
model.save('models/emotion_model.h5')
with open('models/processor.pkl', 'wb') as f:
    pickle.dump(processor, f)

print("💾 Model saved to models/emotion_model.h5 - train_model.py:88")
print("✅ Training COMPLETE! Run: streamlit run app.py - train_model.py:89")