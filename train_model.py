import tensorflow as tf
from tensorflow.keras import layers, models 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import os

# ============================
# Paths
# ============================
DATASET_PATH = "dataset/processed"
MODEL_SAVE_PATH = "model/skin_model.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10   # You can increase later

# ============================
# Data Generators
# ============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

NUM_CLASSES = train_generator.num_classes
print("✅ Number of Classes:", NUM_CLASSES)

# ============================
# Model Architecture (CNN)
# ============================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================
# Training
# ============================
print("\n🚀 Training started...\n")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# ============================
# Save Model
# ============================
os.makedirs("model", exist_ok=True)
model.save(MODEL_SAVE_PATH)

print("\n🎉 Model training completed and saved successfully!")
print("📁 Saved at:", MODEL_SAVE_PATH)
