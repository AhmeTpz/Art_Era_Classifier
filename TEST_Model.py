import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ===================================================
# CONFIGURATION / YAPILANDIRMA
# ===================================================

model_path = "ArtClass_BestModel.h5"  # 👈 Buraya senin .h5 dosya adını yaz
dataset_dir = "Dataset"
img_height, img_width = 224, 224
batch_size = 32

# ===================================================
# TEST DATA PREPARATION / TEST VERİSİ HAZIRLAMA
# ===================================================

test_dir = os.path.join(dataset_dir, "Test")

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# ===================================================
# MODEL EVALUATION / MODEL DEĞERLENDİRME
# ===================================================

model = tf.keras.models.load_model(model_path)
loss, accuracy = model.evaluate(test_generator, verbose=1)
print(f"\n✅ Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
