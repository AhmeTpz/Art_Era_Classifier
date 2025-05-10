import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import datetime

# ===================================================
# GPU CHECK / GPU KONTROLÜ
# ===================================================

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("✅ GPU bulundu:", physical_devices[0])
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("⚠️ GPU bulunamadı, CPU ile devam edilecek.")

# ===================================================
# CONFIGURATION / YAPILANDIRMA
# ===================================================

img_height, img_width = 299, 299
batch_size = 64
epochs = 40
dataset_dir = "Dataset"

train_dir = os.path.join(dataset_dir, "Train")
val_dir = os.path.join(dataset_dir, "Validate")
test_dir = os.path.join(dataset_dir, "Test")

# ===================================================
# DATA AUGMENTATION / VERİ ARTIRMA
# ===================================================

# Görüntü çeşitliliğini artırmak için veri artırma teknikleri
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical', shuffle=False)

# ===================================================
# RESNET50 MODEL / RESNET50 MODELİ
# ===================================================

# Önceden eğitilmiş ResNet50 modelini transfer öğrenme ile kullanma
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Son 30 katmanı eğitilebilir yapma (fine-tuning)
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ===================================================
# CALLBACKS / GERİ ÇAĞIRMA FONKSİYONLARI
# ===================================================

from tensorflow.keras.callbacks import EarlyStopping

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path = f"resnet_model_{timestamp}.h5"

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True,
    verbose=1
)

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
    ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss', verbose=1),
    early_stop
]

# ===================================================
# MODEL TRAINING / MODEL EĞİTİMİ
# ===================================================

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

# ===================================================
# METRIC PLOTTING / METRİK GRAFİKLERİ
# ===================================================

def plot_metric(history, metric, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history[metric], label=f"Train {metric}")
    plt.plot(history.history[f"val_{metric}"], label=f"Validation {metric}")
    plt.title(metric.capitalize())
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

plot_metric(history, "accuracy", "resnet_accuracy.png")
plot_metric(history, "loss", "resnet_loss.png")
print("📊 Eğitim başarı grafiklerini kaydettim (resnet_accuracy.png, resnet_loss.png)")

# ===================================================
# TEST EVALUATION / TEST DEĞERLENDİRME
# ===================================================

test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"\n✅ Test Başarısı: {test_acc:.4f}, Test Kaybı: {test_loss:.4f}")

# ===================================================
# CONFUSION MATRIX / KARIŞIKLIK MATRİSİ
# ===================================================

y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("resnet_confusion_matrix.png")
plt.close()
print("📊 Confusion matrix kaydedildi: resnet_confusion_matrix.png")
