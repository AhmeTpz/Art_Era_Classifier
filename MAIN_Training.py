import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import datetime
import pandas as pd

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

img_height, img_width = 224, 224
batch_size = 32
epochs = 40
dataset_dir = "Dataset"
output_dir = "output_model"
os.makedirs(output_dir, exist_ok=True)

# ===================================================
# DATA PREPARATION / VERİ HAZIRLAMA
# ===================================================

train_dir = os.path.join(dataset_dir, "Train")
val_dir = os.path.join(dataset_dir, "Validate")
test_dir = os.path.join(dataset_dir, "Test")

# Veri artırma teknikleri ile eğitim verilerini zenginleştirme
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.9, 1.1],
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
# DEEP CNN MODEL / DERİN EVRİŞİMLİ SİNİR AĞI MODELİ
# ===================================================

# Çok katmanlı derin öğrenme modeli
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ===================================================
# CALLBACKS / GERİ ÇAĞIRMA FONKSİYONLARI
# ===================================================

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path = os.path.join(output_dir, f"deepcnn_model_{timestamp}.h5")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
    ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss', verbose=1)
]

# ===================================================
# CLASS WEIGHT CALCULATION / SINIF AĞIRLIKLARI HESAPLAMA
# ===================================================

labels = train_generator.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights_dict = dict(enumerate(class_weights))

# Sınıf dengesizliğini gidermek için özel ağırlıklandırma
class_name_to_index = train_generator.class_indices

boosted_weights = {
    'Baroque': 1.8,
    'ModernArt': 1.5,
    'Renaissance': 2.0,
    'Medieval': 1.0
}

for class_name, boost in boosted_weights.items():
    class_idx = class_name_to_index.get(class_name)
    if class_idx is not None:
        class_weights_dict[class_idx] *= boost
        print(f"⚖️ {class_name} ağırlığı artırıldı → {class_weights_dict[class_idx]:.4f}")
    else:
        print(f"❗ {class_name} sınıfı bulunamadı.")

print("📊 Final Class Weights:", class_weights_dict)

# ===================================================
# MODEL TRAINING / MODEL EĞİTİMİ
# ===================================================

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks,
    class_weight=class_weights_dict,
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
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

plot_metric(history, "accuracy", "model_accuracy.png")
plot_metric(history, "loss", "model_loss.png")

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
plt.savefig(os.path.join(output_dir, "model_confusion_matrix.png"))
plt.close()
print("📊 Confusion matrix kaydedildi")

# ===================================================
# ROC CURVES / ROC EĞRİLERİ
# ===================================================

y_true_binarized = label_binarize(y_true, classes=list(range(len(class_labels))))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(class_labels)):
    fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
for i in range(len(class_labels)):
    plt.plot(fpr[i], tpr[i], label=f"{class_labels[i]} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "model_roc_curve.png"))
plt.close()

# ===================================================
# CLASSIFICATION REPORT / SINIFLANDIRMA RAPORU
# ===================================================

report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap="coolwarm")
plt.title("Classification Report Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "model_classification_report.png"))
plt.close()

print("✅ Tüm gelişmiş görseller output_model klasörüne kaydedildi.")