import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ===================================================
# CONFIGURATION / YAPILANDIRMA
# ===================================================

img_height, img_width = 224, 224
batch_size = 32
epochs = 10
model_path = "WITH_Classweight_Models/ArtClass_BestModel.h5"  # Model dosyasının yolu / Path to model file
output_dir = "improved_output_model9"
os.makedirs(output_dir, exist_ok=True)

# ===================================================
# DATA LOADING / VERİ YÜKLEME
# ===================================================

dataset_dir = "Dataset"
train_dir = os.path.join(dataset_dir, "Train")
val_dir = os.path.join(dataset_dir, "Validate")
test_dir = os.path.join(dataset_dir, "Test")

datagen_args = dict(rescale=1./255)

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_args).flow_from_directory(
    train_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')
val_generator = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_args).flow_from_directory(
    val_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_args).flow_from_directory(
    test_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical', shuffle=False)

# ===================================================
# MODEL LOADING / MODEL YÜKLEME
# ===================================================

model = load_model(model_path)

# ===================================================
# CALLBACKS / GERİ ÇAĞIRMA FONKSİYONLARI
# ===================================================

timestamp = tf.timestamp().numpy()
model_save_path = os.path.join(output_dir, f"improved_model_{int(timestamp)}.h5")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath=model_save_path, save_best_only=True, monitor='val_loss', verbose=1)
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
    plt.plot(history.history[metric], label="Train")
    plt.plot(history.history[f"val_{metric}"], label="Validation")
    plt.title(metric.capitalize())
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

plot_metric(history, "accuracy", "improved_accuracy.png")
plot_metric(history, "loss", "improved_loss.png")

# ===================================================
# TEST EVALUATION / TEST DEĞERLENDİRME
# ===================================================

test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"✅ Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

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
plt.savefig(os.path.join(output_dir, "improved_confusion_matrix.png"))
plt.close()

# ===================================================
# ROC CURVES / ROC EĞRİLERİ
# ===================================================

y_true_bin = label_binarize(y_true, classes=list(range(len(class_labels))))
fpr, tpr, roc_auc = {}, {}, {}
for i in range(len(class_labels)):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
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
plt.savefig(os.path.join(output_dir, "improved_roc_curve.png"))
plt.close()

# ===================================================
# CLASSIFICATION REPORT / SINIFLANDIRMA RAPORU
# ===================================================

report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap="coolwarm")
plt.title("Classification Report Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "improved_classification_report.png"))
plt.close()

print("📁 Tüm görseller ve model improved_output_model9 klasörüne kaydedildi.")
