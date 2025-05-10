import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ===================================================
# MODEL CLASSIFIER CLASS / MODEL SINIFLANDIRICI SINIFI
# ===================================================

class ArtEraClassifier:
    def __init__(self):
        self.class_names = ['Baroque', 'Medieval', 'ModernArt', 'Renaissance']
        self.class_names_tr = ['Barok', 'Orta Çağ', 'Modern Sanat', 'Rönesans']
        
# ===================================================
# MODEL LOADING / MODEL YÜKLEME
# ===================================================
        
    def load_model(self, model_path):
        try:
            self.model = tf.keras.models.load_model(model_path)
            return True
        except Exception as e:
            print(f"Model yüklenirken hata oluştu: {e}")
            return False
            
# ===================================================
# IMAGE PREDICTION / GÖRSEL TAHMİN
# ===================================================
            
    def predict_image(self, image_path):
        try:
            img = Image.open(image_path)
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img = img.resize((224, 224))
            
            img_array = np.array(img, dtype=np.float32)
            
            img_array = img_array / 255.0
            
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = self.model.predict(img_array)
            predictions = predictions[0]
            
            results = []
            for i, prob in enumerate(predictions):
                results.append({
                    'class': self.class_names_tr[i],
                    'probability': float(prob) * 100
                })
            
            results.sort(key=lambda x: x['probability'], reverse=True)
            return results
            
        except Exception as e:
            print(f"Tahmin yapılırken hata oluştu: {e}")
            return None 