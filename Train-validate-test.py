import os
import shutil
import random
from pathlib import Path

# ===================================================
# CONFIGURATION / YAPILANDIRMA
# ===================================================

# Veri seti bölme oranları ve klasör yolları
source_root = Path("")      # Görsellerin olduğu ana klasör
target_root = Path("Dataset")      # Bölünmüş veri setinin hedefi
split_ratios = {"Train": 0.7, "Validate": 0.2, "Test": 0.1}  # Eğitim, doğrulama ve test oranları

# ===================================================
# DATA SPLITTING / VERİ BÖLME
# ===================================================

# Her sınıf klasörünü işle ve görselleri böl
for class_folder in source_root.iterdir():
    if class_folder.is_dir():
        class_name = class_folder.name
        print(f"İşleniyor: {class_name}")

        # Sınıf klasöründeki tüm görselleri listele ve karıştır
        images = [f for f in os.listdir(class_folder) if f.endswith(".jpg")]
        random.shuffle(images)

        # Bölme noktalarını hesapla
        total = len(images)
        train_end = int(total * split_ratios["Train"])
        val_end = train_end + int(total * split_ratios["Validate"])

        # Görselleri bölümlere ayır
        splits = {
            "Train": images[:train_end],
            "Validate": images[train_end:val_end],
            "Test": images[val_end:]
        }

        # Her bölüm için hedef klasörleri oluştur ve görselleri taşı
        for split_name, split_files in splits.items():
            target_dir = target_root / split_name / class_name
            os.makedirs(target_dir, exist_ok=True)

            for filename in split_files:
                src = class_folder / filename
                dst = target_dir / filename
                shutil.move(str(src), str(dst))  # Görselleri taşı (kopyalama değil)

print("\n✅ Görseller rastgele olarak Train / Validate / Test klasörlerine taşındı.")
