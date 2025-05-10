import gdown # pip install gdown

# ===================================================
# DOWNLOAD LINKS / İNDİRME BAĞLANTILARI
# ===================================================

# Google Drive'dan indirilecek dosyaların bağlantıları
best_model_url = "https://drive.google.com/uc?export=download&id=1NPGuTGDbsJ5qKftOfwJshv9ZeJ5OkxSx"
all_models_url = "https://drive.google.com/uc?export=download&id=1LVNCWcx928S7gfKl2g1lXaXZGVDGdcNF"
dataset_url = "https://drive.google.com/uc?export=download&id=1ZwMuDv3DMSsdRL7wkgkbk2VgBr13o7BO"

# ===================================================
# USER INTERFACE / KULLANICI ARAYÜZÜ
# ===================================================

# İndirme seçeneklerini göster
print("Choose which model to download: (Hangi modeli indirmek istiyorsunuz?)\n")
print("1 - Download Best Model (~220MB) (En iyi modeli indir)")
print("2 - Download All Models (~2.8GB) (Tüm modelleri indir)")
print("3 - Download Dataset (~1.4GB) (Veri setini indir)")

# Kullanıcı seçimini al
choice = input("\nEnter your choice (1, 2 or 3): (Seçiminizi girin) ").strip()

# ===================================================
# DOWNLOAD PROCESS / İNDİRME İŞLEMİ
# ===================================================

# Seçime göre indirme işlemini gerçekleştir
if choice == "1":
    print("\nDownloading best model to current directory... (En iyi model mevcut dizine indiriliyor...)")
    gdown.download(best_model_url, quiet=False)
    print("✅ Best model downloaded successfully. (✅ En iyi model başarıyla indirildi.)")

elif choice == "2":
    print("\nDownloading all models to current directory... (Tüm modeller mevcut dizine indiriliyor...)")
    gdown.download(all_models_url, quiet=False)
    print("✅ All models downloaded successfully. (✅ Tüm modeller başarıyla indirildi.)")

elif choice == "3":
    print("\nDownloading dataset to current directory... (Veri seti mevcut dizine indiriliyor...)")
    gdown.download(dataset_url, quiet=False)
    print("✅ Dataset downloaded successfully. (✅ Veri seti başarıyla indirildi.)")
    print("\n📦 Please extract the .rar files manually. (Lütfen .rar dosyalarını elle çıkartınız.)")

else:
    print("❌ Invalid choice. Please enter 1, 2 or 3. (❌ Geçersiz seçim. Lütfen 1, 2 ya da 3 girin.)")
    exit()

# ===================================================
# DATASET DOWNLOAD OPTION / VERİ SETİ İNDİRME SEÇENEĞİ
# ===================================================

# Model indirildiyse veri seti indirme seçeneğini sun
if choice in ["1", "2"]:
    download_dataset = input(
        "\nDo you want to download the dataset as well? (Veri setini de indirmek ister misiniz?) [y/n]: ").strip().lower()

    if download_dataset == "y":
        print("\nDownloading dataset to current directory... (Veri seti mevcut dizine indiriliyor...)")
        gdown.download(dataset_url, quiet=False)
        print("✅ Dataset downloaded successfully. (✅ Veri seti başarıyla indirildi.)")
        print("\n📦 Please extract the .rar files manually. (Lütfen .rar dosyalarını elle çıkartınız.)")
    else:
        print("Dataset download skipped. (Veri seti indirilmedi.)")
