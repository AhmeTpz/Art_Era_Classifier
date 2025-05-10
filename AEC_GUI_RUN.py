import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance
import random
import os
from AEC_Model_Classifier import ArtEraClassifier

# ===================================================
# MAIN APPLICATION CLASS / ANA UYGULAMA SINIFI
# ===================================================

class ArtEraClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Art Era Classifier")
        self.root.geometry("900x700")
        self.root.resizable(False, False)

        self.selected_image_path = None
        self.selected_model_path = None
        self.classifier = ArtEraClassifier()

        self.bg_images = []
        self.visual_refs = []
        self.main_tile_refs = []

        self.create_root_background()
        self.create_main_interface()

# ===================================================
# BACKGROUND CREATION / ARKA PLAN OLUŞTURMA
# ===================================================

    def create_root_background(self):
        self.bg_canvas = tk.Canvas(self.root, width=900, height=700)
        self.bg_canvas.pack(fill='both', expand=True)

        tiles = [f for f in os.listdir('assets') if f.endswith(('.png', '.jpg', '.jpeg')) and f != 'star.png']
        for img_name in tiles:
            img_path = os.path.join('assets', img_name)
            img = Image.open(img_path).resize((100, 100))
            img = ImageEnhance.Brightness(img).enhance(0.4)
            self.bg_images.append(ImageTk.PhotoImage(img))

        for i in range(0, 900, 100):
            for j in range(0, 700, 100):
                img = random.choice(self.bg_images)
                self.bg_canvas.create_image(i, j, image=img, anchor='nw')

# ===================================================
# MAIN INTERFACE CREATION / ANA ARAYÜZ OLUŞTURMA
# ===================================================

    def create_main_interface(self):
        self.main_frame = tk.Frame(self.bg_canvas, bg='#000000')
        self.bg_canvas.create_window(450, 350, window=self.main_frame, anchor='center', width=800, height=600)

        self.main_tile_canvas = tk.Canvas(self.main_frame, width=800, height=600, highlightthickness=0, bg='black')
        self.main_tile_canvas.place(x=0, y=0)
        for i in range(0, 800, 100):
            for j in range(0, 600, 100):
                img = random.choice(self.bg_images)
                tile = self.main_tile_canvas.create_image(i, j, image=img, anchor='nw')
                self.main_tile_refs.append(img)
        self.main_tile_canvas.create_rectangle(0, 0, 800, 600, fill='black', stipple='gray75')

        font_main = ('Segoe Script', 26, 'bold')
        font_label = ('Georgia', 12, 'bold')
        fg_color = '#E6C75B'

        tk.Label(self.main_frame, text="Art Era Classifier", font=font_main, bg='#000000', fg='#FFFFFF').pack(pady=10)

        model_frame = tk.Frame(self.main_frame, bg='#000000')
        model_frame.pack(pady=5)
        tk.Label(model_frame, text="Model seçiniz:", bg='#000000', fg=fg_color, font=font_label).pack(side='left')
        ttk.Button(model_frame, text="Modeli Seç", command=self.select_model).pack(side='left', padx=10)
        self.model_label = tk.Label(model_frame, text="", bg='#000000', fg='#C0C0C0', font=('Georgia', 12, 'italic'))
        self.model_label.pack(side='left', padx=10)

        image_frame = tk.Frame(self.main_frame, bg='#000000')
        image_frame.pack(pady=5)
        tk.Label(image_frame, text="Test edilecek görseli seçin:", bg='#000000', fg=fg_color, font=font_label).pack(side='left')
        ttk.Button(image_frame, text="Görseli Seç", command=self.select_image).pack(side='left', padx=10)
        self.image_label = tk.Label(image_frame, text="", bg='#000000', fg='#C0C0C0', font=('Georgia', 12, 'italic'))
        self.image_label.pack(side='left', padx=10)

        ttk.Button(self.main_frame, text="Test Et", command=self.test_image).pack(pady=10)

        self.visual_result_frame = tk.Frame(self.main_frame, bg='#000000')
        self.visual_result_frame.pack(pady=10)

        self.results_frame = tk.Frame(self.main_frame, bg='#000000')
        self.results_frame.pack(pady=10)

# ===================================================
# FILE SELECTION METHODS / DOSYA SEÇİM METODLARI
# ===================================================

    def select_image(self):
        self.selected_image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if self.selected_image_path:
            self.image_label.config(text=os.path.basename(self.selected_image_path))
            self.display_selected_image()

    def select_model(self):
        self.selected_model_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.h5")])
        if self.selected_model_path:
            self.model_label.config(text=os.path.basename(self.selected_model_path))

# ===================================================
# IMAGE DISPLAY AND TESTING / GÖRSEL GÖSTERİM VE TEST
# ===================================================

    def display_selected_image(self):
        for widget in self.visual_result_frame.winfo_children():
            widget.destroy()
        self.visual_refs.clear()

        container = tk.Frame(self.visual_result_frame, bg="#000000")
        container.pack(anchor='w', padx=20)

        if self.selected_image_path:
            img = Image.open(self.selected_image_path)
            img.thumbnail((150, 150))
            img_tk = ImageTk.PhotoImage(img)
            lbl = tk.Label(container, image=img_tk, bg='#000000')
            lbl.image = img_tk
            lbl.pack(side='left')
            self.visual_refs.append(img_tk)

    def test_image(self):
        if not self.selected_image_path or not self.selected_model_path:
            messagebox.showerror("Eksik Seçim", "Lütfen hem görsel hem de model seçiniz.")
            return

        if not self.classifier.load_model(self.selected_model_path):
            messagebox.showerror("Model Yüklenemedi", "Model dosyası geçerli değil.")
            return

        results = self.classifier.predict_image(self.selected_image_path)
        if not results:
            messagebox.showerror("Hata", "Model tahmin yapamadı.")
            return

        for widget in self.visual_result_frame.winfo_children():
            widget.destroy()
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        self.visual_refs.clear()

        container = tk.Frame(self.visual_result_frame, bg="#000000")
        container.pack(anchor='center', padx=20)

        img = Image.open(self.selected_image_path)
        img.thumbnail((150, 150))
        img_tk = ImageTk.PhotoImage(img)
        lbl = tk.Label(container, image=img_tk, bg='#000000')
        lbl.image = img_tk
        lbl.pack()
        self.visual_refs.append(img_tk)

        for i, res in enumerate(results):
            font = ('Georgia', 16, 'bold') if i == 0 else ('Georgia', 12)
            text = f"{res['class']}: %{res['probability']:.2f}"
            frame = tk.Frame(self.results_frame, bg='#000000')
            frame.pack(pady=3, anchor='center')

            if i == 0 and os.path.exists('assets/star.png'):
                star = Image.open('assets/star.png').resize((25, 25))
                star_tk = ImageTk.PhotoImage(star)
                lbl = tk.Label(frame, image=star_tk, bg='#000000')
                lbl.image = star_tk
                lbl.pack(side='left', padx=5)
                self.visual_refs.append(star_tk)

            tk.Label(frame, text=text, font=font, bg='#000000', fg='#E6C75B').pack(side='left')

# ===================================================
# APPLICATION ENTRY POINT / UYGULAMA BAŞLANGIÇ NOKTASI
# ===================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = ArtEraClassifierGUI(root)
    root.mainloop()
