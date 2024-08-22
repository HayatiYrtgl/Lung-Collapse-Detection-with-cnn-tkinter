from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showerror
from gui.keras_function.preprocess_image import preprocess_image_keras
from gui.keras_function.predict_image import predict_lung_collapse


def open_file_name_function(root, model):
    file_path = askopenfilename(title="Select an image", filetypes=[
        ("Png files", "*.png"), ("Jpg files", ".jpg"), ("Jpeg Files", "*.jpeg")
    ], initialdir="/")

    if file_path:
        image, ctk_image = preprocess_image_keras(img_path=file_path)
        root.img_label.configure(image=ctk_image)
        label = predict_lung_collapse(image_tensor=image, model=model)

        if float(label) > 49.99:
            root.label.configure(text="Akciğer Sönmesi Tespit Edildi "+"%"+f"{float(label):.2f}", text_color="red")

        else:
            root.label.configure(text="Akciğer Sönmesi Tespit Edilmedi", text_color="blue")



    else:
        showerror("Hata", "Resim Seçilmedi")
        return False
