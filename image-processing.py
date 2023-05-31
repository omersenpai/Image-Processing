


import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt

# Tkinter arayüzü oluşturma
root = tk.Tk()
root.geometry("800x600") #Arayüz boyutu ayarlama
root.title("Görüntü Filtreleme Uygulaması(KISA SINAV-1)")

# Resim seçme işlemi
def browse_file():
    global img_path #global değişkeni kullandık.Çünkü diğer fonksiyonlarda bu değişkeni kullanmamız gerekti.
    img_path = filedialog.askopenfilename()
    img = cv2.imread(img_path)
    if img != '':
        image_np = np.array(img) #dizinler boş değerden farklıysa kaydeder.

    # NumPy dizisini terminal ekranında gösterir.
    print("Diziler:\n")
    print(image_np)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_plot = plt.subplot(2, 2, 1)
    img_plot.imshow(img)
    img_plot.set_title("Orjinal Görüntü")
    plt.axis('off')
    plt.show()

# Resim Filtreleme işlemi
def apply_filter():
    kernel_size = int(kernel_size_entry.get())
    sigma = float(sigma_entry.get())
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Filtreleme yöntemleri
    blur = cv2.blur(img, (kernel_size, kernel_size), sigma)
    median = cv2.medianBlur(img, kernel_size)
    bilateral = cv2.bilateralFilter(img, kernel_size, sigmaColor=75, sigmaSpace=75)
    gaussian = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

    # Plotlama işlemleri
 
    plt.subplot(2, 2, 2)
    plt.imshow(blur)
    plt.title("Averaging Bulanıklığı")
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(median)
    plt.title("Median Bulanıklığı")
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(bilateral)
    plt.title("Bilateral Bulanıklığı")
    plt.axis('off')
    plt.subplot(2, 2, 1)
    plt.imshow(gaussian)
    plt.title("Gaussian Bulanıklığı")
    plt.axis('off')
    plt.subplot(2, 2, 1)
    plt.show()

    # Resmi kaydetme işlemi
    cv2.imwrite('filtered_image.jpg', bilateral)
    #Filter 1 ekranı hakkında kısayollar.
    k = cv2.waitKey(0)
    if k == 27:   # Escye basınca kapanır.
        cv2.destroyAllWindows()
     
    elif k == ord('s'): # S ye basınca kaydedir.AYrıca Save the figure  butonundan da kaydedebiliriz.
        cv2.imwrite('Kayıtlıresim.jpg',img)
        cv2.destroyAllWindows()

# Arayüz öğeleri
browse_button = tk.Button(root, text="Gözat", command=browse_file)
browse_button.pack()
kernel_size_label = tk.Label(root, text="Kernel Boyutu:")
kernel_size_label.pack()
kernel_size_entry = tk.Entry(root)
kernel_size_entry.pack()
sigma_label = tk.Label(root, text="Sigma Değeri:")
sigma_label.pack()
sigma_entry = tk.Entry(root)
sigma_entry.pack()
apply_button = tk.Button(root, text="Filtrele", command=apply_filter)
apply_button.pack()

root.mainloop() # Tkinter uygulamasının çalışmasını sağlaR ve uygulamayı kapatana kadar programın çalışmasını engelleyen bir döngüye girmesini sağlar.
