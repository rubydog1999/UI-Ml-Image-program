import time
from tkinter import *
import numpy as np
import pandas
from PIL import ImageTk, Image
from tkinter import messagebox
from tkinter import filedialog
import os
from random import randrange
from math import sqrt
import cv2
from matplotlib import animation
from pandastable import Table
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



root = Tk()
root.title("Machine Learning App")
root.iconbitmap('d:/Pycharm/pycharmproject/Clustering/Icon.ico')
root['background'] = "#856ff8"

class FullScreenApp(object):
    def __init__(self, master, **kwargs):
        self.master = master
        pad = 3
        self._geom = '200x200+0+0'
        master.geometry("{0}x{1}+0+0".format(
            master.winfo_screenwidth() - pad, master.winfo_screenheight() - pad))
        master.bind('<Escape>', self.toggle_geom)

    def toggle_geom(self, event):
        geom = self.master.winfo_geometry()
        print(geom, self._geom)
        self.master.geometry(self._geom)
        self._geom = geom


app = FullScreenApp(root)
root.bind('<Escape>', lambda event: exitApp())
root.bind('<F11>', lambda event: root.state('zoomed'))
def doNothing():
    myLabel = Label(root, text="button click")
    myLabel.place(re)


def doDescription(value):
    value = clicked.get()
    if value == "K-mean":
        myDes = Label(description_layer, text="This formula is based on k-mean clustering \n ."
                                              " We will choose any image and enter the number k (number of clusters) we want\n"
                                              " to process. The program calculates and labels the pixels according to \n "
                                              "the k-mean formula and produces a table of results\n (Cluster, Pixel RGB, Pixel XY and Distance from the point to center mass)",
                      bg="black",fg = "white")
        myDes.place(relwidth=1, relheight=1)
    elif value == "Image segmentation":
        myDes = Label(description_layer, text="Image Segmentation involves converting an image into a collection of regions of pixels\n"
                                              " that are represented by a mask or a labeled image\n."
                                              " By dividing an image into segments,\n"
                                              " you can process only the important segments of the image\n"
                                              " instead of processing the entire image. So here we get the k value and image\n"
                                              "​then we use the k-mean clustering algorithm to perform image segmentation",
                      bg = "black", fg = "white")
        myDes.place(relwidth=1, relheight=1)

def exitApp():
    response = messagebox.askyesno("Machine Learning App", "Do you want exit")
    if response == 1:
        root.destroy()

def openimg():
    global my_img
    global filename
    filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select A File", filetypes=(
    ("jpg images", ".jpg"), ("png images", ".png"), ("all files", "*.*")))
    if not filename:
        return
    my_img = ImageTk.PhotoImage(Image.open(filename))
    my_label_img = Label(imgae_layer, image=my_img, bg ="black")
    my_label_img.place(relwidth=1, relheight=1)
root.protocol("WM_DELETE_WINDOW", exitApp)
def run():
    values = clicked.get()
    #run k-mean method
    if values == "K-mean":
        myRgraphic = Label(result_graphic, text="Result Graphic")
        myRgraphic.place(relwidth=1, relheight=1)
        k_val = int(e.get())
        image = cv2.imread(filename=filename)
        def distance(point1, point2):
            return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)

        def point_rgb(point, img):
            return img[point[0], point[1]]

        def k_means(img, k, max_iter):
            Array = []
            centers = []
            clusters = {}

            for i in range(k):
                center = (randrange(len(img)), randrange(len(img[0])))
                if center not in centers:
                    centers.append(center)
                    clusters[center] = []
            for i in range(max_iter):
                for x in range(len(img)):
                    for y in range(len(img[0])):
                        rgb1 = img[x][y]
                        min_distance = distance(rgb1, point_rgb(centers[0], img=img))
                        best_cluster = centers[0]
                        for center in centers:
                            rgb2 = point_rgb(center, img)
                            dist = distance(rgb1, rgb2)
                            if dist < min_distance:
                                min_distance = dist
                                best_cluster = center
                        clusters[best_cluster].append((rgb1, [x, y]))

                avg = []
                for c in clusters:
                    if len(clusters[c]) == 0:
                        break
                    sum_red = 0
                    sum_blue = 0
                    sum_green = 0
                    for point in clusters[c]:
                        sum_red += point[0][0]
                        sum_green += point[0][1]
                        sum_blue += point[0][2]
                    red = sum_red / len(clusters[c])
                    blue = sum_blue / len(clusters[c])
                    green = sum_green / len(clusters[c])
                    avg.append([red, green, blue])

                centers = []
                clusters = {}
                for a in avg:
                    min_distance = distance(a, img[0][0])
                    best_clust = img[0][0]
                    for row in range(len(img)):
                        for column in range(len(img[0])):
                            new_dist = distance(a, img[row][column])
                            if new_dist < min_distance:
                                min_distance = new_dist
                                best_clust = (row, column)
                    centers.append(best_clust)
                    clusters[best_clust] = []

            for x in range(len(img)):
                for y in range(len(img[0])):
                    rgb1 = img[x][y]
                    min_distance = distance(rgb1, point_rgb(centers[0], img=img))
                    best_cluster = centers[0]
                    for center in centers:
                        rgb2 = point_rgb(center, img)
                        dist = distance(rgb1, rgb2)
                        if dist < min_distance:
                            min_distance = dist
                            best_cluster = center
                    clusters[best_cluster].append((best_cluster,rgb1, [x, y], min_distance))
            return clusters
        print()
        cl = k_means(img=image, k=k_val, max_iter=10)
        table_element_2 = []
        for j in cl:
            for z in cl[j]:
                table_element_2.append(z)
        row = len(table_element_2)
        column_names = ["Cluster", "Pixel RGB", "pixel X Y", "Distance"]
        global table_cluster
        table_cluster = pandas.DataFrame(table_element_2, columns=column_names, index=range(row))
        table_final = Table(result_text, dataframe=table_cluster,  showstatusbar=True)
        table_final.show()

#run Image segmentation
    elif values == "Image segmentation":
        global figure
        myRtext = Label(result_text, text="ResultText")
        myRtext.place(relwidth=1, relheight=1)
        k_val = int(e.get())
        image = cv2.imread(filename=filename)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        vectorized = img.reshape((-1, 3))
        vectorized = np.float32(vectorized)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = k_val
        attempts = 10
        ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result_image = res.reshape((img.shape))
        figure_size = 6
        figure = plt.figure(figsize=(figure_size, figure_size))
        plt.subplot(1, 2, 1), plt.imshow(img)
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(1, 2, 2), plt.imshow(result_image)
        plt.title('Segmented Image K = %i' % K), plt.xticks([]), plt.yticks([])
        # plt.show()
        canv = FigureCanvasTkAgg(figure, result_graphic)
        canv.draw()
        get_widz = canv.get_tk_widget()
        get_widz.pack()



    else:
        rep = messagebox.showwarning("Warning","Must choose method !")
        Label(root,text = rep)

def exportCSV():
    values = clicked.get()
    if values == "K-mean":
            export_file_path = filedialog.asksaveasfilename(defaultextension='.csv')
            table_cluster.to_csv(export_file_path, index=True, header=True)
    elif values == "Image segmentation":

            save_result_imgae = filedialog.asksaveasfilename(defaultextension='.png')
            figure.savefig(save_result_imgae)
    else:
        rep = messagebox.showwarning("Warning", "No result to save")
        Label(root, text=rep)



# Main Menu
menu = Menu(root)
root.config(menu=menu)
subMenu = Menu(menu)
menu.add_cascade(label="File", menu=subMenu)
subMenu.add_command(label="Open File", command=openimg)
subMenu.add_command(label="Save Result", command=exportCSV)
otherMenu = Menu(menu)
menu.add_cascade(label="Other", menu=otherMenu)
otherMenu.add_command(label="Exit", command=exitApp)


# Toolbar
toolbar = Frame(root, bg="#4267B2", padx=25, pady=20)
inserBut = Button(toolbar, text="Image Open", command=openimg, padx=15, pady=15)
inserBut.pack(side=LEFT, padx=2, pady=2)
runBut = Button(toolbar, text="Run", command=run, padx=15, pady=15)
runBut.pack(side=LEFT, padx=30, pady=2)
imp_kmean = Label(toolbar, text="Import number of K")
imp_kmean.pack(side=LEFT,padx=5,pady=2,ipadx=20, ipady=5)
e = Entry(toolbar, width=50)
e.pack(side=LEFT, padx=2, pady=10,ipady =5,ipadx=20)
e.insert(0, "")
clicked = StringVar()
clicked.set("Method")
options = ["K-mean", "Image segmentation"]
drop = OptionMenu(toolbar, clicked, *options, command=doDescription)
drop.config(width=30, height=2)
drop.pack(side=RIGHT, padx=2, pady=2)
toolbar.pack(side=TOP, fill=X)

# body app
# creat Frame
imgae_layer = Frame(root, bg="#DEE1E6", padx=5, pady=5)
myImgs = Label(imgae_layer, text="Image")
myImgs.place(relwidth=1, relheight=1)
description_layer = Frame(root, bg="#DEE1E6", padx=5, pady=5)
myDes = Label(description_layer, text="Method Description")
myDes.place(relwidth=1, relheight=1)
result_text = Frame(root, bg="#DEE1E6", padx=5, pady=5)
myRtext = Label(result_text, text="ResultText")
myRtext.place(relwidth=1, relheight=1)
result_graphic = Frame(root, bg="#DEE1E6", padx=5, pady=5)
myRgraphic = Label(result_graphic, text="Result Graphic")
myRgraphic.place(relwidth=1, relheight=1)
imgae_layer.place(relx=0.25, rely=0.20, relwidth=0.35, relheight=0.35, anchor='n')
description_layer.place(relx=0.75, rely=0.20, relwidth=0.35, relheight=0.35, anchor='n')
result_text.place(relx=0.25, rely=0.59, relwidth=0.35, relheight=0.35, anchor='n')
result_graphic.place(relx=0.75, rely=0.59, relwidth=0.35, relheight=0.35, anchor='n')
root.mainloop()
