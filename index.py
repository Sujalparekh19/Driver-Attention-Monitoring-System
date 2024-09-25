from tkinter import *
import os

def d_dtcn(shape_predictor_path):
    root = Tk()
    root.configure(background="white")

    def function1():
        os.system(f"python drowsiness_detection.py --shape_predictor {shape_predictor_path}")
        exit()

    def function2():
        os.system(f"python android_cam.py --shape_predictor {shape_predictor_path}")
        exit()

    root.title("DROWSINESS DETECTION")
    Label(root, text="DROWSINESS DETECTION", font=("times new roman", 20), fg="black", bg="aqua", height=2).grid(row=2, rowspan=2, columnspan=5, sticky=N+E+W+S, padx=5, pady=10)
    Button(root, text="Run using web cam", font=("times new roman", 20), bg="#0D47A1", fg='white', command=function1).grid(row=5, columnspan=5, sticky=W+E+N+S, padx=5, pady=5)
    Button(root, text="Run using phone cam", font=("times new roman", 20), bg="#0D47A1", fg='white', command=function2).grid(row=7, columnspan=5, sticky=W+E+N+S, padx=5, pady=5)
    Button(root, text="Exit", font=("times new roman", 20), bg="#0D47A1", fg='white', command=root.destroy).grid(row=9, columnspan=5, sticky=W+E+N+S, padx=5, pady=5)

    root.mainloop()
