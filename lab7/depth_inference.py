# Copyright 2021 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see https://www.apache.org/licenses/LICENSE-2.0 for details.
# SPDX-License-Identifier: Apache-2.0
#
# Author: Viviane Potocnik <vivianep@iis.ee.ethz.ch> (ETH Zurich)

from pickletools import uint8

import matplotlib

matplotlib.use("Agg")
# matplotlib.use("TkAgg")
import sys
import time
import tkinter as tk

import customtkinter as ctk
import cv2
import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk


def draw_gui(classes, correct_count, test_len, x, class_idx, pred):
    root = ctk.CTk()
    root.title("Real-time inference")
    imgplot = plt.imshow(x.astype(np.uint8))
    f = Figure()
    a = f.add_subplot(111)
    a.imshow(x.astype(np.uint8))
    # store figure to file
    f.savefig("test.png")
    image = Image.open("test.png")
    # photo = ImageTk.PhotoImage(image)
    photo = ctk.CTkImage(image, size=(200, 200))
    root.grid_rowconfigure((0, 1, 2, 3, 4), weight=1)
    root.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)
    color = "#0B8457" if (np.argmax(pred) == class_idx) else "#F24C4C"
    darkgrey = "#2D2D2D"
    yellow = "#FFC107"
    quit = "#DA0037"
    next = "#3E497A"
    next2 = "#3AB4F2"
    ctk.CTkButton(master=root, image=photo, text=None).grid(
        row=0, column=0, padx=(10, 5), pady=10, columnspan=2
    )
    # pre label should be on the left of the grid
    text_var = tk.StringVar(value=f"Prediction: {classes[np.argmax(pred)]}")
    pred_label = ctk.CTkButton(
        root,
        textvariable=text_var,
        font=("Helvetica", 18, "bold"),
        fg_color=color,
        text_color="white",
        hover=False,
    )
    pred_label.grid(row=1, column=0, padx=(10, 5), pady=10, columnspan=1, sticky="nsew")
    target_label = ctk.CTkButton(
        root,
        text=f"Target: {classes[class_idx]}",
        font=("Helvetica", 18, "bold"),
        fg_color=color,
        text_color="white",
        hover=False,
    )
    target_label.grid(
        row=1, column=1, padx=(5, 10), pady=10, columnspan=1, sticky="nsew"
    )
    accuracy = (correct_count / test_len) * 100
    text_var = tk.StringVar(value=f"Accuracy: {accuracy:.2f}%")
    accuracy_label = ctk.CTkButton(
        root,
        textvariable=text_var,
        font=("Helvetica", 18, "bold"),
        fg_color=yellow,
        text_color=darkgrey,
        hover=False,
    )
    accuracy_label.grid(
        row=2, column=0, padx=(10, 10), pady=10, columnspan=2, sticky="nsew"
    )
    # add button to go to the next image
    ctk.CTkButton(
        master=root,
        text="Next",
        command=root.destroy,
        fg_color=next,
        font=("Helvetica", 18, "bold"),
        hover_color=next2,
    ).grid(row=3, column=0, padx=(10, 5), pady=10, columnspan=2, sticky="nsew")
    ctk.CTkButton(
        master=root,
        text="Quit",
        command=sys.exit,
        fg_color=next,
        hover_color=quit,
        font=("Helvetica", 18, "bold"),
    ).grid(row=4, column=0, padx=(10, 5), pady=10, columnspan=2, sticky="nsew")

    root.mainloop()


def main(x_test, y_test, ser: serial.Serial):
    correct_count = 0
    # define how many images from the test set to send to the MCU
    test_len = 2
    img_size = (64, 64)
    # img_size = (64,64)
    num_pixels_in = np.product(img_size) * 3
    num_pixels_out = np.product(img_size) * 1

    _ = get_pred(ser, img_size, num_pixels_in, x_test[0], num_pixels_out=num_pixels_out)
    time.sleep(2)

    ser.reset_input_buffer()
    ser.reset_output_buffer()

    for req_img, class_idx in zip(x_test[:test_len], y_test[:test_len]):
        req_img, pred = get_pred(
            ser, img_size, num_pixels_in, req_img, num_pixels_out=num_pixels_out
        )
        print(pred.shape)

        # print(
        #     f"Target: {classes[class_idx]}, Prediction (from MCU): {classes[np.argmax(pred)]}"
        # )
        # if np.argmax(pred) == class_idx:
        #     correct_count += 1

        # # use_gui = True
        # use_gui = False
        # if use_gui:
        #     draw_gui(
        #         classes=classes,
        #         correct_count=correct_count,
        #         test_len=test_len,
        #         x=req_img,
        #         class_idx=class_idx,
        #         pred=pred,
        #     )
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    # print(f"Accuracy: {(correct_count / test_len) * 100:.2f}%")


def get_pred(ser, img_size, num_pixels_in, req_img, num_pixels_out=None):
    if num_pixels_out is None:
        num_pixels_out = num_pixels_in
    req_img = cv2.resize(req_img, img_size)
    req_img = (req_img * 255).astype("uint8")
    ser.write(req_img.tobytes())
    ser.flush()
    time.sleep(15)
    resp_img = ser.read(num_pixels_in)
    resp_img = np.frombuffer(resp_img, dtype=np.uint8)
    assert (
        len(resp_img) == num_pixels_in
    ), f"Expected {num_pixels_in} bytes, got {len(resp_img)}"
    time.sleep(15)
    pred = ser.read(num_pixels_out)
    pred = np.frombuffer(pred, dtype=np.uint8)
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].imshow(req_img)
    axs[1].imshow(resp_img.reshape(img_size))
    axs[2].imshow(pred.reshape(img_size))

    axs[0].set_title("Request Image")
    axs[1].set_title("Response Image")
    axs[2].set_title("Depth Prediction")

    plt.savefig("depth_test.png")
    pred = pred.reshape(img_size)
    req_img = req_img.reshape(img_size)
    return req_img, pred


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name", type=str, required=False, default="fmnist")
    args, _ = parser.parse_known_args()
    print(args)

    x_test = np.load(
        "/media/master/wext/msc_studies/second_semester/microcontrollers/project/stm32/code/test_data/x_test_depth.npy"
    )
    y_test = np.load(
        "/media/master/wext/msc_studies/second_semester/microcontrollers/project/stm32/code/test_data/y_test_depth.npy"
    ).squeeze()

    ctk.set_appearance_mode("dark")
    # ctk.set_default_color_theme("dark-blue")

    print(f"Loaded x with shape: {x_test.shape}")
    print(f"Loaded y with shape: {y_test.shape}")

    serial_port = "/dev/ttyACM0"
    ser = serial.Serial(port=serial_port, baudrate=115200, timeout=20, buffer_size=12400)
    # flush the serial port
    ser.flush()
    ser.flushInput()
    ser.flushOutput()
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    try:
        main(x_test, y_test, ser)
    finally:
        ser.close()
