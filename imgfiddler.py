from tkinter import *
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.messagebox import showerror, showwarning
from collections import defaultdict
from PIL import Image, ImageTk
from copy import deepcopy

import cv2
import numpy as np



class App(Frame):
    """This is the main class of the application"""

    def __init__(self, parent=None):
        Frame.__init__(self, parent)
        self.parent = parent
        self.img = None
        self.second_img = None
        self.third_img = None
        self.window = None
        self.mod_window = None
        self.third_window = None
        self.hist_window = None
        self.hist_window2 = None
        self.canvas = None
        self.label = None
        self.mod_label = None
        self.third_label = None

        self.height = 0
        self.width = 0
        self.is_gray_scale = None

        self.master.title("ImFiddler")
        self.master.rowconfigure(5, weight=1)
        self.master.columnconfigure(5, weight=1)
        self.grid(sticky= W + E + N + S)

        self.load_button = Button(
            self, text="Load Image", command=self.load_image)
        self.load_button.grid(row=1, padx=3, pady=4, sticky = W + E)

        self.save_button = Button(
            self, text="Save Image", state=DISABLED, command=self.save_image)
        self.save_button.grid(row=2, padx=3, pady=4, sticky = W + E)

        self.copy_button = Button(
            self, text="Copy", state=DISABLED, command=self.copy_image)
        self.copy_button.grid(row=3, padx=3, pady=4, sticky = W + E)

        self.h_flip_button = Button(
            self, text="Horizontal Flip", state=DISABLED, command=self.h_flip)
        self.h_flip_button.grid(row=4, padx=3, pady=4, sticky = W + E)

        self.v_flip_button = Button(
            self, text="Vertical Flip", state=DISABLED, command=self.v_flip)
        self.v_flip_button.grid(row=5, padx=3, pady=4, sticky = W + E)

        self.grayscale_button = Button(
            self, text="Grayscale", state=DISABLED, command=self.grayscale)
        self.grayscale_button.grid(row=6, padx=3, pady=4, sticky = W + E)

        quantize_vcmd = (self.register(self.validate_quantize), '%d',
        '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        bright_vcmd = (self.register(self.validate_bright), '%d',
        '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        contrast_vcmd = (self.register(self.validate_contrast), '%d',
        '%i', '%P', '%s', '%S', '%v', '%V', '%W')

        quantize_lab = Label(self, text="Number of shades for quantization")
        quantize_lab.grid(row=7, sticky=W)
        self.quantize_entry = Entry(self, state=DISABLED, validate="key", validatecommand=quantize_vcmd)
        self.quantize_entry.grid(row=8)
        self.quantize_button = Button(self, text="Quantize", state=DISABLED, command=self.quantize)
        self.quantize_button.grid(row=9, padx=3, pady=4, sticky= W + E)

        self.hist_button = Button(self, state=DISABLED, text="Grayscale histogram", command=self.draw_histogram)
        self.hist_button.grid(row=10, padx=3, pady=4, sticky = W + E)

        bright_lab = Label(self, text="Brightness adjustment")
        bright_lab.grid(row=11)
        self.bright_entry = Entry(self, state=DISABLED, validate="key", validatecommand=bright_vcmd)
        self.bright_entry.grid(row=12)
        self.bright_button = Button(self, text="Apply brightness adjustment", state=DISABLED, command=self.brightness_adj)
        self.bright_button.grid(row=13, padx=3, pady=4, sticky = W + E)
        
        contrast_lab = Label(self, text="Contrast adjustment")
        contrast_lab.grid(row=14)
        self.contrast_entry = Entry(self, state=DISABLED, validate="key", validatecommand=contrast_vcmd)
        self.contrast_entry.grid(row=15)
        self.contrast_button = Button(self, text="Apply contrast adjustment", state=DISABLED, command=self.contrast_adj)
        self.contrast_button.grid(row=16, padx=3, pady=4, sticky = W + E)

        self.negative_button = Button(self, text="Negative", state=DISABLED, command=self.negative)
        self.negative_button.grid(row=17, padx=3, pady=4, sticky = W + E)

        self.negative_button = Button(self, text="Histogram equalization", state=DISABLED, command=self.histogram_equalization)
        self.negative_button.grid(row=18, padx=3, pady=4, sticky = W + E)

    def validate_quantize(self, action, index, value_if_allowed, prior_value,
                 text, validation_type, trigger_type, widget_name):
        """This function enables the quantize button when the something is put in the text field"""
        self.quantize_button.config(
            state=(NORMAL if value_if_allowed and text else DISABLED))
        if str.isdigit(value_if_allowed) or value_if_allowed == "":
            return True
        else:
            self.bell()
            return False
    
    def validate_bright(self, action, index, value_if_allowed, prior_value,
                 text, validation_type, trigger_type, widget_name):
        """This function enables the quantize button when the something is put in the text field"""
        self.bright_button.config(
            state=(NORMAL if value_if_allowed and text else DISABLED))
        if text in '0123456789-' or value_if_allowed == "":
            return True
        else:
            self.bell()
            return False

    def validate_contrast(self, action, index, value_if_allowed, prior_value,
                 text, validation_type, trigger_type, widget_name):
        """This function enables the quantize button when the something is put in the text field"""
        self.contrast_button.config(
            state=(NORMAL if value_if_allowed and text else DISABLED))
        if text in '0123456789.' or value_if_allowed == "":
            return True
        else:
            self.bell()
            return False

    def load_image(self):
        """Loads image from dialog box, rearranges its color channels and displays it"""
        imgname = askopenfilename(filetypes=(("JPEG files", (".jpeg, .jpg")),
                                             ("PNG files", "*.png"),
                                             ("Bitmap files", "*.bmp"),
                                             ("All files", "*.*")))

        if imgname:
            try:
                self.img = cv2.imread(imgname)
            except IOError:                     # <- naked except is a bad idea
                showerror("Open image file",
                          "Failed to read file '%s' \n" % imgname)
        # opencv2 uses BGR scheme, gotta translate it to RGB
        blue, green, red = cv2.split(self.img)

        self.remove_window()
        self.remove_second_window()
        self.remove_third_window
        self.remove_hist_window()
        self.remove_hist_window2

        self.img = cv2.merge((red, green, blue))

        self.window = Toplevel(root)
        self.window.title("Original Image")
        self.window.protocol('WM_DELETE_WINDOW', self.remove_window)

        im = Image.fromarray(self.img)
        imgtk = ImageTk.PhotoImage(image=im)
        self.label = Label(self.window, image=imgtk)
        self.label.image = imgtk
        self.label.pack()

        self.copy_button.config(state=NORMAL)

    def save_image(self):
        """Saves image located in the copy area"""
        if self.mod_window is not None:
            imgname = asksaveasfilename(filetypes=(("JPEG", ".jpeg"),
                                                   ("PNG", "*.png"),
                                                   ("Bitmap", "*.bmp")))

            sv_img = Image.fromarray(self.second_img)
            sv_img.save(imgname)
        else:
            showwarning("Warning", "No image to save.")

    def copy_image(self):
        """Copies the loaded image onto another window, on which we'll perform modifications"""
        if self.img is None:
            showerror("Oops!", "No image to copy.")
        elif isinstance(self.mod_window, Toplevel):
            if self.mod_window.state != "normal":
                showwarning("Warning", "Copy already exists.")
        else:
            self.mod_window = Toplevel(root)
            self.mod_window.protocol(
                'WM_DELETE_WINDOW', self.remove_second_window)
            self.mod_window.title("Modified Image")

            self.is_gray_scale = False

            self.second_img = self.img.copy()
            mod_im = Image.fromarray(self.second_img)
            imgtk = ImageTk.PhotoImage(image=mod_im)
            self.mod_label = Label(self.mod_window, image=imgtk)
            self.mod_label.image = imgtk
            self.mod_label.pack()

            self.v_flip_button.config(state=NORMAL)
            self.h_flip_button.config(state=NORMAL)
            self.grayscale_button.config(state=NORMAL)
            self.save_button.config(state=NORMAL)
            self.quantize_entry.config(state=NORMAL)
            self.hist_button.config(state=NORMAL)
            self.bright_entry.config(state=NORMAL)
            self.contrast_entry.config(state=NORMAL)
            self.negative_button.config(state=NORMAL)



    def h_flip(self):
        """Flips an image horizontally"""
        buffer = self.second_img.copy()
        if self.mod_window is not None:
            height, width = self.second_img.shape[0], self.second_img.shape[1]
            for i in range(height):
                for j in range(width):
                    # zero-based indexing
                    self.second_img[i, j] = buffer[i, width - j - 1]
            self.show_modified_image()
        else:
            showwarning("Warning", "No copy to modify.")

    def v_flip(self):
        """Flips an image horizontally"""
        buffer = self.second_img.copy()
        if self.mod_window is not None:
            height, width = self.second_img.shape[0], self.second_img.shape[1]
            for i in range(width):
                for j in range(height):
                    self.second_img[j, i] = buffer[height -
                                                   j - 1, i]  # account for zero-based indexing
            self.show_modified_image()
        else:
            showwarning("Warning", "No copy to modify.")

    def grayscale(self,mod_inplace=True):
        if self.mod_window is not None:
            height, width = self.second_img.shape[0], self.second_img.shape[1]
            #
            #   the code below is too slow; TODO: explore using cython to speed up 'for' loops
            #
            # for i in range(height):
            #     for j in range(width):
            #         new_val = (0.299 * (self.second_img[i][j][0])
            #                 + 0.587 * (self.second_img[i][j][1])
            #                 + 0.114 * (self.second_img[i][j][2]))
            #         new_val = np.round(new_val)
            #         self.second_img[i][j] = new_val
            rgb2grayscale = np.array([0.299, 0.587, 0.114])
            buffer = deepcopy(self.second_img)
            buffer[..., 0] = np.round(
                np.sum(buffer * rgb2grayscale, axis=-1)).astype('uint8')
            buffer[..., 1] = np.round(
                np.sum(buffer * rgb2grayscale, axis=-1)).astype('uint8')
            buffer[..., 2] = np.round(
                np.sum(buffer * rgb2grayscale, axis=-1)).astype('uint8')
            if mod_inplace:
                self.is_gray_scale = True
                self.second_img = buffer.copy()
                self.show_modified_image()
            else:
                return buffer
        else:
            showwarning("Warning", "No copy to modify.")

    def quantize(self):
        if self.mod_window is not None and self.is_gray_scale:
            shades = self.quantize_entry.get()
            try:
                shades = int(shades)
            except ValueError:
                showerror(
                    "Error", "Invalid input. Must type a number between 0 and 255.")

            if shades not in range(0, 256):
                showerror(
                    "Error", "Invalid input. Must type a number between 0 and 255.")
            else:
                # build a hash table mapping the values to the quanta
                bin_hash = {}
                interval = int(np.ceil(255 / shades))
                for bin in range(interval, shades * interval + 1, interval):
                    for value in range(bin - interval, bin + 1):
                        if bin > 255:
                            bin_hash[value] = 255
                        else:
                            bin_hash[value] = bin

                buffer = self.second_img.copy()
                height, width = self.second_img.shape[0], self.second_img.shape[1]
                for i in range(height):
                    for j in range(width):
                        buffer[i][j] = bin_hash[self.second_img[i][j][0]]

                self.second_img = buffer.copy()
                self.show_modified_image()
        else:
            if self.mod_window is None:
                showwarning("Warning", "No copy to modify.")
            elif not self.is_gray_scale:
                showwarning("Warning", "Apply grayscale before trying this.")

    def draw_histogram(self):
        if self.mod_window is not None:            
            self.hist_window = Toplevel(root)
            self.hist_window.protocol('WM_DELETE_WINDOW', self.remove_hist_window)
            self.hist_window.title("Histogram")
            height,width=300,512
            self.canvas = Canvas(self.hist_window, height=256, width=256, bg="azure3")
            self.canvas.pack(expand=YES, fill=BOTH)

            histogram = self.calculate_grayscale_histogram()
            number_of_stripes = 2 * 256 + 1
            bar_width = width/number_of_stripes
            unit_height = 256 / max(histogram.values())

            for i in range(256):
                self.canvas.create_rectangle(
                    (2 * i + 1) * bar_width, height - unit_height,
                    (2 * i + 2) * bar_width, height - ((histogram[i] + 1) * unit_height),
                    fill = 'black'
                )                

        else:
            if self.mod_window is None:
                showwarning("Warning", "No copy to modify.")

    def calculate_grayscale_histogram(self):
        height, width = self.second_img.shape[0], self.second_img.shape[1]
        histogram = defaultdict(int)
        buffer = self.grayscale(mod_inplace=False)

        for i in range(height):
            for j in range(width):
                histogram[buffer[i][j][0]] += 1
        
        return histogram

    def histogram_equalization(self):
        if self.mod_window is not None:
            # self.hist_window2 = Toplevel(root)
            # self.hist_window2.protocol('WM_DELETE_WINDOW', self.remove_hist_window2)
            # self.hist_window2.title("Histogram")
            self.third_window = Toplevel(root)
            self.third_window.protocol('WM_DELETE_WINDOW', self.remove_third_window)
            self.third_window.title("Image after equalization")

            if self.is_gray_scale:
                histogram = self.calculate_grayscale_histogram()
                alpha = 255 / (self.second_img.shape[0] * self.second_img.shape[1])
                cum_histogram = {}
                cum_histogram[0] = alpha * histogram[0]
                for i in range(1,256):
                    cum_histogram[i] = cum_histogram[i-1] + (alpha * histogram[i])
                
                self.third_img = self.second_img.copy()
                img_height, img_width = self.third_img.shape[0], self.third_img.shape[1]

                for i in range(img_height):
                    for j in range(img_width):
                        self.third_img[i][j] = cum_histogram[self.second_img[i][j][0]]

                im = Image.fromarray(self.third_img)
                imgtk = ImageTk.PhotoImage(image=im)
                self.third_label = Label(self.third_window, image=imgtk)
                self.third_label.image = imgtk
                self.third_label.pack()

    def brightness_adj(self):
        if self.mod_window is not None:
            adjust_value = self.bright_entry.get()
            try:
                adjust_value = int(adjust_value)
            except ValueError:
                showerror(
                    "Error", "Invalid input. Must type a number between -255 and 255.")
            if adjust_value not in range (-255,256):
                showerror(
                    "Error", "Invalid input. Must type a number between -255 and 255.")
            else:
                height, width = self.second_img.shape[0], self.second_img.shape[1]
                buffer = self.second_img.copy()
                #
                # This is PAINFULLY slow, but it works
                #
                # for c in range(3):
                #     for i in range(height):
                #         for j in range(width):
                #             if buffer[i][j][c] + adjust_value > 255:
                #                 buffer[i][j][c] = 255
                #             elif buffer[i][j][c] + adjust_value < 0:
                #                 buffer[i][j][c] = 0
                #             else:
                #                 buffer[i][j][c] += adjust_value
                #
                #numpy trickery to simulate a ternary operator, MUCH faster than manually operating on individual pixels
                buffer = np.where((255 - self.second_img) < adjust_value, 255, 
                                    (np.where((self.second_img + adjust_value) < 0, 0, self.second_img + adjust_value)))

                self.second_img = buffer.copy()
                self.show_modified_image()
        else:
            if self.mod_window is None:
                showwarning("Warning", "No copy to modify.")

    def contrast_adj(self):
        if self.mod_window is not None:
            adjust_value = self.contrast_entry.get()
            try:
                adjust_value = float(adjust_value)
            except ValueError:
                showerror(
                    "Error", "Invalid input. Must type a number between 0 and 255.")
            if int(adjust_value) not in range (0,256):
                showerror(
                    "Error", "Invalid input. Must type a number between 0 and 255.")
            else:
                height, width = self.second_img.shape[0], self.second_img.shape[1]
                buffer = self.second_img.copy()
                #
                # slower than a turtle.
                #
                # for c in range(3):
                #     for i in range(height):
                #         for j in range(width):
                #             if buffer[i][j][c] * adjust_value > 255:
                #                 buffer[i][j][c] = 255
                #             elif buffer[i][j][c] * adjust_value < 0:
                #                 buffer[i][j][c] = 0
                #             else:
                #                 buffer[i][j][c] *= adjust_value

                buffer = np.where((self.second_img * adjust_value) > 255, 255, self.second_img * adjust_value)
            
                self.second_img = buffer.copy()
                self.show_modified_image()
        else:
            if self.mod_window is None:
                showwarning("Warning", "No copy to modify.")

    def negative(self):
        if self.mod_window is not None:
            height, width = self.second_img.shape[0], self.second_img.shape[1]

            for i in range(height):
                for j in range(width):
                    self.second_img[i,j,:] = 255 - self.second_img[i,j,:]

            #self.second_img = 255 - self.second_img

            self.show_modified_image()
        else:
            if self.mod_window is None:
                showwarning("Warning", "No copy to modify.")  


    def show_modified_image(self):
        image = Image.fromarray(self.second_img.astype('uint8'))
        imgtk = ImageTk.PhotoImage(image=image)
        self.mod_label.configure(image=imgtk)
        self.mod_label.image = imgtk

    def remove_second_window(self):
        """Destroys second image window and sets it to none"""
        if self.mod_window is not None:
            self.mod_window.destroy()
            self.mod_window = None
    
    def remove_third_window(self):
        """Destroys second image window and sets it to none"""
        if self.third_window is not None:
            self.third_window.destroy()
            self.third_window = None

    def remove_window(self):
        """Destroy first image window and sets it to none"""
        if self.window is not None:
            self.window.destroy()
            self.window = None

    def remove_hist_window(self):
        """Destroy first image window and sets it to none"""
        if self.hist_window is not None:
            self.hist_window.destroy()
            self.hist_window = None
    
    def remove_hist_window2(self):
        """Destroy first image window and sets it to none"""
        if self.hist_window2 is not None:
            self.hist_window2.destroy()
            self.hist_window2 = None

if __name__ == "__main__":
    root = Tk()
    frame = App(root)
    root.mainloop()
