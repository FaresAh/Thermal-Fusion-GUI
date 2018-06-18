from tkinter import *
from tkinter.ttk import *
from queue import *
import cv2
import numpy as np
from main import main, show_images
from PIL import Image, ImageTk
import threading
from pywt import families
from tkinter import filedialog
import math
import os

class Window(Frame):
	def __init__(self, master=None):
		
		
		Frame.__init__(self, master)   
		self.master = master
		
		self.init_window()

 
	def init_window(self):
		self.master.title("Thermal Fusion GUI")
		
		menu = Menu(self)
		self.master.config(menu=menu)
		
		filemenu = Menu(menu)
		
		menu.add_cascade(label="File", menu=filemenu)
		filemenu.add_command(label="Open RGB Image", command=lambda: self.loadFile(True))
		filemenu.add_command(label="Open IR Image", command=lambda: self.loadFile(False))
		filemenu.add_command(label="Open Example Images", command=self.loadDefault)
		filemenu.add_separator()
		filemenu.add_command(label="Exit", command=self.client_exit)
		
		self.pack(fill=BOTH, expand=1)
		
		canvasRGB = Canvas(self, width=382, height=288, bg='white')
		canvasRGB.grid(column=0, row=0)
		
		RGB_id = canvasRGB.create_text(191, 142, fill="blue", tag="RGB")
		canvasRGB.insert(RGB_id, 12, "RGB")
		
		self.canvasRGB = canvasRGB
		
		canvasIR = Canvas(self, width=382, height=288, bg='white')
		canvasIR.grid(column=1, row=0)
		
		IR_id = canvasIR.create_text(191, 142, fill="blue", tag="IR")
		canvasIR.insert(IR_id, 12, "IR")
		
		self.canvasIR = canvasIR
		
		button = Button(self, text = "Load example images", command=self.loadDefault)
		button.grid(column=2, row=0)
		self.button = button
		
		buttonRGB = Button(self, text = "Select RGB Image", command=lambda: self.loadFile(True))
		buttonRGB.grid(column=0, row=1)
		self.buttonRGB = buttonRGB

		buttonIR = Button(self, text = "Select IR Image", command=lambda: self.loadFile(False))
		buttonIR.grid(column=1, row=1)
		self.buttonIR = buttonIR
		
		fusion = Button(self, text = "Start fusion process", command=self.startFusion)
		fusion.grid(column=0, row=2)
		self.fusion = fusion
		
		text = Text(self, width=70, height=15)
		text.grid(column=0, row=3)
		self.text=text
				
		self.progressbar = Progressbar(self, orient=HORIZONTAL, mode='indeterminate',length=100)
		self.progressbar.grid(column=1, row=3)

		options = ["All", "Min", "Max", "Mean", "Entropy", "MACD", "Edge", "Deviation"]
		
		self.variable = StringVar(self)
		self.variable.set(options[0])
		
		self.dropdown = OptionMenu(self, self.variable, options[0], *options)
		self.dropdown.grid(column=3, row=1)
		
		# Only the first wavelets are functionnal
		wavelets = families(short=False)[:7]
		shortWavelets = families()[:7]
		
		self.dictWavelets = dict(zip(wavelets, shortWavelets))
		
		self.waveletVar = StringVar(self)
		self.waveletVar.set(wavelets[1])
		
		self.waveletDropdown = OptionMenu(self, self.waveletVar, wavelets[1], *wavelets)
		self.waveletDropdown.grid(column=4, row=1)
		
		self.queue = Queue()
		
		self.rgb_path = ""
		self.ir_path = ""
	
	def loadDefault(self):
		self.rgb_path = 'examples/rgb.jpg'
		self.ir_path = 'examples/ir.png'
		
		self.showImg()
	
	def loadFile(self, is_rgb = True):
		filename = filedialog.askopenfilename(initialdir=os.getcwd(),title='Choose a file', filetypes = [("Image File", "*.jpg *.png")])

		if filename:
			if (is_rgb):
				self.rgb_path = filename 
			else:
				self.ir_path = filename
			
			self.showImg()
		
	def showImg(self):
		if (self.rgb_path):
			RGB = ImageTk.PhotoImage(Image.open(self.rgb_path))
			self.canvasRGB.create_image(0, 0, image=RGB, anchor=NW)
			self.canvasRGB.image = RGB
		
		if (self.ir_path):
			IR = ImageTk.PhotoImage(Image.open(self.ir_path))
			self.canvasIR.create_image(0, 0, image=IR, anchor=NW)
			self.canvasIR.image = IR
		
	def startFusion(self):
		if (self.rgb_path and self.ir_path):
			self.progressbar.start()
			
			
			ThreadedTask(self.rgb_path, self.ir_path, self.queue, self.variable.get(), 
						 self.waveletVar.get(), self.dictWavelets[self.waveletVar.get()]).start()
			self.master.after(2000, self.process_queue)

	def process_queue(self):
		try:
			arr, Results, Titles = self.queue.get(0)
			
			self.displayMetrics(arr)
			show_images(Results, math.ceil(len(Results) / 3.0), Titles)
			
			
			if (self.queue.empty()):
				self.progressbar.stop()
			else:
				self.master.after(2000, self.process_queue)
			
		except Empty:
			self.master.after(2000, self.process_queue)
			
	def displayMetrics(self, array):
		for s in array:
			string = s + '\n'
			self.text.insert(END, string)

	def client_exit(self):
		exit()

class ThreadedTask(threading.Thread):
	
	def __init__(self, rgb_path, ir_path, queue, strat, wavelet, shortWavelet):
		threading.Thread.__init__(self)
		
		self.rgb_path = rgb_path
		self.ir_path = ir_path
		self.queue = queue
		self.strat = strat
		self.wavelet = wavelet
		self.shortWavelet = shortWavelet
		
		print("Thread started !")
		
	def run(self):
		Res = main(self.rgb_path, self.ir_path, self.strat, self.shortWavelet)
		
		Res[0].insert(0, "Metrics for (strategy " + self.strat + ", wavelet " + self.wavelet + ")")
		Res[0].append("**************************")

		self.queue.put(Res)
		
		print("Thread (strategy : " + self.strat + ", wavelet " + self.wavelet + ") ended ! ")


if __name__ == '__main__':
	root = Tk()

	root.geometry("1200x600+0+0")

	app = Window(root)
	
	root.mainloop()
