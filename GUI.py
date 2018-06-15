from tkinter import *
from tkinter.ttk import *
from queue import *
import cv2
import numpy as np
from main import main, show_images
from PIL import Image, ImageTk
import threading
from pywt import families
import math

class Window(Frame):
	def __init__(self, master=None):
		
		
		Frame.__init__(self, master)   
		self.master = master
		
		self.init_window()

 
	def init_window(self):
		self.master.title("Thermal Fusion GUI")

		
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
		
		#entry = Entry(self)
		#entry.grid(column=0, row=1)
		#self.entry = entry
		
		button = Button(self, text = "Show RGB and IR images", command = self.showImg)
		button.grid(column=1, row=1)
		self.button = button
		
		fusion = Button(self, text = "Start fusion process", command = self.startFusion)
		fusion.grid_forget()
		self.fusion = fusion
		
		text = Text(self, width=70, height=15)
		text.grid(column=0, row=3)
		self.text=text
				
		self.progressbar = Progressbar(self, orient = HORIZONTAL, mode = 'indeterminate',length=100)
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
		
	def showImg(self):
		#s = self.entry.get()
		
		RGB = ImageTk.PhotoImage(Image.open('examples/rgb.jpg'))
		IR = ImageTk.PhotoImage(Image.open('examples/ir.png'))
		
		self.canvasRGB.create_image(0, 0, image=RGB, anchor=NW)
		self.canvasIR.create_image(0, 0, image=IR, anchor=NW)
		
		self.canvasRGB.image = RGB
		self.canvasIR.image = IR
		
		self.fusion.grid(column=0, row=2)
		
		self.queue = Queue()
		
	def startFusion(self):
		#s = self.entry.get()
	
		self.progressbar.start()
		
		
		ThreadedTask(self.queue, self.variable.get(), 
					 self.waveletVar.get(), self.dictWavelets[self.waveletVar.get()]).start()
		self.master.after(2000, self.process_queue)

	def process_queue(self):
		try:
			arr, Results, Titles = self.queue.get(0)
			
			self.displayMetrics(arr)
			show_images(Results, 3, Titles)
			
			
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
	
	def __init__(self, queue, strat, wavelet, shortWavelet):
		threading.Thread.__init__(self)
		
		self.queue = queue
		self.strat = strat
		self.wavelet = wavelet
		self.shortWavelet = shortWavelet
		
		print("Thread started !")
		
	def run(self):
		rgb_path = 'examples/rgb.jpg'
		ir_path = 'examples/ir.png'
		Res = main(rgb_path, ir_path, self.strat, self.shortWavelet)
		
		Res[0].insert(0, "Metrics for (strategy " + self.strat + ", wavelet " + self.wavelet + ")")
		Res[0].append("**************************")

		self.queue.put(Res)
		
		print("Thread (strategy : " + self.strat + ", wavelet " + self.wavelet + ") ended ! ")


if __name__ == '__main__':
	root = Tk()

	root.geometry("1200x600+0+0")

	app = Window(root)
	
	root.mainloop()
