import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import config
from simple_object_detector import detect_objects_simple

class ImageAreaDetectorApp:
    def __init__(self, master):
        self.master = master
        master.title(config.APP_WINDOW_TITLE)
        master.geometry(config.APP_WINDOW_GEOMETRY) # Set initial window size

        self.selected_image_path = tk.StringVar()
        self.selected_image_path.set("No image selected.")

        self.create_widgets()

    def create_widgets(self):
        # Frame for file selection
        file_frame = tk.LabelFrame(self.master, text="Image Selection", padx=10, pady=10)
        file_frame.pack(pady=10, padx=10, fill="x")

        self.select_button = tk.Button(file_frame, text="Select Image", command=self.select_image)
        self.select_button.pack(side=tk.LEFT, padx=5)

        self.path_label = tk.Label(file_frame, textvariable=self.selected_image_path, wraplength=400)
        self.path_label.pack(side=tk.LEFT, fill="x", expand=True)

        # Frame for actions
        action_frame = tk.LabelFrame(self.master, text="Actions", padx=10, pady=10)
        action_frame.pack(pady=10, padx=10, fill="x")

        self.find_areas_button = tk.Button(action_frame, text="Find Areas", command=self.run_find_areas)
        self.find_areas_button.pack(pady=5)

        # Output Text Area
        self.output_label = tk.Label(self.master, text="Output:")
        self.output_label.pack(pady=5)

        self.output_text = scrolledtext.ScrolledText(self.master, width=70, height=10, wrap=tk.WORD)
        self.output_text.pack(pady=5, padx=10, fill="both", expand=True)
        self.output_text.insert(tk.END, "Ready to process images.\n")

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image File",
            filetypes=config.APP_FILETYPES
        )
        if file_path:
            self.selected_image_path.set(file_path)
            self.log_message(f"Selected image: {file_path}")
        else:
            self.log_message("No image selected.")

    def run_find_areas(self):
        image_path = self.selected_image_path.get()
        if not image_path or image_path == "No image selected.":
            messagebox.showwarning("No Image", "Please select an image first.")
            self.log_message("Attempted to find areas without selecting an image.")
            return

        if not os.path.exists(image_path):
            messagebox.showerror("File Not Found", f"The selected image file does not exist: {image_path}")
            self.log_message(f"Error: File not found: {image_path}")
            return

        self.log_message(f"Attempting to find areas in: {image_path}")
        try:
            detect_objects_simple(image_path)
            self.log_message("Image processing completed. Check OpenCV window.")
        except FileNotFoundError:
            self.log_message(f"Error: File not found during processing: {image_path}")
            messagebox.showerror("Processing Error", f"File not found during processing: {image_path}")
        except Exception as e:
            self.log_message(f"An unexpected error occurred during processing: {e}")
            messagebox.showerror("Processing Error", f"An unexpected error occurred: {e}")

    def log_message(self, message):
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END) # Auto-scroll to the end

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAreaDetectorApp(root)
    root.mainloop()
