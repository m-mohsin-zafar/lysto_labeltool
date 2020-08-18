from tkinter import *
from tkinter import ttk
from classification_tool import ClassificationTool
from detection_tool import DetectionTool


class LabelTool:

    def __init__(self, master):

        # set up the main frame
        self.parent = master
        self.parent.title("Label Tool")
        self.main_frame = Frame(self.parent, bg="#FFE4E1")
        self.main_frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width=FALSE, height=FALSE)

        # set up ttk styles
        self.style = ttk.Style(self.parent)
        self.style.configure('main_notebook.TNotebook', tabposition="n", padding=4)
        self.style.configure('main_notebook.TNotebook.Tab', padding=(30, 4, 30, 4))
        self.style.configure('mask_preview_options.TMenubutton', background="azure")

        # set up menu bar
        self.tool_menu = Menu(self.parent)
        self.parent.config(menu=self.tool_menu)
        # set up menu bar items
        self.file_menu = Menu(self.tool_menu)
        self.edit_menu = Menu(self.tool_menu)
        self.tool_menu.add_cascade(label="File", menu=self.file_menu)
        self.tool_menu.add_cascade(label="Edit", menu=self.edit_menu)
        # set up menu bar options for File Menu
        self.file_menu.add_command(label="Exit", command=self.parent.quit)
        # set up menu bar options for Edit Menu
        self.edit_menu.add_command(label="Settings", command=self.open_settings_window)

        # set up tabbed notebook (for different tabs)
        self.notebook = ttk.Notebook(self.main_frame, style="main_notebook.TNotebook")
        self.classification_frame = Frame(self.notebook)
        self.detection_frame = Frame(self.notebook)
        self.classification_frame.pack(fill="both", expand=1)
        self.detection_frame.pack(fill="both", expand=1)
        self.notebook.add(self.classification_frame, text="Classification")
        self.notebook.add(self.detection_frame, text="Detection")
        self.notebook.pack(fill="both", expand=1)

        self.clf_tool = ClassificationTool(self.classification_frame)
        self.det_tool = DetectionTool(self.parent, self.detection_frame)

    def open_settings_window(self):
        # TODO Implementation of Settings Window
        pass


if __name__ == '__main__':
    root = Tk()
    tool = LabelTool(root)
    root.geometry('1280x700')
    root.mainloop()
