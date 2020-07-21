from tkinter import *
from tkinter import ttk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image, ImageDraw
from skimage.morphology import remove_small_objects, disk, dilation
from annotations_writer import VOCWriter
from datetime import datetime
from skimage import img_as_float, img_as_bool
from skimage.segmentation import chan_vese, morphological_chan_vese, morphological_geodesic_active_contour, \
    inverse_gaussian_gradient
import os
import cv2
import numpy as np
import pandas as pd

# colors for the bboxes
COLORS = ['dark slate gray', 'dim gray', 'slate gray',
          'light slate gray', 'gray', 'light grey', 'midnight blue', 'navy', 'cornflower blue', 'dark slate blue',
          'slate blue', 'medium slate blue', 'light slate blue', 'medium blue', 'royal blue', 'blue',
          'dodger blue', 'gold', 'light goldenrod', 'goldenrod', 'dark goldenrod', 'rosy brown',
          'indian red', 'saddle brown', 'sandy brown',
          'dark salmon', 'salmon', 'light salmon', 'orange', 'dark orange',
          'coral', 'light coral', 'tomato', 'orange red', 'red', 'hot pink', 'deep pink', 'pink', 'light pink',
          'pale violet red', 'maroon', 'medium violet red', 'violet red',
          'medium orchid', 'dark orchid', 'dark violet', 'blue violet', 'purple', 'medium purple', 'brown1', 'brown2',
          'brown3', 'brown4', 'salmon1', 'salmon2',
          'salmon3', 'salmon4', 'LightSalmon2', 'LightSalmon3', 'LightSalmon4', 'orange2',
          'orange3', 'orange4', 'DarkOrange1', 'DarkOrange2', 'DarkOrange3', 'DarkOrange4',
          'coral1', 'coral2', 'coral3', 'coral4', 'tomato2', 'tomato3', 'tomato4', 'OrangeRed2',
          'OrangeRed3', 'OrangeRed4', 'red2', 'red3', 'red4']


class DetectionTool:
    def __init__(self, main_root, master):

        self.super_parent = main_root
        self.super_parent.protocol("WM_DELETE_WINDOW", self.on_exit)
        # set up the main frame
        self.parent = master

        # Options and Settings for Detection Tasks - [Begin]
        # ----------------------------------------

        # constants
        self.default_img_preview = ImageTk.PhotoImage(Image.new(mode='RGB', size=(256, 256), color=(255, 255, 255)))
        self.img_w = 256
        self.img_h = 256
        self.keys = [
            'masks_cv',
            'masks_mcv',
            'masks_mgac',
            'masks_adpt',
            'masks_elipses'
        ]
        self.strings_to_keys = {
            'Chan Vese': 'masks_cv',
            'Morph CV': 'masks_mcv',
            'MGAC': 'masks_mgac',
            'Adaptive Th': 'masks_adpt',
            'Elipses': 'masks_elipses'
        }
        # Initialize Path Variables and Strings
        self.det_inp_dir = ""
        self.det_out_dir = ""
        self.det_org_imgs_dir_path = ""
        self.det_dab_imgs_dir_path = ""
        self.det_curr_org_img_preview_path = ""
        self.det_curr_dab_img_preview_path = ""
        self.imagename = ''

        # BBox Type and Size
        self.fixed_bbox_size = 16
        self.bbox_type = 0

        # References for Images, Lists, Helper Variables
        self.det_imgs_loaded = False
        self.det_imgs_list = []
        self.det_curr_img_index = 0
        self.det_loaded_imgs_total = 0
        self.det_curr_org_img_preview = None
        self.det_curr_dab_img_preview = None
        self.det_imgs_done = 0
        self.det_imgs_done_fnames = []
        self.elipse_radius = 6
        self.curr_resultant_masks = None
        self.labels_df = None
        self.curr_lymphocyte_count = 0
        self.annotations_type = 0
        self.collect_patches = True
        self.masks_generated_for_current = False
        self.progress_saved = True
        self.already_done = None

        # initialize mouse state
        self.STATE = dict()
        self.STATE['click'] = 0
        self.STATE['x'], self.STATE['y'] = 0, 0

        # reference to bbox
        self.bboxIdList = []
        self.bboxId = None
        self.bboxList = []
        self.hl = None
        self.vl = None

        # All results
        self.result = dict()

        # Options and Settings for Detection Tasks - [End]
        # ----------------------------------------

        # Set up options for Detection Notebook Tab
        self.det_tab_main_frame = Frame(self.parent, bg="#F5F5F5")
        self.det_tab_main_frame.pack(fill="both", expand=1)

        # Top Panel - [Start]
        # directory selection frame
        #       Includes Load I/O directories, load images etc.

        # Widgets
        self.det_header_container = LabelFrame(self.det_tab_main_frame, text="Select I/O Ops", bg="#F5F5F5", pady=5)
        self.det_choose_in_dir_btn = Button(self.det_header_container, text="Select Input Directory",
                                            relief="groove", command=self.select_det_input_dir, width=25,
                                            bg="#FFFFF0", cursor="hand2")
        self.in_dir_path_var = StringVar()
        self.det_choose_in_dir_olabel = Label(self.det_header_container, textvariable=self.in_dir_path_var,
                                              bg="#DCDCDC", anchor="nw", padx=4, width=115)
        self.in_dir_path_var.set("PATH = ")
        self.det_load_imgs_btn = Button(self.det_header_container, text="Load Images", relief="groove",
                                        command=self.det_load_imgs, width=25, bg="#FFFFF0", cursor="hand2")
        self.det_choose_out_dir_btn = Button(self.det_header_container, text="Select Output Directory",
                                             relief="groove", command=self.select_det_output_dir, width=25,
                                             bg="#FFFFF0", cursor="hand2")
        self.out_dir_path_var = StringVar()
        self.det_choose_out_dir_olabel = Label(self.det_header_container, textvariable=self.out_dir_path_var,
                                               bg="#DCDCDC", anchor="nw", padx=4, width=115)
        self.det_choose_out_dir_olabel.grid(row=1, column=1, columnspan=4, padx=10, pady=5)
        self.out_dir_path_var.set("PATH = ")
        self.det_save_prog_btn = Button(self.det_header_container, text="Save Progress", relief="groove",
                                        command=self.det_save_progress, width=25, bg="#FFFFF0", cursor="hand2")

        # Layouts
        self.det_header_container.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="nwe")
        self.det_choose_in_dir_btn.grid(row=0, column=0, padx=10, pady=5)
        self.det_choose_in_dir_olabel.grid(row=0, column=1, columnspan=4, padx=10, pady=5)
        self.det_load_imgs_btn.grid(row=0, column=5, padx=10, pady=5)
        self.det_choose_out_dir_btn.grid(row=1, column=0, padx=10, pady=5)
        self.det_save_prog_btn.grid(row=1, column=5, padx=10, pady=5)
        # Top Panel - [End]

        # Main Panel - [Start]
        #   Includes:
        #       Images Preview, Navigation Buttons, Count Hint, xy coords, radio buttons, slider and main buttons

        # Images Preview Frame - Center Left Panel
        # Widgets
        self.det_main_container = LabelFrame(self.det_tab_main_frame, text="Images Preview", bg="#F5F5F5", pady=5)
        self.det_org_img_lbl = Label(self.det_main_container, text="Original Image (Reference Only)", width=38,
                                     bg="#DCDCDC",
                                     anchor="nw", padx=4)
        self.det_dab_img_lbl = Label(self.det_main_container, text="DAB Image (Draw BBox Here)", width=38,
                                     bg="#DCDCDC", anchor="nw",
                                     padx=4)
        self.det_out_mask_img_lbl = Label(self.det_main_container, text="Output Mask", width=38,
                                          bg="#DCDCDC", anchor="nw",
                                          padx=4)
        self.det_org_img_preview = Label(self.det_main_container, image=self.default_img_preview, anchor="nw",
                                         padx=4, relief="ridge")
        # Canvas
        self.det_canvas_dab_img = Canvas(self.det_main_container, width=self.img_w, height=self.img_h,
                                         cursor="tcross")
        self.det_canvas_dab_img.create_image(0, 0, image=self.default_img_preview, anchor="nw")
        self.det_canvas_dab_img.bind("<Motion>", self.mouse_move)
        self.det_canvas_dab_img.bind("<Button-1>", self.mouse_click)
        self.det_canvas_dab_img.bind("<Leave>", self.mouse_leave)
        self.super_parent.bind("<Escape>", self.cancel_bbox)

        self.det_out_mask_img_preview = Label(self.det_main_container, image=self.default_img_preview,
                                              anchor="nw",
                                              padx=4, relief="ridge")

        self.det_prev_img_btn = Button(self.det_main_container, text="< Back", relief="groove",
                                       command=self.det_prev_img, width=20, bg="#FFFFF0", cursor="hand2")

        self.det_lymph_hint_frame = Frame(self.det_main_container, bg="#F5F5F5")

        self.det_lymph_count_lbl = Label(self.det_lymph_hint_frame, text="(Ref) Count = ",
                                         bg="brown", fg="white", width=19)

        self.det_lymph_coords_lbl = Label(self.det_lymph_hint_frame, text="x: , y: ",
                                          bg="brown1", fg="white", width=19)

        self.det_next_img_btn = Button(self.det_main_container, text="Next >", relief="groove",
                                       command=self.det_next_img, width=20, bg="#FFFFF0", cursor="hand2")

        self.sep = ttk.Separator(self.det_main_container, orient="horizontal")

        # Layouts
        self.det_tab_main_frame.grid_rowconfigure(1, weight=1)
        self.det_main_container.grid(row=1, column=0, padx=10, pady=5, sticky="nws")
        self.det_org_img_lbl.grid(row=0, column=0, padx=10, pady=5)
        self.det_dab_img_lbl.grid(row=0, column=1, padx=10, pady=5)
        self.det_out_mask_img_lbl.grid(row=0, column=2, padx=10, pady=5)
        self.det_org_img_preview.grid(row=1, column=0, padx=10, pady=5)
        self.det_canvas_dab_img.grid(row=1, column=1, padx=10, pady=5)
        self.det_out_mask_img_preview.grid(row=1, column=2, padx=10, pady=5)
        self.det_prev_img_btn.grid(row=2, column=0, padx=10, pady=3)
        self.det_lymph_hint_frame.grid(row=2, column=1, padx=10, pady=5, sticky="nwes")
        self.det_lymph_count_lbl.grid(row=0, column=0, sticky="nwes")
        self.det_lymph_coords_lbl.grid(row=0, column=1, sticky="nwes")
        self.det_next_img_btn.grid(row=2, column=2, padx=10, pady=3)
        self.sep.grid(row=3, column=0, columnspan=3, sticky="we", padx=10, pady=5)

        # Options Panel
        # Widgets
        self.det_options_container = Frame(self.det_main_container, bg="#F5F5F5")
        self.det_bbox_choice_header = Label(self.det_options_container, text="Choose BBox Type", bg="#DCDCDC",
                                            anchor="nw", padx=4)
        self.det_bbox_choice_header.grid(row=0, column=0, columnspan=2, sticky="nwe")

        self.bbox_choice = IntVar()
        self.bbox_choice.set("0")

        self.det_rb_vbbox = Radiobutton(self.det_options_container, text="Variable BBox", variable=self.bbox_choice,
                                        value=0, command=lambda: self.set_bbox_type(self.bbox_choice.get()),
                                        bg="#F5F5F5")

        self.det_rb_fbbox = Radiobutton(self.det_options_container, text="Fixed BBox", variable=self.bbox_choice,
                                        value=1, command=lambda: self.set_bbox_type(self.bbox_choice.get()),
                                        bg="#F5F5F5")

        self.det_annotations_choice_header = Label(self.det_options_container, text="Choose Annotations Type",
                                                   bg="#DCDCDC",
                                                   anchor="nw", padx=4)

        self.annotations_choice = IntVar()
        self.annotations_choice.set("0")

        self.det_rb_annotations_clear = Radiobutton(self.det_options_container, text="Clear",
                                                    variable=self.annotations_choice,
                                                    value=0, command=lambda: self.set_annotations_type(
                self.annotations_choice.get()),
                                                    bg="#F5F5F5")

        self.det_rb_annotations_ambiguous = Radiobutton(self.det_options_container, text="Ambiguous",
                                                        variable=self.annotations_choice,
                                                        value=1, command=lambda: self.set_annotations_type(
                self.annotations_choice.get()),
                                                        bg="#F5F5F5")

        self.det_options_container_B = Frame(self.det_main_container, bg="#F5F5F5")
        self.det_bbox_size_header = Label(self.det_options_container_B, text="Choose Fixed BBox Size", bg="#DCDCDC",
                                          anchor="nw", padx=4)

        self.det_bbox_size_slider = Scale(self.det_options_container_B, from_=16, to=32, resolution=2,
                                          orient="horizontal",
                                          command=self.set_fixed_bbox_size, cursor="hand2", bg="#F5F5F5")
        self.fixed_bbox_size = self.det_bbox_size_slider.get()

        self.det_collect_patches_header = Label(self.det_options_container_B, text="Collect Patches",
                                                bg="#DCDCDC",
                                                anchor="nw", padx=4)

        self.collect_patches_choice = BooleanVar()
        self.collect_patches_choice.set(True)

        self.det_rb_yes = Radiobutton(self.det_options_container_B, text="Yes",
                                      variable=self.collect_patches_choice,
                                      value=True,
                                      command=lambda: self.set_collect_patches(self.collect_patches_choice.get()),
                                      bg="#F5F5F5")

        self.det_rb_no = Radiobutton(self.det_options_container_B, text="No",
                                     variable=self.collect_patches_choice,
                                     value=False,
                                     command=lambda: self.set_collect_patches(self.collect_patches_choice.get()),
                                     bg="#F5F5F5")

        self.det_options_container_C = Frame(self.det_main_container, bg="#F5F5F5")

        self.det_ellipse_radius_choice_label = Label(self.det_options_container_C, text="Ellipse Radius:",
                                                     bg="#DCDCDC", anchor="nw", padx=4)
        self.ellipse_radius_options = [3, 4, 5, 6, 7, 8]
        self.ellipse_radius_choice = IntVar()
        self.ellipse_radius_choice.set(self.ellipse_radius_options[3])
        self.view_ellipse_radius = ttk.OptionMenu(self.det_options_container_C, self.ellipse_radius_choice,
                                                  self.ellipse_radius_options[3], *self.ellipse_radius_options,
                                                  style='mask_preview_options.TMenubutton',
                                                  command=self.set_ellipse_radius)
        self.det_mask_preview_choice_label = Label(self.det_options_container_C, text="Preview Mask:",
                                                   bg="#DCDCDC",
                                                   anchor="nw", padx=4)
        self.mask_options = [k for k in self.strings_to_keys.keys()]
        self.mask_choice = StringVar()
        self.mask_choice.set(self.mask_options[3])
        self.view_mask = ttk.OptionMenu(self.det_options_container_C, self.mask_choice, self.mask_options[3],
                                        *self.mask_options, style='mask_preview_options.TMenubutton',
                                        command=self.set_mask_preview)

        self.det_gen_mask_img_btn = Button(self.det_options_container_C, text="Generate Mask", relief="groove",
                                           command=self.det_gen_mask_img, width=15, bg="#FFFFF0", cursor="hand2")

        self.det_save_patches_btn = Button(self.det_options_container_C, text="Save Patch(es)", relief="groove",
                                           command=self.det_save_patches, width=15, bg="#FFFFF0", cursor="hand2")

        self.det_save_results_btn = Button(self.det_options_container_C, text="Save Results", relief="groove",
                                           command=self.det_save_results, width=15, bg="#FFFFF0", cursor="hand2")

        # Layouts
        self.det_options_container.grid(row=4, column=0, rowspan=4, padx=10, pady=5, sticky="nwe")
        self.det_options_container.grid_columnconfigure(0, weight=1)
        self.det_rb_vbbox.grid(row=1, column=0, padx=10, pady=5, sticky="nw")
        self.det_rb_fbbox.grid(row=1, column=1, padx=10, pady=5, sticky="nw")
        self.det_annotations_choice_header.grid(row=2, column=0, columnspan=2, sticky="nwe")
        self.det_rb_annotations_clear.grid(row=3, column=0, padx=10, pady=5, sticky="nw")
        self.det_rb_annotations_ambiguous.grid(row=3, column=1, padx=10, pady=5, sticky="nw")
        self.det_options_container_B.grid(row=4, column=1, rowspan=4, padx=10, pady=5, sticky="nwe")
        self.det_bbox_size_header.grid(row=0, column=0, columnspan=2, sticky="we")
        self.det_options_container_B.grid_columnconfigure(0, weight=1)
        self.det_bbox_size_slider.grid(row=1, column=0, columnspan=2, padx=10, pady=1, sticky="we")
        self.det_collect_patches_header.grid(row=2, column=0, columnspan=2, pady=(5, 0), sticky="nwe")
        self.det_rb_yes.grid(row=3, column=0, padx=10, pady=(5, 0), sticky="nw")
        self.det_rb_no.grid(row=3, column=1, padx=10, pady=(5, 0), sticky="nw")
        self.det_options_container_C.grid(row=4, column=2, rowspan=4, padx=10, pady=5, sticky="nwe")
        self.det_ellipse_radius_choice_label.grid(row=0, column=0, padx=(10, 0), pady=(0, 5), sticky="nwe")
        self.view_ellipse_radius.grid(row=0, column=1, padx=(0, 10), pady=(0, 5), sticky='nwe')
        self.det_mask_preview_choice_label.grid(row=1, column=0, padx=(10, 0), pady=3, sticky="nwe")
        self.view_mask.grid(row=1, column=1, padx=(0, 10), pady=3, sticky='nwe')
        self.det_gen_mask_img_btn.grid(row=2, column=0, columnspan=2, padx=10, pady=3, sticky="we")
        self.det_save_patches_btn.grid(row=3, column=0, padx=10, pady=3)
        self.det_save_results_btn.grid(row=3, column=1, padx=10, pady=3)

        # Main Panel - [End]

        # Lists Panel - [Start]
        #   Includes:
        #       List boxes, Labels, and Buttons

        # Listboxes Container Frame
        # Widgets
        self.det_lists_frame = Frame(self.det_tab_main_frame, bg="#F5F5F5", pady=5)

        self.det_loaded_imgs_header = Label(self.det_lists_frame, text="Loaded Images", bg="#DCDCDC",
                                            anchor="nw", padx=4)
        self.det_bbox_list_header = Label(self.det_lists_frame, text="BBoxes / Image", bg="#DCDCDC",
                                          anchor="nw", padx=4)

        self.det_imgs_list_frame = Frame(self.det_lists_frame, bg="#F5F5F5")

        self.det_img_list_box_scrollbar = Scrollbar(self.det_imgs_list_frame, orient="vertical")

        self.det_imgs_name_list_box = Listbox(self.det_imgs_list_frame, bg="#F5F5DC", height=11, width=10,
                                              yscrollcommand=self.det_img_list_box_scrollbar.set, exportselection=0)
        self.det_img_list_box_scrollbar.config(command=self.det_imgs_name_list_box.yview)

        self.det_imgs_name_list_box.bind("<<ListboxSelect>>", self.det_img_name_list_item_select)

        self.det_bbox_list_frame = Frame(self.det_lists_frame, bg="#F5F5F5")

        self.det_bbox_list_box_scrollbar = Scrollbar(self.det_bbox_list_frame, orient="vertical")

        self.det_bbox_list_box = Listbox(self.det_bbox_list_frame, bg="#F5F5DC", height=11, width=10,
                                         yscrollcommand=self.det_bbox_list_box_scrollbar.set, exportselection=0)
        self.det_bbox_list_box_scrollbar.config(command=self.det_bbox_list_box.yview)

        self.det_imgs_progress = Label(self.det_lists_frame, text="Image:  [ ] / [ ]  ", bg="khaki1", relief='flat',
                                       anchor="w", padx=4)

        self.btnDel = Button(self.det_lists_frame, text='Delete', command=self.del_bbox, bg="#FFFFF0", cursor="hand2",
                             relief="groove")
        self.det_imgs_completed = Label(self.det_lists_frame, text="Done: [ ] / [ ] ", bg="khaki1", relief='flat',
                                        anchor="w", padx=4)
        self.btnClear = Button(self.det_lists_frame, text='Clear All', command=self.clear_bbox_list, bg="#FFFFF0",
                               cursor="hand2", relief="groove")

        self.det_final_result_frame = Frame(self.det_lists_frame, bg="#F5F5F5")

        # Layouts
        self.det_tab_main_frame.grid_columnconfigure(1, weight=1)
        self.det_lists_frame.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky="nwes")
        self.det_loaded_imgs_header.grid(row=0, column=0, padx=4, pady=5, sticky="we")
        self.det_bbox_list_header.grid(row=0, column=1, padx=4, pady=5, sticky="we")
        self.det_imgs_list_frame.grid(row=1, column=0, sticky="we")
        self.det_img_list_box_scrollbar.pack(side="right", fill="y", padx=(0, 5))
        self.det_imgs_name_list_box.pack(fill="x", padx=(4, 0), pady=5)
        self.det_bbox_list_frame.grid(row=1, column=1, sticky="we")
        self.det_bbox_list_box_scrollbar.pack(side="right", fill="y", padx=(0, 5))
        self.det_bbox_list_box.pack(fill="x", padx=(4, 0), pady=5)
        self.det_imgs_progress.grid(row=2, column=0, padx=5, pady=2, sticky="nwes")
        self.btnDel.grid(row=2, column=1, padx=6, pady=2, sticky="wen")
        self.det_imgs_completed.grid(row=3, column=0, padx=5, pady=2, sticky="nwes")
        self.btnClear.grid(row=3, column=1, padx=6, pady=2, sticky="wen")
        self.det_final_result_frame.grid(row=4, column=0, sticky="we")
        self.det_lists_frame.grid_columnconfigure(0, weight=1)
        self.det_lists_frame.grid_columnconfigure(1, weight=1)
        self.det_lists_frame.grid_rowconfigure(4, weight=1)
        # Lists Panel - [End]

    def select_det_input_dir(self):
        folder_selected = filedialog.askdirectory(initialdir=os.getcwd(), title="Select Input Directory")
        self.in_dir_path_var.set("PATH = " + folder_selected)
        self.det_inp_dir = folder_selected
        self.det_org_imgs_dir_path = os.path.join(os.path.realpath(self.det_inp_dir), "images_org")
        self.det_dab_imgs_dir_path = os.path.join(os.path.realpath(self.det_inp_dir), "images_dab")

    def select_det_output_dir(self):
        folder_selected = filedialog.askdirectory(initialdir=os.getcwd(), title="Select Output Directory")
        self.out_dir_path_var.set("PATH = " + folder_selected)
        self.det_out_dir = folder_selected

    def det_load_imgs(self):
        if not self.det_imgs_loaded:
            labels_csv = os.path.join(self.det_inp_dir, 'labels.csv')
            self.labels_df = pd.read_csv(labels_csv)
            progress_csv_path = os.path.join(self.det_inp_dir, 'progress.csv')
            if os.path.exists(progress_csv_path):
                tmp_df = pd.read_csv(progress_csv_path)
                self.already_done = tmp_df['fname'].tolist()
                all_images = sorted(os.listdir(self.det_org_imgs_dir_path))
                not_done = list(set(all_images) - set(self.already_done))
                self.det_imgs_list = sorted(not_done)
            else:
                self.det_imgs_list = sorted(os.listdir(self.det_org_imgs_dir_path))

            for fname in self.det_imgs_list:
                self.det_imgs_name_list_box.insert("end", fname)

            self.det_imgs_loaded = True
            self.det_loaded_imgs_total = self.det_imgs_name_list_box.size()
            self.det_imgs_name_list_box.selection_set(self.det_curr_img_index)

            self.imagename = self.det_imgs_list[self.det_curr_img_index]

            self.det_imgs_progress.config(
                text="Image: %d / %d" % (self.det_curr_img_index + 1, self.det_loaded_imgs_total))

            self.set_preview_images()

            self.det_prev_img_btn.config(state="disabled")

    def det_save_progress(self):
        if not self.progress_saved:
            if self.already_done:
                progress_df = pd.DataFrame({'fname': self.det_imgs_done_fnames + self.already_done})
            else:
                progress_df = pd.DataFrame({'fname': self.det_imgs_done_fnames})
            progress_df.to_csv(os.path.join(self.det_inp_dir, 'progress.csv'), index=False)
            timestamp = str(datetime.now().timestamp()).replace('.', '_')
            results_dump_df = pd.DataFrame(self.result)
            results_dump_df.to_json(os.path.join(self.det_out_dir, 'dumps', 'results_' + timestamp + '.json'))
            self.progress_saved = True

    def det_next_img(self):

        if self.bboxList:
            messagebox.showwarning(title="Data Warning", message="Save your results first!")
        else:
            if self.det_curr_img_index + 1 >= self.det_loaded_imgs_total:
                self.det_next_img_btn.config(state="disabled")

            if self.det_imgs_loaded and not self.det_curr_img_index + 1 >= self.det_loaded_imgs_total:
                self.masks_generated_for_current = False
                self.det_imgs_name_list_box.selection_clear(self.det_curr_img_index)
                self.det_curr_img_index += 1
                self.det_imgs_name_list_box.see(self.det_curr_img_index)
                self.det_imgs_name_list_box.selection_set(self.det_curr_img_index)
                self.imagename = self.det_imgs_list[self.det_curr_img_index]
                self.det_imgs_progress.config(
                    text="Image: %d / %d" % (self.det_curr_img_index + 1, self.det_loaded_imgs_total))
                self.set_preview_images()

                if self.det_prev_img_btn['state'] == 'disabled':
                    self.det_prev_img_btn.config(state="normal")

    def det_prev_img(self):

        if self.bboxList:
            messagebox.showwarning(title="Data Warning", message="Save your results first!")
        else:
            if self.det_curr_img_index - 1 < 0:
                self.det_prev_img_btn.config(state="disabled")

            if self.det_imgs_loaded and not self.det_curr_img_index - 1 < 0:
                self.masks_generated_for_current = False
                self.det_imgs_name_list_box.selection_clear(self.det_curr_img_index)
                self.det_curr_img_index -= 1
                self.det_imgs_name_list_box.see(self.det_curr_img_index)
                self.det_imgs_name_list_box.selection_set(self.det_curr_img_index)
                self.imagename = self.det_imgs_list[self.det_curr_img_index]
                self.det_imgs_progress.config(
                    text="Image: %d / %d" % (self.det_curr_img_index + 1, self.det_loaded_imgs_total))
                self.set_preview_images()

                if self.det_next_img_btn['state'] == 'disabled':
                    self.det_next_img_btn.config(state="normal")

    def set_bbox_type(self, v):
        self.bbox_type = v

    def set_fixed_bbox_size(self, v):
        self.fixed_bbox_size = v

    def set_annotations_type(self, v):
        self.annotations_type = v

    def set_collect_patches(self, v):
        self.collect_patches = v
        if v is True:
            self.det_save_patches_btn.config(state="normal")
        elif v is False:
            self.det_save_patches_btn.config(state="disabled")

    def set_mask_preview(self, v):
        if self.masks_generated_for_current:
            preview_mask = ImageTk.PhotoImage(self.curr_resultant_masks[self.strings_to_keys[v]])
            self.det_out_mask_img_preview.config(image=preview_mask)
            self.det_out_mask_img_preview.image = preview_mask

    def set_ellipse_radius(self, v):
        self.elipse_radius = v

    def det_save_patches(self):
        if not self.bboxList:
            messagebox.showwarning(title="Data Warning", message="No, Bounding Boxes Found!")
        else:
            im = Image.open(self.det_curr_dab_img_preview_path)
            odir = os.path.join(self.det_out_dir, 'patches')
            for i, bbox in enumerate(self.bboxList):
                ofile = os.path.join(odir, self.imagename.split('.')[0] + '_' + str(i) + '.png')
                crp_im = im.crop(bbox)
                crp_im.save(ofile)

    def det_gen_mask_img(self):
        if self.det_curr_dab_img_preview and self.bboxList:
            # 1. Load DAB image in greyscale format and 5 New Black Image
            im = cv2.imread(self.det_curr_dab_img_preview_path, 0)
            im = cv2.resize(im, (256, 256))
            resultant_masks = {}
            for k in self.keys:
                resultant_masks[k] = Image.new(mode='L', size=(256, 256), color=0)

            # 2. Traverse over bbox list to get all bounding boxes
            for bbox in self.bboxList:
                # 3. For each bbox we get x1, y1, x2, y2 and then get region from DAB image as well as collect patches
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                patch = im[y1:y2, x1:x2]
                # 4. We process the extracted patch
                resultant_patches = self.process_patch(patch)
                # 5. We past all masks into their respective full mask images
                ex1 = cx - self.elipse_radius
                ex2 = cx + self.elipse_radius
                ey1 = cy - self.elipse_radius
                ey2 = cy + self.elipse_radius
                coords = [ex1, ey1, ex2, ey2]

                for k in self.keys:
                    if k == 'masks_elipses':
                        draw = ImageDraw.Draw(resultant_masks[k])
                        draw.ellipse(coords, fill=255)
                    else:
                        resultant_masks[k].paste(im=resultant_patches[k], box=bbox)

            preview_mask = ImageTk.PhotoImage(resultant_masks[self.strings_to_keys[self.mask_choice.get()]])
            self.det_out_mask_img_preview.config(image=preview_mask)
            self.det_out_mask_img_preview.image = preview_mask
            self.curr_resultant_masks = resultant_masks
            self.masks_generated_for_current = True
            self.progress_saved = False

    def det_save_results(self):
        # 1. Traverse resultant masks and save them
        for k in self.keys:
            opath = os.path.join(self.det_out_dir, k, self.imagename)
            self.curr_resultant_masks[k].save(opath)

        # 2. Create VOCWriter Object and write bboxes annotations
        ipath = self.det_curr_dab_img_preview_path
        ann_path = os.path.join(self.det_out_dir, 'annotations', self.imagename.split('.')[0] + '.xml')
        writer = VOCWriter(path=ipath, width=256, height=256, depth=3, database="Lysto-19")
        for bbox in self.bboxList:
            x1, y1, x2, y2 = bbox
            writer.addObject(
                name="lymphocyte",
                xmin=x1,
                ymin=y1,
                xmax=x2,
                ymax=y2
            )
        writer.save(ann_path)

        # 2B. Add results to a global result dictionary under image name as key
        self.result[self.imagename] = {
            'object_class': 1,
            'objects_count': len(self.bboxList),
            'ambiguity': self.annotations_type,
            'bbox': self.bboxList
        }

        # TODO: 3. Write annotations to tree list view

        # 4. Update Images Done Label
        self.det_imgs_done += 1
        self.det_imgs_done_fnames.append(self.imagename)
        self.det_imgs_completed.config(text="Done: %d / %d " % (self.det_imgs_done, self.det_loaded_imgs_total))

        # 5. Change Done Image Color, Clear BBox List
        self.det_imgs_name_list_box.itemconfig(self.det_curr_img_index, fg="blue")
        self.clear_bbox_list()

        # 6. Move to Next Image
        self.det_next_img()

    def det_img_name_list_item_select(self, event=None):

        if self.bboxList:
            messagebox.showwarning(title="Data Warning", message="Save your results first!")
            self.det_imgs_name_list_box.selection_clear(self.det_imgs_name_list_box.curselection())
            self.det_imgs_name_list_box.selection_set(self.det_curr_img_index)
        else:
            self.masks_generated_for_current = False
            index = int(self.det_imgs_name_list_box.curselection()[0])
            self.det_curr_img_index = index
            self.imagename = self.det_imgs_list[self.det_curr_img_index]
            self.det_imgs_progress.config(
                text="Image: %d / %d" % (self.det_curr_img_index + 1, self.det_loaded_imgs_total))

            self.set_preview_images()

            # Check for Navigation Buttons
            if self.det_next_img_btn[
                'state'] == 'disabled' and self.det_curr_img_index + 1 < self.det_loaded_imgs_total:
                self.det_next_img_btn.config(state="normal")
            if self.det_prev_img_btn['state'] == 'disabled' and self.det_curr_img_index - 1 >= 0:
                self.det_prev_img_btn.config(state="normal")
            if self.det_curr_img_index + 1 >= self.det_loaded_imgs_total:
                self.det_next_img_btn.config(state="disabled")
            if self.det_curr_img_index - 1 < 0:
                self.det_prev_img_btn.config(state="disabled")

    def del_bbox(self):
        sel = self.det_bbox_list_box.curselection()
        if len(sel) != 1:
            return
        idx = int(sel[0])
        self.det_canvas_dab_img.delete(self.bboxIdList[idx])
        self.bboxIdList.pop(idx)
        self.bboxList.pop(idx)
        self.det_bbox_list_box.delete(idx)
        self.masks_generated_for_current = False

    def clear_bbox_list(self):
        for idx in range(len(self.bboxIdList)):
            self.det_canvas_dab_img.delete(self.bboxIdList[idx])
        self.det_bbox_list_box.delete(0, len(self.bboxList))
        self.bboxIdList = []
        self.bboxList = []
        self.det_out_mask_img_preview.config(image=self.default_img_preview)
        self.masks_generated_for_current = False

    def set_preview_images(self):
        self.det_curr_org_img_preview_path = os.path.join(self.det_org_imgs_dir_path,
                                                          self.det_imgs_name_list_box.get(self.det_curr_img_index))
        self.det_curr_dab_img_preview_path = os.path.join(self.det_dab_imgs_dir_path,
                                                          self.det_imgs_name_list_box.get(self.det_curr_img_index))

        self.det_curr_org_img_preview = ImageTk.PhotoImage(
            Image.open(self.det_curr_org_img_preview_path).resize((256, 256)))
        self.det_curr_dab_img_preview = ImageTk.PhotoImage(
            Image.open(self.det_curr_dab_img_preview_path).resize((256, 256)))

        self.det_org_img_preview.config(image=self.det_curr_org_img_preview)
        self.det_canvas_dab_img.create_image(0, 0, image=self.det_curr_dab_img_preview, anchor="nw")
        self.det_out_mask_img_preview.config(image=self.default_img_preview)

        self.curr_lymphocyte_count = self.labels_df.loc[self.labels_df['x'] == self.imagename, 'y'].iloc[0]
        self.det_lymph_count_lbl.config(text="(Ref) Count = %d" % self.curr_lymphocyte_count)

    def mouse_move(self, event):
        self.det_lymph_coords_lbl.config(text='x: %d, y: %d' % (event.x, event.y))
        if self.bbox_type == 0:
            if self.det_curr_dab_img_preview:
                if self.hl:
                    self.det_canvas_dab_img.delete(self.hl)
                self.hl = self.det_canvas_dab_img.create_line(0, event.y, self.det_curr_dab_img_preview.width(),
                                                              event.y, width=2)
                if self.vl:
                    self.det_canvas_dab_img.delete(self.vl)
                self.vl = self.det_canvas_dab_img.create_line(event.x, 0, event.x,
                                                              self.det_curr_dab_img_preview.height(), width=2)
            if 1 == self.STATE['click']:
                if self.bboxId:
                    self.det_canvas_dab_img.delete(self.bboxId)
                self.bboxId = self.det_canvas_dab_img.create_rectangle(self.STATE['x'], self.STATE['y'],
                                                                       event.x, event.y,
                                                                       width=2,
                                                                       outline=COLORS[len(self.bboxList) % len(COLORS)])

    def mouse_click(self, event):

        if self.bbox_type == 0:
            if self.STATE['click'] == 0:
                self.STATE['x'], self.STATE['y'] = event.x, event.y
            else:
                x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
                y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)
                self.bboxList.append((x1, y1, x2, y2))
                self.bboxIdList.append(self.bboxId)
                self.bboxId = None
                self.det_bbox_list_box.insert(END, '(%d, %d) -> (%d, %d)' % (x1, y1, x2, y2))
                self.det_bbox_list_box.itemconfig(len(self.bboxIdList) - 1,
                                                  fg=COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])
            self.STATE['click'] = 1 - self.STATE['click']
        elif self.bbox_type == 1:
            bbox_size = int(self.fixed_bbox_size)
            cx, cy = event.x, event.y
            x1, y1, x2, y2 = cx - bbox_size // 2, cy - bbox_size // 2, cx + bbox_size // 2, cy + bbox_size // 2

            if cx - bbox_size / 2 < 0:
                x1 = 0
            if cx + bbox_size / 2 > 256:
                x2 = 256
            if cy - bbox_size / 2 < 0:
                y1 = 0
            if cy + bbox_size / 2 > 256:
                y2 = 256

            self.bboxId = self.det_canvas_dab_img.create_rectangle(x1, y1,
                                                                   x2, y2,
                                                                   width=2,
                                                                   outline=COLORS[len(self.bboxList) % len(COLORS)])

            self.bboxList.append((x1, y1, x2, y2))
            self.bboxIdList.append(self.bboxId)
            self.bboxId = None
            self.det_bbox_list_box.insert(END, '(%d, %d) -> (%d, %d)' % (x1, y1, x2, y2))
            self.det_bbox_list_box.itemconfig(len(self.bboxIdList) - 1,
                                              fg=COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])

    def mouse_leave(self, event):
        if self.hl:
            self.det_canvas_dab_img.delete(self.hl)
        if self.vl:
            self.det_canvas_dab_img.delete(self.vl)

    def cancel_bbox(self, event):
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.det_canvas_dab_img.delete(self.bboxId)
                self.bboxId = None
                self.STATE['click'] = 0

    def process_patch(self, patch):
        cv = chan_vese(
            img_as_float(patch),
            mu=0.25,
            lambda1=1,
            lambda2=1,
            tol=1e-3,
            max_iter=200,
            dt=0.5,
            init_level_set="checkerboard"
        ).astype(np.uint8)
        cv[cv == 1] = 255
        cv = Image.fromarray(cv)

        mcv = morphological_chan_vese(
            img_as_float(patch),
            iterations=10,
            init_level_set="checkerboard",
            smoothing=2
        ).astype(np.uint8)
        mcv[mcv == 1] = 255
        mcv = Image.fromarray(mcv)

        gimage = inverse_gaussian_gradient(img_as_float(patch))
        mgac = morphological_geodesic_active_contour(
            gimage,
            20,
            'circle',
            smoothing=1,
            threshold=0.55,
            balloon=0
        ).astype(np.uint8)
        mgac[mgac == 1] = 255
        mgac = Image.fromarray(mgac)

        adap_th = cv2.adaptiveThreshold(
            patch,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            161,
            31
        ).astype(bool)
        adap_th = remove_small_objects(adap_th, 64, connectivity=2)
        adap_th = dilation(adap_th, disk(1)).astype(np.uint8)
        adap_th[adap_th == 1] = 255
        adap_th = Image.fromarray(adap_th)

        result = {
            'masks_cv': cv,
            'masks_mcv': mcv,
            'masks_mgac': mgac,
            'masks_adpt': adap_th
        }

        return result

    def on_exit(self):
        if not self.progress_saved or self.bboxList:
            ans = messagebox.askyesno("Exit", "Work not Saved! Still want to quit the application?")
            if ans:
                self.super_parent.destroy()
        else:
            self.super_parent.destroy()
