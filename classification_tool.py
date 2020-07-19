from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
import os
import pandas as pd


class ClassificationTool:
    def __init__(self, master):

        self.parent = master

        # Variables and Options for Classification Tasks
        self.clf_inp_dir = ""
        self.clf_out_dir = ""
        self.clf_org_imgs_dir_path = ""
        self.clf_dab_imgs_dir_path = ""
        self.clf_curr_org_img_preview_path = ""
        self.clf_curr_dab_img_preview_path = ""
        self.clf_imgs_list = []
        self.clf_imgs_loaded = False
        self.clf_curr_img_index = 0
        self.clf_loaded_imgs_total = 0
        self.clf_curr_org_img_preview = None
        self.clf_curr_dab_img_preview = None
        self.clf_default_img_preview = ImageTk.PhotoImage(Image.new(mode='RGB', size=(256, 256), color=(255, 255, 255)))
        self.clf_img_labeled_fnames = []
        self.clf_img_labeled_tags = []
        self.clf_imgs_done = 0
        self.progress_saved = True
        self.already_done = None

        # set up options in classification tab
        self.clf_tab_main_frame = Frame(self.parent, bg="#F5F5F5")
        self.clf_tab_main_frame.pack(fill="both", expand=1)

        self.clf_tab_dir_sel_frame = LabelFrame(self.clf_tab_main_frame, text="Select I/O Directories", height=50,
                                                bg="#F5F5F5", pady=5)
        self.clf_tab_dir_sel_frame.grid(row=0, column=0, columnspan=12, padx=25, pady=10, sticky="we")

        self.clf_tab_choose_in_dir_btn = Button(self.clf_tab_dir_sel_frame, text="Select Input Directory",
                                                relief="groove", command=self.select_clf_input_dir, width=20,
                                                bg="#FFFFF0", cursor="hand2")
        self.clf_tab_choose_in_dir_btn.grid(row=0, column=0, padx=10, pady=5)

        self.in_dir_path_var = StringVar()
        self.clf_tab_choose_in_dir_olabel = Label(self.clf_tab_dir_sel_frame, textvariable=self.in_dir_path_var,
                                                  bg="#DCDCDC", anchor="nw", padx=4, width=120)
        self.clf_tab_choose_in_dir_olabel.grid(row=0, column=1, columnspan=4, padx=10, pady=5)
        self.in_dir_path_var.set("PATH = ")

        self.clf_tab_load_imgs_btn = Button(self.clf_tab_dir_sel_frame, text="Load Images", relief="groove",
                                            command=self.clf_load_imgs, width=20, bg="#FFFFF0", cursor="hand2")
        self.clf_tab_load_imgs_btn.grid(row=0, column=5, padx=10, pady=5)

        self.clf_tab_choose_out_dir_btn = Button(self.clf_tab_dir_sel_frame, text="Select Output Directory",
                                                 relief="groove", command=self.select_clf_output_dir, width=20,
                                                 bg="#FFFFF0", cursor="hand2")
        self.clf_tab_choose_out_dir_btn.grid(row=1, column=0, padx=10, pady=5)

        self.out_dir_path_var = StringVar()
        self.clf_tab_choose_out_dir_olabel = Label(self.clf_tab_dir_sel_frame, textvariable=self.out_dir_path_var,
                                                   bg="#DCDCDC", anchor="nw", padx=4, width=120)
        self.clf_tab_choose_out_dir_olabel.grid(row=1, column=1, columnspan=4, padx=10, pady=5)
        self.out_dir_path_var.set("PATH = ")

        self.clf_save_prog_btn = Button(self.clf_tab_dir_sel_frame, text="Save Progress", relief="groove",
                                        command=self.clf_save_progress, width=20, bg="#FFFFF0", cursor="hand2")
        self.clf_save_prog_btn.grid(row=1, column=5, padx=10, pady=5)

        self.clf_imgs_list_frame = LabelFrame(self.clf_tab_main_frame, text="List of Images", bg="#F5F5F5", pady=5)
        self.clf_imgs_list_frame.grid(row=1, column=0, columnspan=3, padx=(25, 10), pady=10, sticky="w")

        self.clf_img_list_box_scrollbar = Scrollbar(self.clf_imgs_list_frame, orient="vertical")

        self.clf_imgs_name_list_box = Listbox(self.clf_imgs_list_frame, bg="#F5F5DC", height=18,
                                              yscrollcommand=self.clf_img_list_box_scrollbar.set,
                                              exportselection=0)
        self.clf_img_list_box_scrollbar.config(command=self.clf_imgs_name_list_box.yview)
        self.clf_img_list_box_scrollbar.pack(side="right", fill="y", padx=(0, 5))
        self.clf_imgs_name_list_box.pack(fill="x", padx=(10, 2), pady=5)

        self.clf_imgs_name_list_box.bind("<<ListboxSelect>>", self.clf_img_name_list_item_select)
        self.clf_imgs_progress = Label(self.clf_tab_main_frame, text="Image:  [ ] / [ ]  ", bg="khaki1", relief='flat',
                                       anchor="w", padx=4)
        self.clf_imgs_progress.grid(row=2, column=0, columnspan=3, padx=(25, 10), pady=2, sticky="nwe")

        self.clf_imgs_preview_frame = LabelFrame(self.clf_tab_main_frame, text="Images Preview", bg="#F5F5F5", pady=5)
        self.clf_imgs_preview_frame.grid(row=1, column=3, padx=(25, 10), pady=10, sticky="nws")

        self.clf_org_img_lbl = Label(self.clf_imgs_preview_frame, text="Original Image", width=50, bg="#DCDCDC",
                                     anchor="nw", padx=4)
        self.clf_org_img_lbl.grid(row=0, column=0, padx=10, pady=5)

        self.clf_dab_img_lbl = Label(self.clf_imgs_preview_frame, text="DAB Image", width=50, bg="#DCDCDC", anchor="nw",
                                     padx=4)
        self.clf_dab_img_lbl.grid(row=0, column=1, padx=10, pady=5)

        self.clf_org_img_preview = Label(self.clf_imgs_preview_frame, image=self.clf_default_img_preview, anchor="nw",
                                         padx=4, relief="ridge")
        self.clf_org_img_preview.grid(row=1, column=0, padx=10, pady=5)

        self.clf_dab_img_preview = Label(self.clf_imgs_preview_frame, image=self.clf_default_img_preview, anchor="nw",
                                         padx=4, relief="ridge")
        self.clf_dab_img_preview.grid(row=1, column=1, padx=10, pady=5)

        self.clf_prev_img_btn = Button(self.clf_imgs_preview_frame, text="< Back", relief="groove",
                                       command=self.clf_prev_img, width=20, bg="#FFFFF0", cursor="hand2")
        self.clf_prev_img_btn.grid(row=2, column=0, padx=10, pady=3, sticky="s")

        self.clf_next_img_btn = Button(self.clf_imgs_preview_frame, text="Next >", relief="groove",
                                       command=self.clf_next_img, width=20, bg="#FFFFF0", cursor="hand2")
        self.clf_next_img_btn.grid(row=2, column=1, padx=10, pady=3, sticky="s")

        self.clf_output_list = LabelFrame(self.clf_tab_main_frame, text="Label Output", bg="#F5F5F5", pady=5)
        self.clf_output_list.grid(row=1, column=4, padx=(25, 10), pady=10, sticky="we")

        self.clf_output_list_box_scrollbar = Scrollbar(self.clf_output_list, orient="vertical")

        self.clf_output_labels_list_box = Listbox(self.clf_output_list, bg="#F5F5DC", height=18, width=21,
                                                  yscrollcommand=self.clf_output_list_box_scrollbar.set,
                                                  exportselection=0)
        self.clf_output_list_box_scrollbar.config(command=self.clf_output_labels_list_box.yview)
        self.clf_output_list_box_scrollbar.pack(side="right", fill="y", padx=(0, 5))
        self.clf_output_labels_list_box.pack(fill="x", padx=(10, 2), pady=5)
        self.clf_imgs_completed = Label(self.clf_tab_main_frame, text="Done: [ ] / [ ] ", bg="khaki1", relief='flat',
                                        anchor="w", padx=4)
        self.clf_imgs_completed.grid(row=2, column=4, padx=(25, 10), pady=2, sticky="nwe")

        self.clf_imgs_tag_frame = LabelFrame(self.clf_tab_main_frame, text="Tag Image & Save Results", bg="#F5F5F5",
                                             pady=5)
        self.clf_imgs_tag_frame.grid(row=2, column=3, rowspan=2, padx=(25, 10), pady=5, sticky="we")

        self.clf_imgs_tag_header = Label(self.clf_imgs_tag_frame, text="Choose Image Label", width=40, bg="#DCDCDC",
                                         anchor="n", padx=4)
        self.clf_imgs_tag_header.grid(row=0, column=0, columnspan=2, padx=10, pady=5)

        self.img_tag = IntVar()
        self.img_tag.set("0")

        self.clf_rb_a = Radiobutton(self.clf_imgs_tag_frame, text="Artifact Image", variable=self.img_tag, value=0)
        self.clf_rb_a.grid(row=1, column=0, padx=(10, 5), pady=5, sticky="n")

        self.clf_rb_b = Radiobutton(self.clf_imgs_tag_frame, text="Normal Image", variable=self.img_tag, value=1)
        self.clf_rb_b.grid(row=1, column=1, padx=5, pady=5, sticky="n")

        self.clf_save_img_tag_btn = Button(self.clf_imgs_tag_frame, text="Save Label", relief="groove",
                                           command=lambda: self.clf_save_img_tag(self.img_tag.get()), width=20,
                                           bg="#FFFFF0", cursor="hand2")
        self.clf_save_img_tag_btn.grid(row=1, column=3, padx=10, pady=3, sticky="s")

        self.clf_save_all_results_as_csv_btn = Button(self.clf_imgs_tag_frame, text="Save All Results as CSV",
                                                      relief="groove", command=self.clf_save_all_results_as_csv,
                                                      width=30, bg="#FFFFF0", cursor="hand2")
        self.clf_save_all_results_as_csv_btn.grid(row=1, column=4, padx=10, pady=3, sticky="s")

    def select_clf_input_dir(self):
        folder_selected = filedialog.askdirectory(initialdir=os.getcwd(), title="Select Input Directory")
        self.in_dir_path_var.set("PATH = " + folder_selected)
        self.clf_inp_dir = folder_selected
        self.clf_org_imgs_dir_path = os.path.join(self.clf_inp_dir, "images_org")
        self.clf_dab_imgs_dir_path = os.path.join(self.clf_inp_dir, "images_dab")

    def select_clf_output_dir(self):
        folder_selected = filedialog.askdirectory(initialdir=os.getcwd(), title="Select Output Directory")
        self.out_dir_path_var.set("PATH = " + folder_selected)
        self.clf_out_dir = folder_selected

    def clf_load_imgs(self):
        if not self.clf_imgs_loaded:
            progress_csv_path = os.path.join(self.clf_inp_dir, 'progress.csv')
            if os.path.exists(progress_csv_path):
                tmp_df = pd.read_csv(progress_csv_path)
                self.already_done = tmp_df['fname'].tolist()
                all_images = sorted(os.listdir(self.clf_org_imgs_dir_path))
                not_done = list(set(all_images) - set(self.already_done))
                self.clf_imgs_list = sorted(not_done)
            else:
                self.clf_imgs_list = sorted(os.listdir(self.clf_org_imgs_dir_path))

            for fname in self.clf_imgs_list:
                self.clf_imgs_name_list_box.insert("end", fname)

            self.clf_imgs_loaded = True
            self.clf_loaded_imgs_total = self.clf_imgs_name_list_box.size()
            self.clf_imgs_name_list_box.selection_set(self.clf_curr_img_index)

            self.set_preview_images()
            self.clf_imgs_progress.config(
                text="Image: %d / %d" % (self.clf_curr_img_index + 1, self.clf_loaded_imgs_total))

            self.clf_prev_img_btn.config(state="disabled")

    def clf_img_name_list_item_select(self, event=None):
        index = int(self.clf_imgs_name_list_box.curselection()[0])
        self.clf_curr_img_index = index
        self.set_preview_images()
        self.clf_imgs_progress.config(
            text="Image: %d / %d" % (self.clf_curr_img_index + 1, self.clf_loaded_imgs_total))
        # Check for Navigation Buttons
        if self.clf_next_img_btn[
            'state'] == 'disabled' and self.clf_curr_img_index + 1 < self.clf_loaded_imgs_total:
            self.clf_next_img_btn.config(state="normal")
        if self.clf_prev_img_btn['state'] == 'disabled' and self.clf_curr_img_index - 1 >= 0:
            self.clf_prev_img_btn.config(state="normal")
        if self.clf_curr_img_index + 1 >= self.clf_loaded_imgs_total:
            self.clf_next_img_btn.config(state="disabled")
        if self.clf_curr_img_index - 1 < 0:
            self.clf_prev_img_btn.config(state="disabled")

    def clf_next_img(self):

        if self.clf_curr_img_index + 1 >= self.clf_loaded_imgs_total:
            self.clf_next_img_btn.config(state="disabled")

        if self.clf_imgs_loaded and not self.clf_curr_img_index + 1 >= self.clf_loaded_imgs_total:

            self.clf_imgs_name_list_box.selection_clear(self.clf_curr_img_index)
            self.clf_curr_img_index += 1
            self.clf_imgs_name_list_box.see(self.clf_curr_img_index)
            self.clf_imgs_name_list_box.selection_set(self.clf_curr_img_index)

            self.set_preview_images()
            self.clf_imgs_progress.config(
                text="Image: %d / %d" % (self.clf_curr_img_index + 1, self.clf_loaded_imgs_total))
            if self.clf_prev_img_btn['state'] == 'disabled':
                self.clf_prev_img_btn.config(state="normal")

    def clf_prev_img(self):

        if self.clf_curr_img_index - 1 < 0:
            self.clf_prev_img_btn.config(state="disabled")

        if self.clf_imgs_loaded and not self.clf_curr_img_index - 1 < 0:

            self.clf_imgs_name_list_box.selection_clear(self.clf_curr_img_index)
            self.clf_curr_img_index -= 1
            self.clf_imgs_name_list_box.see(self.clf_curr_img_index)
            self.clf_imgs_name_list_box.selection_set(self.clf_curr_img_index)

            self.set_preview_images()
            self.clf_imgs_progress.config(
                text="Image: %d / %d" % (self.clf_curr_img_index + 1, self.clf_loaded_imgs_total))
            if self.clf_next_img_btn['state'] == 'disabled':
                self.clf_next_img_btn.config(state="normal")

    def clf_save_img_tag(self, selected_label):

        fname = self.clf_imgs_name_list_box.get(self.clf_curr_img_index)
        if fname not in self.clf_img_labeled_fnames:
            self.clf_img_labeled_fnames.append(fname)
            fname_index = self.clf_img_labeled_fnames.index(fname)
            self.clf_img_labeled_tags.append(selected_label)
            result = str(fname + " , " + str(selected_label))
            self.clf_output_labels_list_box.insert("end", result)

            self.clf_output_labels_list_box.see(fname_index)
            self.clf_imgs_name_list_box.itemconfig(self.clf_curr_img_index, fg="blue")
            self.clf_imgs_done += 1
            self.clf_imgs_completed.config(text="Done: %d / %d " % (self.clf_imgs_done, self.clf_loaded_imgs_total))
            self.clf_next_img()
            self.progress_saved = False
        else:
            fname_index = self.clf_img_labeled_fnames.index(fname)
            if self.clf_img_labeled_tags[fname_index] != selected_label:
                self.clf_img_labeled_tags[fname_index] = selected_label
                self.clf_output_labels_list_box.delete(fname_index)
                updated_item = str(fname + " , " + str(selected_label))
                self.clf_output_labels_list_box.insert(fname_index, updated_item)
                self.clf_next_img()
                self.progress_saved = False

    def clf_save_all_results_as_csv(self):
        results = {
            'fname': self.clf_img_labeled_fnames,
            'label': self.clf_img_labeled_tags
        }

        df = pd.DataFrame(results, columns=['fname', 'label'])
        save_results_fp = filedialog.asksaveasfilename(initialdir=self.clf_out_dir, defaultextension='.csv',
                                                       title='Save Classification Results')
        df.to_csv(save_results_fp, index=False, header=True)

        # 1. Save Progress CSV as well
        self.clf_save_progress()

        # 2. Clear out variables and list boxes
        # 2A. Get indices for all done files for clearing img_list and list box
        for f in self.clf_img_labeled_fnames:
            listbox_vals = list(self.clf_imgs_name_list_box.get(0, "end"))
            i = listbox_vals.index(f)
            self.clf_imgs_name_list_box.delete(i)
            self.clf_imgs_list.remove(f)

        self.clf_curr_img_index = 0
        self.clf_loaded_imgs_total = self.clf_imgs_name_list_box.size()
        self.clf_imgs_name_list_box.selection_clear(self.clf_imgs_name_list_box.curselection())
        self.clf_imgs_name_list_box.selection_set(self.clf_curr_img_index)

        self.clf_img_labeled_fnames = []
        self.clf_img_labeled_tags = []
        self.clf_imgs_done = 0

        self.clf_imgs_progress.config(
            text="Image: %d / %d" % (self.clf_curr_img_index + 1, self.clf_loaded_imgs_total))
        self.clf_imgs_completed.config(
            text="Image: [ ] / [ ]"
        )

        self.clf_output_labels_list_box.delete(0, "end")

    def set_preview_images(self):
        self.clf_curr_org_img_preview_path = os.path.join(self.clf_org_imgs_dir_path,
                                                          self.clf_imgs_name_list_box.get(self.clf_curr_img_index))
        self.clf_curr_dab_img_preview_path = os.path.join(self.clf_dab_imgs_dir_path,
                                                          self.clf_imgs_name_list_box.get(self.clf_curr_img_index))

        self.clf_curr_org_img_preview = ImageTk.PhotoImage(
            Image.open(self.clf_curr_org_img_preview_path).resize((256, 256)))
        self.clf_curr_dab_img_preview = ImageTk.PhotoImage(
            Image.open(self.clf_curr_dab_img_preview_path).resize((256, 256)))

        self.clf_org_img_preview.config(image=self.clf_curr_org_img_preview)
        self.clf_dab_img_preview.config(image=self.clf_curr_dab_img_preview)

    def clf_save_progress(self):
        if not self.progress_saved:
            if self.already_done:
                progress_df = pd.DataFrame({'fname': self.clf_img_labeled_fnames + self.already_done})
            else:
                progress_df = pd.DataFrame({'fname': self.clf_img_labeled_fnames})
            progress_df.to_csv(os.path.join(self.clf_inp_dir, 'progress.csv'), index=False)
            self.progress_saved = True
