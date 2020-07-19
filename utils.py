import os
from PIL import Image
from tqdm import tqdm


def crop_resize(image_path):
    crop_width, crop_height = 267, 267
    im = Image.open(image_path).convert('RGB')
    width, height = im.size
    left = (width - crop_width) / 2
    top = (height - crop_height) / 2
    right = (width + crop_width) / 2
    bottom = (height + crop_height) / 2
    im = im.crop((left, top, right, bottom))
    im = im.resize((256, 256))

    return im


if __name__ == "__main__":
    # im_path = r"C:\Users\mohsi\PycharmProjects\labeltool\classification\input_dir\images_dab\lysto_pilot_1.png"
    # cropped_image = crop_resize(im_path)
    # cropped_image.show()
    root = r'D:\Datasets\Lysto'
    dirs = os.listdir(os.path.join(root, 'Groups'))
    for i in dirs:
        ndir = os.path.join(root, 'resized_cropped', 'Groups_original', i)
        if not os.path.exists(ndir):
            os.mkdir(ndir)

    for di in dirs:

        src_dir = os.path.join(root, 'Groups_original', di)
        dst_dir = os.path.join(root, 'resized_cropped', 'Groups_original', di)

        fnames = os.listdir(src_dir)

        for fname in tqdm(fnames):
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)

            res_img = crop_resize(src_path)
            res_img.save(dst_path)

