from SupMaskSu import handlerMask
import numpy as np
from PIL import ImageDraw,Image, ImageFilter
import os

# Specify the directory to read the .png files from
directory_path = '/mnt/sessd/dataset/subject-images-directory/'

if __name__ == '__main__':
    # Get all .png files in the directory
    png_files = [f for f in os.listdir(directory_path) if f.endswith('.png')]

    # Output the absolute paths of the .png files
    for png_file in png_files:
        file_path = os.path.abspath(os.path.join(directory_path, png_file))
        if png_file.endswith('_mask.png'):
            print(f'this is a mask this {png_file}')
            continue
        save_path = file_path.replace('.png', '_mask.png')
        if os.path.exists(save_path):
            print(f'remove this{save_path}')
            os.remove(save_path)
        image_org = Image.open(file_path)
        try:
            remove_and_rege_mask = handlerMask(file_path)
            merged_mask_pixels = np.argwhere(remove_and_rege_mask > 0)
            print(f'run this {file_path}')
            merged_mask_pixels = [[int(x), int(y)] for y, x in merged_mask_pixels]
            mask_clear = Image.new("L", image_org.size, 0)
            draw = ImageDraw.Draw(mask_clear)
            for point in merged_mask_pixels:
                draw.point(point, fill=255)
            mask_clear.save(save_path)
        except ImportError:
            print(f'handler error {png_file}')
            continue
        print(f'finish this {file_path} and save to {save_path}')