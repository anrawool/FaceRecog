# Code for conversion of png to jpeg
from PIL import Image


class DatasetGeneralizer():
    def image_converter(to_convert_file, new_file, convert_file_ext, new_file_ext, start_array=1, end_array=2, ):
        for i in range(start_array, end_array):
            img = Image.open(f'{to_convert_file}{i}.{convert_file_ext}')
            rgb_img = img.convert('RGB')
            rgb_img.save(f'{new_file}{i}.{new_file_ext}')
