# Code for conversion of png to jpeg
from PIL import Image
import shutil
import random


class DatasetGeneralizer():
    def image_converter(self, to_convert_file, new_file, convert_file_ext, new_file_ext, start_array=1, end_array=2, convert_to='RGB'):
        for i in range(start_array, end_array):
            img = Image.open(f'{to_convert_file}{i}.{convert_file_ext}')
            rgb_img = img.convert(convert_to)
            rgb_img.save(f'{new_file}{i}.{new_file_ext}')

    def duplicate_dataset(self, num_initial_files, expected_num_of_files, initial_file_name, copy_file_name, initial_file_extension='jpg', 
        copy_file_extension='jpeg'):

        for file_no in range(num_initial_files+1, expected_num_of_files):
            try:
                rand_no = random.randint(1, num_initial_files)
                shutil.copy(f"{initial_file_name}{rand_no}.{copy_file_extension}",
                            f"{copy_file_name}{file_no}.{initial_file_extension}")
                file_no += 1
            except Exception:
                continue
gen = DatasetGeneralizer()
gen.duplicate_dataset(2000, 5001, 'Dataset/Training_Set/sarthak/sarthak_', 'Dataset/Training_Set/sarthak/sarthak_', initial_file_extension='jpg', copy_file_extension='jpg')