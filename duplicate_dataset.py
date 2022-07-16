import shutil
import random

for file_no in range(31, 401):
    rand_no = random.randint(1, 30)
    shutil.copy(f"train_image_{rand_no}.jpeg", f"train_image_{file_no}.jpeg")
    file_no += 1
