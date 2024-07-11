import os
import shutil
import random
from PIL import Image

def preprocess_data(data_dir, processed_dir, target_count=12):
    # 创建处理后的数据目录
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    # 遍历原始数据目录
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            processed_class_dir = os.path.join(processed_dir, class_name)
            os.makedirs(processed_class_dir, exist_ok=True)

            for person_name in os.listdir(class_dir):
                person_dir = os.path.join(class_dir, person_name)
                if os.path.isdir(person_dir):
                    processed_person_dir = os.path.join(processed_class_dir, person_name)
                    os.makedirs(processed_person_dir, exist_ok=True)

                    # 获取当前人物文件夹下的所有照片
                    photos = [f for f in os.listdir(person_dir) if f.endswith('.jpg') or f.endswith('.png')]

                    if len(photos) == 0:
                        print(f"警告: {person_dir} 文件夹为空,跳过处理")
                        continue

                    if len(photos) < target_count:
                        # 照片数量不足,随机重复采样补足到目标数量
                        selected_photos = photos + random.choices(photos, k=target_count-len(photos))
                    else:
                        # 照片数量超过目标数量,随机选择目标数量的照片
                        selected_photos = random.sample(photos, target_count)

                    # 复制选定的照片到处理后的文件夹,并重命名
                    for i, photo in enumerate(selected_photos, start=1):
                        src_path = os.path.join(person_dir, photo)
                        _, ext = os.path.splitext(photo)
                        dst_filename = f"{i}{ext}"
                        dst_path = os.path.join(processed_person_dir, dst_filename)
                        shutil.copy(src_path, dst_path)

    print("数据预处理完成")

# 指定原始数据目录和处理后的数据目录
data_dir = 'data'
processed_dir = 'data_processed'

# 调用预处理函数
preprocess_data(data_dir, processed_dir)