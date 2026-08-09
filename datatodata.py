import os
from PIL import Image


# 原始图片文件夹所在路径
# 例如：
# dataset/
#   ├── imgs1
#   ├── imgs2
#   ...
ROOT_DIR = "./dataset"

# 输出路径
OUTPUT_DIR = "./processed"


# 支持的图片格式
IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
)


def convert_folder(input_folder, output_folder):
    """
    单个文件夹图片转换
    """

    os.makedirs(output_folder, exist_ok=True)

    # 获取所有图片
    images = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    # 排序，保证编号稳定
    images.sort()

    print(f"{input_folder}: 共发现 {len(images)} 张图片")

    count = 1

    for img_name in images:

        input_path = os.path.join(
            input_folder,
            img_name
        )

        output_name = f"{count:03d}.png"

        output_path = os.path.join(
            output_folder,
            output_name
        )

        try:
            img = Image.open(input_path)

            # PNG不支持部分模式，例如RGBA以外的模式
            # 统一转RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            img.save(
                output_path,
                "PNG"
            )

            print(
                f"{img_name} -> {output_name}"
            )

            count += 1


        except Exception as e:
            print(
                f"处理失败: {img_name}, 原因: {e}"
            )


def main():

    # imgs1 ~ imgs6
    for i in range(1, 7):

        folder_name = f"imgs{i}"

        input_folder = os.path.join(
            ROOT_DIR,
            folder_name
        )

        output_folder = os.path.join(
            OUTPUT_DIR,
            folder_name
        )


        if not os.path.exists(input_folder):
            print(
                f"跳过不存在文件夹: {input_folder}"
            )
            continue


        print("\n====================")
        print(f"开始处理 {folder_name}")
        print("====================")


        convert_folder(
            input_folder,
            output_folder
        )


    print("\n全部处理完成！")


if __name__ == "__main__":
    main()