import PIL
print(PIL.__version__)  # 检查 Pillow 的版本
from PIL import Image
img = Image.new('RGB', (100, 100), color='red')  # 简单测试 Pillow 功能
img.show()
