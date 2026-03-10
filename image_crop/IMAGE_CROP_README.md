# 图像裁剪工具使用指南

这个工具可以让你通过鼠标交互式地选择图像区域并进行裁剪。

## 功能特性

✅ 鼠标拖拽选择裁剪区域  
✅ 支持单张或批量处理  
✅ 可选择多个区域  
✅ 简单模式和高级模式  
✅ 与现有代码兼容  

## 快速开始

### 1. 命令行使用

#### 裁剪单张图像
```bash
# 交互式选择区域
python image_crop_tool.py --input image.jpg --output cropped.jpg

# 使用简单模式(OpenCV内置选择器)
python image_crop_tool.py --input image.jpg --output cropped.jpg --simple
```

#### 批量裁剪
```bash
# 批量裁剪文件夹中的所有图像
python image_crop_tool.py --input-folder ./data/simu_56/us --output-folder ./data/simu_56/us_cropped
```

### 2. 在Python代码中使用

#### 基本用法
```python
import cv2
from image_crop_tool import CropSelector, crop_image

# 读取图像
image = cv2.imread("image.jpg")

# 创建选择器
selector = CropSelector(image, "选择裁剪区域")

# 交互式选择区域
crop_region = selector.select()

if crop_region:
    # 裁剪图像
    cropped = crop_image(image, crop_region)
    
    # 保存
    cv2.imwrite("cropped.jpg", cropped)
    
    print(f"裁剪区域: {crop_region}")
    print(f"尺寸: {crop_region.width} x {crop_region.height}")
```

#### 选择多个区域
```python
from image_crop_tool import CropSelector, crop_image

selector = CropSelector(image)
regions = selector.select_multiple()  # 可以绘制多个矩形

for i, region in enumerate(regions):
    cropped = crop_image(image, region)
    cv2.imwrite(f"crop_{i}.jpg", cropped)
```

#### 程序化裁剪(不需要交互)
```python
from image_crop_tool import CropRegion, crop_image

# 直接指定裁剪区域
crop_region = CropRegion(x=100, y=100, width=200, height=200)

# 裁剪
cropped = crop_image(image, crop_region)
```

#### 批量处理
```python
from image_crop_tool import batch_crop_images

# 会从第一张图像选择区域,然后应用到所有图像
batch_crop_images(
    input_folder="./images",
    output_folder="./cropped"
)

# 或者使用预定义的裁剪区域
crop_region = CropRegion(x=50, y=50, width=300, height=300)
batch_crop_images(
    input_folder="./images",
    output_folder="./cropped",
    crop_region=crop_region
)
```

## 操作说明

### 交互式选择界面

- **鼠标拖拽**: 绘制矩形选择区域
- **ENTER**: 确认选择
- **ESC / Q**: 取消操作
- **C**: 清除所有矩形
- **Z**: 撤销最后一个矩形

### CropRegion 对象

`CropRegion` 提供多种格式输出:

```python
crop_region = CropRegion(x=100, y=100, width=200, height=200)

# 获取坐标
crop_region.top_left        # (100, 100)
crop_region.bottom_right    # (300, 300)

# 用于numpy切片
y_slice, x_slice = crop_region.to_slice()
cropped = image[y_slice, x_slice]

# 用于兼容你的extractImagesWithROI.py格式
roi_list = crop_region.to_list()  # [100, 100, 300, 300] = [top, left, bottom, right]
```

## 与现有代码集成

### 集成到数据集处理流程

```python
from image_crop_tool import CropSelector, crop_image
import cv2

# 在处理数据集时添加交互式裁剪
def process_dataset_with_crop(image_folder):
    # 从第一张图像选择裁剪区域
    first_image = cv2.imread(f"{image_folder}/img_0.jpg")
    
    selector = CropSelector(first_image, "选择ROI区域")
    crop_region = selector.select()
    
    if crop_region:
        # 获取与extractImagesWithROI.py兼容的格式
        roi_2d = crop_region.to_list()  # [top, left, bottom, right]
        
        # 批量处理所有图像
        for i in range(num_images):
            image = cv2.imread(f"{image_folder}/img_{i}.jpg")
            cropped = crop_image(image, crop_region)
            cv2.imwrite(f"{output_folder}/img_{i}.jpg", cropped)
        
        return roi_2d
    
    return None
```

### 替代现有的 select_crop_rectangle 函数

你的 `extractImagesWithROI.py` 中的函数可以替换为:

```python
# 原来的代码:
def select_crop_rectangle(video_file, timecode=0.0):
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_POS_MSEC, timecode * 1000)
    ret, frame = cap.read()
    cap.release()
    
    roi = cv2.selectROI("Select ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    
    x, y, w, h = roi
    return [y, x, y + h, x + w]

# 新的增强版本:
from image_crop_tool import CropSelector

def select_crop_rectangle(video_file, timecode=0.0):
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_POS_MSEC, timecode * 1000)
    ret, frame = cap.read()
    cap.release()
    
    # 使用增强的选择器(支持撤销、多选等)
    selector = CropSelector(frame, "Select ROI")
    crop_region = selector.select()
    
    if crop_region:
        return crop_region.to_list()  # [top, left, bottom, right]
    else:
        return [0, 0, frame.shape[0], frame.shape[1]]  # 默认全图
```

## 运行示例

查看 `crop_example.py` 了解更多使用示例:

```bash
python crop_example.py
```

示例包括:
1. 基本单张图像裁剪
2. 选择多个区域
3. 程序化裁剪
4. 批量处理文件夹
5. 集成到数据集处理

## API 参考

### CropSelector 类

```python
CropSelector(image: np.ndarray, window_name: str = "选择裁剪区域")
```

**方法:**
- `select() -> Optional[CropRegion]`: 选择单个区域
- `select_multiple() -> List[CropRegion]`: 选择多个区域

### CropRegion 类

```python
CropRegion(x: int, y: int, width: int, height: int)
```

**属性:**
- `top_left: Tuple[int, int]`: 左上角坐标
- `bottom_right: Tuple[int, int]`: 右下角坐标

**方法:**
- `to_slice() -> Tuple[slice, slice]`: 返回numpy切片格式
- `to_list() -> List[int]`: 返回 [top, left, bottom, right] 格式

### 辅助函数

```python
crop_image(image: np.ndarray, crop_region: CropRegion) -> np.ndarray
```
裁剪图像

```python
batch_crop_images(input_folder: str, output_folder: str, 
                 crop_region: Optional[CropRegion] = None)
```
批量裁剪文件夹中的图像

## 常见问题

### Q: 如何保存裁剪区域信息以便后续使用?

```python
import json

# 保存
crop_region = CropRegion(x=100, y=100, width=200, height=200)
with open('crop_info.json', 'w') as f:
    json.dump(vars(crop_region), f)

# 加载
with open('crop_info.json', 'r') as f:
    data = json.load(f)
    crop_region = CropRegion(**data)
```

### Q: 如何应用到视频帧?

```python
import cv2
from image_crop_tool import CropSelector, crop_image

cap = cv2.VideoCapture("video.mp4")
ret, first_frame = cap.read()

# 从第一帧选择区域
selector = CropSelector(first_frame)
crop_region = selector.select()

# 处理所有帧
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cropped_frame = crop_image(frame, crop_region)
    # 处理裁剪后的帧...

cap.release()
```

### Q: 与torch tensor兼容吗?

```python
import torch
from image_crop_tool import crop_image, CropRegion

# 从numpy裁剪
crop_region = CropRegion(x=100, y=100, width=200, height=200)
cropped_np = crop_image(image_np, crop_region)

# 转换为tensor
cropped_tensor = torch.from_numpy(cropped_np)

# 或者使用torchvision
from torchvision.transforms.functional import crop as torch_crop
y, x = crop_region.y, crop_region.x
h, w = crop_region.height, crop_region.width
cropped_tensor = torch_crop(image_tensor, y, x, h, w)
```

## 依赖

- OpenCV (cv2)
- NumPy
- Python 3.7+

安装依赖:
```bash
pip install opencv-python numpy
```
