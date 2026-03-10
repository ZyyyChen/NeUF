"""
图像Crop工具 - 支持鼠标交互式选择区域进行裁剪

使用方法:
1. 直接运行: python image_crop_tool.py --input path/to/image.jpg
2. 在代码中导入使用: from image_crop_tool import CropSelector
"""

import cv2
import numpy as np
import argparse
import os
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class CropRegion:
    """裁剪区域信息"""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def top_left(self) -> Tuple[int, int]:
        return (self.x, self.y)
    
    @property
    def bottom_right(self) -> Tuple[int, int]:
        return (self.x + self.width, self.y + self.height)
    
    def to_slice(self) -> Tuple[slice, slice]:
        """返回numpy切片格式 [y:y+h, x:x+w]"""
        return (slice(self.y, self.y + self.height), 
                slice(self.x, self.x + self.width))
    
    def to_list(self) -> List[int]:
        """返回列表格式 [y, x, y+h, x+w] (类似你的extractImagesWithROI.py)"""
        return [self.y, self.x, self.y + self.height, self.x + self.width]


class CropSelector:
    """交互式图像裁剪选择器"""
    
    def __init__(self, image: np.ndarray, window_name: str = "选择裁剪区域"):
        """
        初始化裁剪选择器
        
        Args:
            image: 输入图像 (numpy array)
            window_name: 显示窗口名称
        """
        self.original_image = image.copy()
        self.display_image = image.copy()
        self.window_name = window_name
        
        # 绘制状态
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.current_rect = None
        
        # 存储所有绘制的矩形
        self.rectangles: List[CropRegion] = []
        self.selected_index = -1
        
    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 开始绘制
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # 更新当前矩形
                self.end_point = (x, y)
                self._update_display()
                
        elif event == cv2.EVENT_LBUTTONUP:
            # 完成绘制
            self.drawing = False
            self.end_point = (x, y)
            
            # 保存矩形
            if self.start_point != self.end_point:
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                
                # 确保坐标正确排序
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                
                crop_region = CropRegion(
                    x=x_min,
                    y=y_min,
                    width=x_max - x_min,
                    height=y_max - y_min
                )
                
                self.rectangles.append(crop_region)
                self.selected_index = len(self.rectangles) - 1
                
            self.start_point = None
            self.end_point = None
            self._update_display()
    
    def _update_display(self):
        """更新显示图像"""
        self.display_image = self.original_image.copy()
        
        # 绘制所有已保存的矩形
        for i, rect in enumerate(self.rectangles):
            color = (0, 255, 0) if i == self.selected_index else (255, 0, 0)
            thickness = 2 if i == self.selected_index else 1
            cv2.rectangle(
                self.display_image,
                rect.top_left,
                rect.bottom_right,
                color,
                thickness
            )
            
            # 显示矩形编号
            cv2.putText(
                self.display_image,
                f"#{i+1}",
                (rect.x, rect.y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
        
        # 绘制正在绘制的矩形
        if self.drawing and self.start_point and self.end_point:
            cv2.rectangle(
                self.display_image,
                self.start_point,
                self.end_point,
                (0, 255, 255),  # 黄色
                2
            )
        
        # 显示帮助文本
        help_text = [
            "操作说明:",
            "- 鼠标拖拽: 绘制矩形",
            "- ENTER: 确认选择",
            "- ESC/Q: 取消",
            "- C: 清除所有",
            "- Z: 撤销上一个"
        ]
        
        y_offset = 30
        for text in help_text:
            cv2.putText(
                self.display_image,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            y_offset += 20
        
        cv2.imshow(self.window_name, self.display_image)
    
    def select(self) -> Optional[CropRegion]:
        """
        开始交互式选择
        
        Returns:
            选中的裁剪区域,如果取消则返回None
        """
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        self._update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13 or key == ord('\r'):  # ENTER
                cv2.destroyAllWindows()
                if self.rectangles:
                    return self.rectangles[self.selected_index]
                return None
                
            elif key == 27 or key == ord('q'):  # ESC or Q
                cv2.destroyAllWindows()
                return None
                
            elif key == ord('c'):  # Clear all
                self.rectangles.clear()
                self.selected_index = -1
                self._update_display()
                
            elif key == ord('z'):  # Undo
                if self.rectangles:
                    self.rectangles.pop()
                    self.selected_index = len(self.rectangles) - 1
                    self._update_display()
    
    def select_multiple(self) -> List[CropRegion]:
        """
        选择多个区域
        
        Returns:
            所有选中的裁剪区域列表
        """
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        self._update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13 or key == ord('\r'):  # ENTER
                cv2.destroyAllWindows()
                return self.rectangles
                
            elif key == 27 or key == ord('q'):  # ESC or Q
                cv2.destroyAllWindows()
                return []
                
            elif key == ord('c'):  # Clear all
                self.rectangles.clear()
                self.selected_index = -1
                self._update_display()
                
            elif key == ord('z'):  # Undo
                if self.rectangles:
                    self.rectangles.pop()
                    self.selected_index = len(self.rectangles) - 1
                    self._update_display()


def select_crop_region_simple(image_path: str) -> Optional[CropRegion]:
    """
    简单的ROI选择函数(使用OpenCV内置的selectROI)
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        CropRegion对象,如果取消则返回None
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    print("使用鼠标拖拽选择区域,按SPACE或ENTER确认,按C取消")
    roi = cv2.selectROI("选择裁剪区域", image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    
    x, y, w, h = roi
    if w > 0 and h > 0:
        return CropRegion(x=x, y=y, width=w, height=h)
    return None


def crop_image(image: np.ndarray, crop_region: CropRegion) -> np.ndarray:
    """
    裁剪图像
    
    Args:
        image: 输入图像
        crop_region: 裁剪区域
        
    Returns:
        裁剪后的图像
    """
    y_slice, x_slice = crop_region.to_slice()
    return image[y_slice, x_slice]


def crop_and_save(input_path: str, output_path: str, crop_region: CropRegion):
    """
    裁剪图像并保存
    
    Args:
        input_path: 输入图像路径
        output_path: 输出图像路径
        crop_region: 裁剪区域
    """
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"无法读取图像: {input_path}")
    
    cropped = crop_image(image, crop_region)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cv2.imwrite(output_path, cropped)
    print(f"裁剪后的图像已保存到: {output_path}")
    print(f"裁剪区域: {crop_region}")


def batch_crop_images(input_folder: str, output_folder: str, 
                      crop_region: Optional[CropRegion] = None,
                      extensions: List[str] = ['.jpg', '.png', '.jpeg']):
    """
    批量裁剪文件夹中的所有图像
    
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        crop_region: 裁剪区域,如果为None则从第一张图像交互式选择
        extensions: 要处理的图像扩展名列表
    """
    # 获取所有图像文件
    image_files = []
    for ext in extensions:
        image_files.extend([f for f in os.listdir(input_folder) 
                          if f.lower().endswith(ext.lower())])
    
    if not image_files:
        print(f"在 {input_folder} 中没有找到图像文件")
        return
    
    image_files.sort()
    
    # 如果没有提供裁剪区域,从第一张图像选择
    if crop_region is None:
        first_image_path = os.path.join(input_folder, image_files[0])
        image = cv2.imread(first_image_path)
        
        selector = CropSelector(image, "选择裁剪区域(将应用于所有图像)")
        crop_region = selector.select()
        
        if crop_region is None:
            print("取消操作")
            return
    
    # 批量裁剪
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in image_files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        try:
            crop_and_save(input_path, output_path, crop_region)
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")
    
    print(f"\n完成! 处理了 {len(image_files)} 张图像")
    print(f"输出文件夹: {output_folder}")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="交互式图像裁剪工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 裁剪单张图像
  python image_crop_tool.py --input image.jpg --output cropped.jpg
  
  # 批量裁剪文件夹中的所有图像
  python image_crop_tool.py --input-folder ./images --output-folder ./cropped
  
  # 使用简单模式(OpenCV内置选择器)
  python image_crop_tool.py --input image.jpg --simple
        """
    )
    
    parser.add_argument('--input', '-i', type=str,
                       help='输入图像文件路径')
    parser.add_argument('--output', '-o', type=str,
                       help='输出图像文件路径')
    parser.add_argument('--input-folder', type=str,
                       help='输入文件夹路径(批量处理)')
    parser.add_argument('--output-folder', type=str,
                       help='输出文件夹路径(批量处理)')
    parser.add_argument('--simple', action='store_true',
                       help='使用简单模式(OpenCV内置选择器)')
    
    args = parser.parse_args()
    
    # 批量处理模式
    if args.input_folder and args.output_folder:
        batch_crop_images(args.input_folder, args.output_folder)
        return
    
    # 单文件处理模式
    if not args.input:
        parser.print_help()
        return
    
    if not os.path.exists(args.input):
        print(f"错误: 文件不存在 {args.input}")
        return
    
    # 读取图像
    image = cv2.imread(args.input)
    if image is None:
        print(f"错误: 无法读取图像 {args.input}")
        return
    
    # 选择裁剪区域
    if args.simple:
        crop_region = select_crop_region_simple(args.input)
    else:
        selector = CropSelector(image)
        crop_region = selector.select()
    
    if crop_region is None:
        print("取消操作")
        return
    
    print(f"选择的裁剪区域: {crop_region}")
    print(f"区域大小: {crop_region.width} x {crop_region.height}")
    
    # 裁剪并保存
    if args.output:
        crop_and_save(args.input, args.output, crop_region)
    else:
        # 显示裁剪结果
        cropped = crop_image(image, crop_region)
        cv2.imshow("裁剪结果", cropped)
        print("按任意键关闭...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
