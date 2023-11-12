import flet as ft
import os
import json
from IPython import display

import ultralytics

from ultralytics import YOLO

from IPython.display import display, Image
from flet_ivid import VideoContainer 
import cv2
import matplotlib.pyplot as plt
import easyocr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from flet import AppBar, ElevatedButton, Page, Text, View, colors

vc = None
	
def main(page: ft.Page):

    file_path_images = []
    
    #диалог выбора файлов
    def pick_files_result(e: ft.FilePickerResultEvent):
        selected_files.value = (
            str(len(e.files)) + " files selected" if e.files else "Cancelled!"
        ),
        
        selected_files.update()
        
        
    
    
    
    #нажатие на кнопку загрузить в модель
    def load_to_model(e):
            # model_path = 'best.pt'
            # model = YOLO(model_path)
            
            # if pick_files_dialog.result is not None and pick_files_dialog.result.files is not None:
            #     for f in pick_files_dialog.result.files:
            #         results = model.predict(f.path, save=False, imgsz=1600, conf=0.25) 
            #         for r in results:
            #             # Забираем координаты обнаруженного объекта
            #             boxes = r.boxes.xyxy
            #             # Преобразуем к нампи массиву
            #             numpy_array = boxes.cpu().numpy()
            #         img = cv2.imread(f.path)
            #         # plt.imshow(img)
            #         # ax = plt.gca()
            #         min_x = min(numpy_array[0][0], numpy_array[0][2]) # left
            #         min_y = min(numpy_array[0][1], numpy_array[0][3]) # top
            #         width = max(numpy_array[0][0], numpy_array[0][2])-min_x
            #         height = max(numpy_array[0][1], numpy_array[0][3])-min_y
            #         # draw rectangle 
            #         crop_img = img[int(numpy_array[0][1]):int(numpy_array[0][3]), int(numpy_array[0][0]):int(numpy_array[0][2])]
            #         rect = patches.Rectangle((min_x, min_y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            #         crop_img = cv2.imwrite('crop_img.jpeg', crop_img)
            #         # ax.add_patch(rect)
            #         # plt.show()
                    
                    
                    
            #         image_path = 'crop_img.jpeg'
            #         img = cv2.imread(image_path)
            #         img_crop = cv2.imread(image_path)

            #         resized = enlarge_img(img, 300)
            #         gray_image = grayscale(resized)
            #         thresh, im_bw = cv2.threshold(gray_image, 100, 300, cv2.THRESH_OTSU)
            #         no_noise = noise_removal(im_bw)
            #         eroded_image = thick_font(no_noise)
            #         dilated_image = thick_font(no_noise)
            #         cv2.imwrite("dilated_image.jpg", dilated_image)
                    
            #         number = text_recognition(file_path='dilated_image.jpg')
            #         number_all = ""
            #         for n in number:
            #             number_all+=n
            #         print(number_all)
            #         check_number(number_all)
                    
           # file_path_images.append((f.path, img, rect,img_crop,number_all))
                
        details()
        
    def details_image(n):
        pass
        
    
    def details():
        page.add(
            ft.Row(
                [
                    ft.Image(
                        src=abs[0],
                        width=400,
                        height=400
                    ),
                    ft.Image(
                        src=f'img_rect.jpg',
                        width=500,
                        height=500
                    ),
                    ft.Image(
                        src=f'crop_img.jpeg',
                        width=200,
                        height=200
                    ),
                ]  
        ),
            ft.Row(
                [
                    ft.Text("Распознанный номер: "),
                    ft.Text(abs[4])
                ]
             
            # ft.Row(
            #     [
            #         ft.Text(txt)
            #     ]
            )
        )
            # images.controls.append(
            # ft.Container(
            #         image_src=n[0],
            #         alignment=ft.alignment.center,
            #         image_fit=ft.ImageFit.SCALE_DOWN,
            #         border_radius=10,
            #         ink=True,
            #         on_click=details_image(n),
            #     ),
            # )
        
        page.update()
        
    
    pick_files_dialog = ft.FilePicker(on_result=pick_files_result)
    
    selected_files = ft.Text()
    
    def get_file(e):
        global vc
        print(pick_files_dialog.result.files[0].path)
        if pick_files_dialog.result.files[0].path is not None:
            if vc is not None:
                page.controls.pop()
                page.update()
                vc = VideoContainer(pick_files_dialog.result.files[0].path, play_after_loading=True, border_radius=18, expand=True) # This is a VideoContainer
                page.add(ft.Row([vc], alignment="center", width=600, height=600))
                page.update()
            else:
                vc = VideoContainer(pick_files_dialog.result.files[0].path, play_after_loading=True, border_radius=18, expand=True) # This is a VideoContainer
                page.add(ft.Row([vc], alignment="center", width=600, height=600))
                page.update()
            vc.play()
             
    def play_again(e):
        global vc
        if vc is not None:
            vc.pause()
            vc.play()

    
    page.overlay.append(pick_files_dialog)
    
    
    page.add(
        ft.Row(
            [
            ft.ElevatedButton(
                    "Pick files",
                    icon=ft.icons.UPLOAD_FILE,
                    on_click=lambda _: pick_files_dialog.pick_files(
                        allow_multiple=False, file_type=ft.FilePickerFileType.VIDEO 
                    ),
                ),
                selected_files,
                ft.ElevatedButton(
                    "Send files to model",
                    on_click=get_file,
                ),
                ft.ElevatedButton(
                    "Play again",
                    on_click=play_again,
                ),
                
            ]
        )
    )
    
    page.title = "Number Recognition"
    page.update()

    
ft.app(target=main, assets_dir="train_dataset_dataset")