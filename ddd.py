import threading

import torch
import time

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
from multiprocessing import Pool
import pandas as pd

class HumanDetector:
    def __init__(self,
                 weights,
                 img_size:int=640,
                 confidence_threshold:float=0.75,
                 iou_threshold:float=0.45,
                 device='0',
                  ) -> None:
        '''
        weights: 학습된 가중치 파일 위치
        img_size: 입력할 이미지 사이즈 // 기본 yolo 계열은 640으로 고정되어 있음, 사이즈 변경을 원할 시 재학습 요구
        confidence_threshold: 객체를 탐지했다고 인식할 최소 점수 // 점수가 낮을수록 더 많은 객체가 검출됨
        iou_threshold: NMS에서 박스를 통합할 iou 최소 점수 // 점수가 높을수록 한 객체에 대한 결과가 많아질 수 있음
        '''

        self.device = select_device(device)
        self.model = attempt_load(weights, self.device)

        self.stride = int(self.model.stride.max())  # model stride
        self.img_size = check_img_size(img_size, s=self.stride)  # check img_size
        self.augment = True

        self.conf_thres = confidence_threshold
        self.iou_thres = iou_threshold
        self.agnostic_nms = True
        self.df = pd.DataFrame(columns=['No', 'Start Time', 'End Time', 'Execution Time'])



    def detect_from_files(self, index, folder_path, save_json_path: str = None, return_json=False,
                          save_images_path: str = None):

        dataset = LoadImages(folder_path, img_size=self.img_size, stride=self.stride)

        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        program_start = time.time() * 1000  # Convert to milliseconds
        result_dict = {}

        for path, img, im0s, vid_cap in dataset:
            print("")
            start_time = (time.time() * 1000) - program_start
            img = torch.from_numpy(img).to(self.device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = self.model(img, augment=self.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=self.agnostic_nms)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                print("dd")
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    result_dict[path] = []
                    for *xyxy, conf, cls in det:
                        result_dict[path].append(
                            {'box': list(map(float, xyxy)), 'class': names[int(cls)], 'conf': conf.item()})

            end_time = (time.time() * 1000) - program_start

            execution_time = end_time - start_time
            self.df.loc[frame] = [frame, start_time, end_time, execution_time]

        save_path = (str(index) + '_' + "file" + '.csv')
        print(save_path)
        self.df.to_csv(save_path, index=False)
        return "0"


def run_human_detector(index, folder_path, save_json_path, save_images_path):
    human_dect = HumanDetector(weights='weights/yolov7_training.pt')
    human_dect.detect_from_files(index, folder_path, save_json_path, save_images_path)

if __name__ == '__main__':


    paths = [
        ('1', 'inference/videos/', 'detect_results/result.json', 'detect_results'),
    ]

    threads = []

    for path_info in paths:
        thread = threading.Thread(target=run_human_detector, args=path_info)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    print('finished')
