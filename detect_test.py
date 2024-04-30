import threading

import torch
import time

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

class HumanDetector:
    def __init__(self,
                 weights,
                 img_size:int=640,
                 confidence_threshold:float=0.75,
                 iou_threshold:float=0.45,
                 device='cpu',
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

    def detect_from_files(self, folder_path, save_json_path: str = None, return_json=False,
                          save_images_path: str = None):
        '''
        folder_path: 입력할 이미지들의 폴더 위치
        save_json_path: 저장할 json 경로
        return_json: True면 json 리턴 // False면 dict 리턴
        save_images_path: 바운딩 박스가 그려진 이미지를 저장할 폴더 경로
        '''
        total_min = float('inf')
        total_max = float('-inf')
        total_sum = 0

        numpy_min = float('inf')
        numpy_max = float('-inf')
        numpy_sum = 0

        unsqueeze_min = float('inf')
        unsqueeze_max = float('-inf')
        unsqueeze_sum = 0

        pred1_min = float('inf')
        pred1_max = float('-inf')
        pred1_sum = 0

        pred2_min = float('inf')
        pred2_max = float('-inf')
        pred2_sum = 0

        scale_coords_min = float('inf')
        scale_coords_max = float('-inf')
        scale_coords_sum = 0


        cnt = 0

        dataset = LoadImages(folder_path, img_size=self.img_size, stride=self.stride)

        for path, img, im0s, vid_cap in dataset:
            total_start_time = (time.time() * 1000)

            start_time = (time.time() * 1000)
            img = torch.from_numpy(img).to(self.device)
            end_time = (time.time() * 1000)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            numpy_min, numpy_max, numpy_sum = self.measure_execution_time(numpy_min, numpy_max, numpy_sum, start_time, end_time)

            if img.ndimension() == 3:
                start_time = (time.time() * 1000)
                img = img.unsqueeze(0)
                end_time = (time.time() * 1000)

                unsqueeze_min, unsqueeze_max, unsqueeze_sum = self.measure_execution_time(unsqueeze_min, unsqueeze_max, unsqueeze_sum,
                                                                              start_time, end_time)

            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                start_time = (time.time() * 1000)
                pred = self.model(img, augment=self.augment)[0]
                end_time = (time.time() * 1000)

                pred1_min, pred1_max, pred1_sum = self.measure_execution_time(pred1_min, pred1_max, pred1_sum,
                                                                              start_time, end_time)
            # Apply NMS
            start_time = (time.time() * 1000)
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=self.agnostic_nms)
            end_time = (time.time() * 1000)

            pred2_min, pred2_max, pred2_sum = self.measure_execution_time(pred2_min, pred2_max, pred2_sum, start_time, end_time)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                cnt += 1
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    start_time = (time.time() * 1000)
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    end_time = (time.time() * 1000)

                    scale_coords_min, scale_coords_max, scale_coords_sum = self.measure_execution_time(scale_coords_min, scale_coords_max, scale_coords_sum,
                                                                                  start_time, end_time)
            total_end_time = (time.time() * 1000)
            total_min, total_max, total_sum = self.measure_execution_time(total_min, total_max, total_sum, total_start_time, total_end_time)

        self.resultPrint("tatal", total_min, total_max, total_sum, cnt)

        self.resultPrint("numpy", numpy_min, numpy_max, numpy_sum, cnt)

        self.resultPrint("unsqueeze", unsqueeze_min, unsqueeze_max, unsqueeze_sum, cnt)

        self.resultPrint("pred1", pred1_min, pred1_max, pred1_sum, cnt)

        self.resultPrint("pred2", pred2_min, pred2_max, pred2_sum, cnt)

        self.resultPrint("scale_coords", scale_coords_min, scale_coords_max, scale_coords_sum, cnt)

        print(f"frame: {cnt}\n")
        return "0"

    def resultPrint(self, name, min, max, sum, cnt):
        print(f"\n{name} Maximum time: {max}")
        print(f"{name} Minimum time: {min}")
        print(f"{name} execution time: {sum / cnt}")

    def measure_execution_time(self, min_val, max_val, sum_val, start_time, end_time):
        execution_time = end_time - start_time
        min_val = min(min_val, execution_time)
        max_val = max(max_val, execution_time)
        sum_val += execution_time
        return min_val, max_val, sum_val

if __name__ == '__main__':

    human_dect = HumanDetector(weights='weights/yolov7_training.pt')
    human_dect.detect_from_files(folder_path='inference/videos/',save_json_path='detect_results/result.json', save_images_path='detect_results')
    print('finished')
