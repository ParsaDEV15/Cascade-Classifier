from ultralytics import YOLO
import cv2 as cv
import os
import numpy as np


class ObjectExtractor:
    def __init__(self, pos_images_dir, neg_images_dir, save_dir):
        self.save_dir = save_dir
        self.pos_images_dir = pos_images_dir
        self.neg_images_dir = neg_images_dir
        self.model = YOLO('../ModelFiles/yolov8n.pt')

    def extract(self):
        for label in os.listdir(self.pos_images_dir):
            label_path = os.path.join(self.pos_images_dir, label)
            for img in os.listdir(label_path):
                image_full_path = os.path.join(label_path, img)

                try:
                    result = self.model(image_full_path)[0]
                    new_img = cv.imread(image_full_path)

                    box = result.boxes[0]

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    new_img = new_img[y1:y2, x1:x2]
                    processed_img = self.preprocess_images(new_img)
                    self.save_positive_image(processed_img, label)

                    try:
                        self.augmenter(image=processed_img, label=label)
                    except:
                        print('Something Wrong Happened!!')
                except Exception as e:
                    print(e)

        for image in os.listdir(self.neg_images_dir):
            image_full_path = os.path.join(self.neg_images_dir, image)

            try:
                result = self.model(image_full_path)[0]
                new_img = cv.imread(image_full_path)

                box = result.boxes[0]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                new_img = new_img[y1:y2, x1:x2]
                processed_img = self.preprocess_images(new_img)

                self.save_negative_image(processed_img)

                try:
                    self.augmenter(image=processed_img, is_negative=True)
                except:
                    print('Something Wrong Happened!!')
            except Exception as e:
                print(e)

    def preprocess_images(self, image):
        resized_img = cv.resize(image, (64, 64), interpolation=cv.INTER_CUBIC)
        gray_scaled_img = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
        return gray_scaled_img

    def augmenter(self, samples=10, image=None, label=None, apply_rotation=True, contrast=(0.1, 0.9), brightness=(1, 100), is_negative=False):
        for i in range(samples):
            rand_contrast = np.random.uniform(*contrast)
            rand_brightness = np.random.uniform(*brightness)
            transformed_img = cv.convertScaleAbs(image, alpha=rand_contrast, beta=rand_brightness)

            if apply_rotation:
                rotation_types = [cv.ROTATE_180, cv.ROTATE_90_CLOCKWISE, cv.ROTATE_90_COUNTERCLOCKWISE]
                rand_rotation = np.random.choice(rotation_types)
                transformed_img = cv.rotate(transformed_img, rand_rotation)

            if not is_negative:
                self.save_positive_image(transformed_img, label)
            else:
                self.save_negative_image(transformed_img)

    def save_positive_image(self, image, label):
        save_path = str(os.path.join(self.save_dir, 'Positive', label))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        img_count = len(os.listdir(save_path))
        image_name = f'{label}_{img_count}.jpg'
        file_name = os.path.join(save_path, image_name)

        text_file_dir = os.path.join(self.save_dir, 'Positive.txt')

        with open(text_file_dir, 'a') as f:
            f.write(f'Positive/{label}/{image_name} 1 0 0 64 64\n')

        cv.imwrite(file_name, image)

    def save_negative_image(self, image):
        save_path = str(os.path.join(self.save_dir, 'Negative'))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        img_count = len(os.listdir(save_path))
        image_name = f'negative_{img_count}.jpg'
        file_name = os.path.join(save_path, image_name)

        text_file_dir = os.path.join(self.save_dir, 'Negative.txt')

        with open(text_file_dir, 'a') as f:
            f.write(f'Negative/{image_name}\n')

        cv.imwrite(file_name, image)