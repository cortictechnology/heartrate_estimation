import numpy as np
import math
import time
import cv2
import os
import sys
import collections
from datetime import datetime
import random
from utils import *


class HeartrateEstimator(object):
    def __init__(self, pixel_threshold=1.0):

        self.fps = 0
        self.buffer_size = 250
        self.essential_frame_accumulated = 30
        self.data_buffer_a = collections.deque(maxlen=self.buffer_size)
        self.data_buffer_b = collections.deque(maxlen=self.buffer_size)
        self.current_buffer = "a"
        self.times_a = collections.deque(maxlen=self.buffer_size)
        self.times_b = collections.deque(maxlen=self.buffer_size)

        self.freqs = []
        self.fft = []
        self.t0 = time.time()
        self.bpms = circularlist(self.essential_frame_accumulated)
        self.bpm = 0
        self.pixel_threshold = pixel_threshold

        self.face_rect = None

        self.find_faces = True

        self.icon_size = 30
        self.text_icon_size = 60

        self.pre_alpha = 0
        self.pre_reverse = False

        self.heart_img = cv2.imread("./imgs/heart.png", cv2.IMREAD_UNCHANGED)

        self.heart_img = cv2.resize(self.heart_img, (self.icon_size, self.icon_size))

        self.bpm_img = cv2.imread("./imgs/bpm.png", cv2.IMREAD_UNCHANGED)
        self.bpm_img = cv2.resize(
            self.bpm_img, (self.text_icon_size, self.text_icon_size)
        )

        self.ft = cv2.freetype.createFreeType2()
        self.ft.loadFontData(fontFileName="./fonts/HelveticaNeue.ttf", id=0)

    # Draw disconnected rectangle for a bounding box region
    def draw_disconnected_rect(self, img, pt1, pt2, color, thickness):
        width = pt2[0] - pt1[0]
        height = pt2[1] - pt1[1]
        line_width = min(20, width // 4)
        line_height = min(20, height // 4)
        line_length = max(line_width, line_height)
        cv2.line(img, pt1, (pt1[0] + line_length, pt1[1]), color, thickness)
        cv2.line(img, pt1, (pt1[0], pt1[1] + line_length), color, thickness)
        cv2.line(
            img, (pt2[0] - line_length, pt1[1]), (pt2[0], pt1[1]), color, thickness
        )
        cv2.line(
            img, (pt2[0], pt1[1]), (pt2[0], pt1[1] + line_length), color, thickness
        )
        cv2.line(
            img, (pt1[0], pt2[1]), (pt1[0] + line_length, pt2[1]), color, thickness
        )
        cv2.line(
            img, (pt1[0], pt2[1] - line_length), (pt1[0], pt2[1]), color, thickness
        )
        cv2.line(img, pt2, (pt2[0] - line_length, pt2[1]), color, thickness)
        cv2.line(img, (pt2[0], pt2[1] - line_length), pt2, color, thickness)

    def find_faces_toggle(self):
        self.find_faces = not self.find_faces
        return self.find_faces

    # Otain the scaled coordinates of a region with normalized coordinates
    def get_subface_coord(self, fh_x, fh_y, fh_w, fh_h):
        x, y, w, h = self.face_rect
        return [
            int(x + w * fh_x - (w * fh_w / 2.0)),
            int(y + h * fh_y - (h * fh_h / 2.0)),
            int(w * fh_w),
            int(h * fh_h),
        ]

    # Get the mean values for B, G, R channels of an image
    def get_subface_means(self, frame, coord):
        x, y, w, h = coord
        subframe = frame[y : y + h, x : x + w, :]
        v1 = np.mean(subframe[:, :, 0])
        v2 = np.mean(subframe[:, :, 1])
        v3 = np.mean(subframe[:, :, 2])
        return v1, v2, v3

    def select_storing_buffer(self, current_value, threshold):
        selected_buffer = None
        selected_times = None
        if self.current_buffer == "a":
            selected_buffer = self.data_buffer_a
            selected_times = self.times_a
        else:
            selected_buffer = self.data_buffer_b
            selected_times = self.times_b

        if len(selected_buffer) > 0:
            diff = abs(current_value - selected_buffer[-1])
            if diff > threshold:
                print(
                    "Change exceeds pixel threshold, may need to switch to new data buffer"
                )
                if self.current_buffer == "a":
                    if len(self.data_buffer_a) < self.buffer_size:
                        # No need to switch buffer, just need to clear existing data of current buffer
                        self.data_buffer_a = collections.deque(maxlen=self.buffer_size)
                        self.times_a = collections.deque(maxlen=self.buffer_size)
                        selected_buffer = self.data_buffer_a
                        selected_times = self.times_a
                    else:
                        # Switch to another buffer to preserve the current steady signal
                        self.current_buffer = "b"
                        self.data_buffer_b = collections.deque(maxlen=self.buffer_size)
                        self.times_b = collections.deque(maxlen=self.buffer_size)
                        selected_buffer = self.data_buffer_b
                        selected_times = self.times_b

                else:
                    if len(self.data_buffer_b) < self.buffer_size:
                        # No need to switch buffer, just need to clear existing data of current buffer
                        self.data_buffer_b = collections.deque(maxlen=self.buffer_size)
                        self.times_b = collections.deque(maxlen=self.buffer_size)
                        selected_buffer = self.data_buffer_b
                        selected_times = self.times_b
                    else:
                        # Switch to another buffer to preserve the current steady signal
                        self.current_buffer = "a"
                        self.data_buffer_a = collections.deque(maxlen=self.buffer_size)
                        self.times_a = collections.deque(maxlen=self.buffer_size)
                        selected_buffer = self.data_buffer_a
                        selected_times = self.times_a
        return selected_buffer, selected_times

    def update_storing_buffer(self, buffer, times):
        if self.current_buffer == "a":
            self.data_buffer_a = buffer
            self.times_a = times
        else:
            self.data_buffer_b = buffer
            self.times_b = times

    def calculate_bpm(self, buffer, times):
        processed = np.array(buffer)
        bpm = 0
        L = len(buffer)
        if L > self.essential_frame_accumulated:
            fps = float(L) / (times[-1] - times[0])
            even_times = np.linspace(times[0], times[-1], L)
            interpolated = np.interp(even_times, times, processed)
            interpolated = np.hamming(L) * interpolated
            interpolated = interpolated - np.mean(interpolated)
            raw = np.fft.rfft(interpolated)
            phase = np.angle(raw)
            fft = np.abs(raw)
            freqs = float(fps) / L * np.arange(L / 2 + 1)
            freqs = 60.0 * freqs
            idx = np.where((freqs > 50) & (freqs < 240))
            pruned = fft[idx]
            phase = phase[idx]

            pfreq = freqs[idx]
            freqs = pfreq
            fft = pruned
            idx2 = np.argmax(pruned)

            gap = (self.buffer_size - L) / fps
            if gap:
                bpm = 0
            else:
                bpm = freqs[idx2]
        return bpm

    def draw_object_imgs(self, image, object_img, x1, y1, x2, y2, alpha):
        if x1 >= 0 and y1 >= 0 and x2 < image.shape[1] and y2 < image.shape[0]:
            object_alpha = object_img[:, :, 3] / 255.0
            combined_alpha = object_alpha * alpha
            y2 = y2 + (object_img.shape[0] - (y2 - y1))
            image[y1:y2, x1:x2, 0] = (1.0 - combined_alpha) * image[
                y1:y2, x1:x2, 0
            ] + combined_alpha * object_img[:, :, 0]
            image[y1:y2, x1:x2, 1] = (1.0 - combined_alpha) * image[
                y1:y2, x1:x2, 1
            ] + combined_alpha * object_img[:, :, 1]
            image[y1:y2, x1:x2, 2] = (1.0 - combined_alpha) * image[
                y1:y2, x1:x2, 2
            ] + combined_alpha * object_img[:, :, 2]

    def draw_fade_heart(self, frame, x, y, w, h):
        self.draw_object_imgs(
            frame,
            self.heart_img,
            x + w // 2 - self.icon_size // 2,
            y + int(0.2 * h) - self.icon_size // 2,
            x + w // 2 - self.icon_size // 2 + self.icon_size,
            y + int(0.2 * h) - self.icon_size // 2,
            self.pre_alpha,
        )
        if not self.pre_reverse:
            self.pre_alpha = self.pre_alpha + 0.05
            if self.pre_alpha >= 1:
                self.pre_reverse = True
        else:
            self.pre_alpha = self.pre_alpha - 0.05
            if self.pre_alpha <= 0:
                self.pre_reverse = False

    def draw_pump_heart(self, frame, x, y, alpha):
        scale_factor = (((alpha - 0) * 0.5) / 1) + 0.5
        heart_img = cv2.resize(
            self.heart_img,
            (0, 0),
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_NEAREST,
        )
        x1 = x - self.icon_size - 70 + (self.icon_size // 2 - heart_img.shape[1] // 2)
        y1 = y + 10 + (self.icon_size // 2 - heart_img.shape[0] // 2)
        x2 = x1 + heart_img.shape[1]
        y2 = y1 + heart_img.shape[0]
        self.draw_object_imgs(
            frame,
            heart_img,
            x1,
            y1,
            x2,
            y2,
            1,
        )

    def draw_bpm_text(self, frame, bpm, x, y):
        text = str(int(bpm))
        textSize = self.ft.getTextSize(text, fontHeight=17, thickness=-1)[0]
        self.ft.putText(
            img=frame,
            text=text,
            org=(x - 65 + 1, y + 10 + textSize[1] + textSize[1] // 2 + 1),
            fontHeight=17,
            color=(0, 0, 0),
            thickness=-1,
            line_type=cv2.LINE_AA,
            bottomLeftOrigin=True,
        )
        self.ft.putText(
            img=frame,
            text=text,
            org=(x - 65, y + 10 + textSize[1] + textSize[1] // 2),
            fontHeight=17,
            color=(255, 255, 255),
            thickness=-1,
            line_type=cv2.LINE_AA,
            bottomLeftOrigin=True,
        )
        self.draw_object_imgs(
            frame,
            self.bpm_img,
            x - 65 + textSize[0] - 10,
            y - 5,
            x - 65 + textSize[0] - 10 + self.text_icon_size,
            y - 5 + self.text_icon_size,
            1,
        )

    def measure_pulse(self, frame, bbox, landmarks=None):
        forehead_x = 0.5
        y_start = 0.2
        width = 0.3
        height = 0.15
        col = (100, 255, 100)

        if self.find_faces:
            # Clearing out the data buffers if not estimating an heart rate
            self.data_buffer_a = collections.deque(maxlen=self.buffer_size)
            self.data_buffer_b = collections.deque(maxlen=self.buffer_size)
            self.times_a = collections.deque(maxlen=self.buffer_size)
            self.times_b = collections.deque(maxlen=self.buffer_size)

            if bbox != []:
                self.face_rect = [
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                ]
                if landmarks is not None:
                    left_eye_x = landmarks[2]
                    right_eye_x = landmarks[0]
                    forehead_x = right_eye_x + (left_eye_x - right_eye_x) / 2
            # BPM is set to 0 if not estimating
            return 0
        if self.face_rect is None:
            # BPM is set to 0 if not face is present
            return 0
        if bbox != []:
            self.face_rect = [
                bbox[0],
                bbox[1],
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
            ]
            if landmarks is not None:
                left_eye_x = landmarks[2]
                right_eye_x = landmarks[0]
                forehead_x = right_eye_x + (left_eye_x - right_eye_x) / 2

        # Extract the skin region around the forehead for estimation
        forehead1 = self.get_subface_coord(forehead_x, y_start, width, height)

        # Using the green channel for estimation
        _, v2, v3 = self.get_subface_means(frame, forehead1)

        # Select which data buffer to store the new measurement
        selected_buffer, selected_times = self.select_storing_buffer(
            v2, self.pixel_threshold
        )

        selected_buffer.append(v2)
        selected_times.append(time.time() - self.t0)

        # Update the data buffer with latest measurement
        self.update_storing_buffer(selected_buffer, selected_times)

        other_bpm = 0
        other_data_length = 0

        # If there is already a steady signal stored in another data buffer, obtain its bpm value for later use.
        if self.current_buffer == "a":
            other_bpm = self.calculate_bpm(self.data_buffer_b, self.times_b)
            other_data_length = len(self.data_buffer_b)
        else:
            other_bpm = self.calculate_bpm(self.data_buffer_a, self.times_a)
            other_data_length = len(self.data_buffer_a)

        total_data_length = other_data_length + len(selected_buffer)

        processed = np.array(selected_buffer)
        L = len(selected_buffer)
        # Start heartrate estimation only if there are enough measurements stored in the data buffer
        if L > self.essential_frame_accumulated:
            self.fps = float(L) / (selected_times[-1] - selected_times[0])
            even_times = np.linspace(selected_times[0], selected_times[-1], L)
            interpolated = np.interp(even_times, selected_times, processed)
            interpolated = np.hamming(L) * interpolated
            interpolated = interpolated - np.mean(interpolated)
            raw = np.fft.rfft(interpolated)
            phase = np.angle(raw)
            self.fft = np.abs(raw)
            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)
            freqs = 60.0 * self.freqs
            idx = np.where((freqs > 50) & (freqs < 220))
            pruned = self.fft[idx]
            phase = phase[idx]

            pfreq = freqs[idx]
            self.freqs = pfreq
            self.fft = pruned
            idx2 = np.argmax(pruned)

            t = (np.sin(phase[idx2]) + 1.0) / 2.0
            t = 0.9 * t + 0.1
            self.alpha = t

            gap = (self.buffer_size - L) / self.fps

            # If the selected data buffer is not completely filled, we don't consider it a steady signal.
            if gap:
                # Since the signal is not considered steady, we use a steady signal from the other data buffer to
                # perform weighting on the final result
                this_bpm = self.freqs[idx2]
                other_portion = (other_data_length / total_data_length) * other_bpm
                this_portion = (len(selected_buffer) / total_data_length) * this_bpm
                self.bpm = other_portion + this_portion
                self.bpms.append(self.bpm)
                if other_bpm == 0:
                    # If there isn't a steady signal in the other data buffer, we don't return the current value.
                    return 0
                else:
                    # If a weighted value is obtained, we return the moving average of the value.
                    return self.bpms.calc_average()
            else:
                self.bpm = self.freqs[idx2]
                self.bpms.append(self.bpm)
                return self.bpms.calc_average()
        else:
            if other_data_length >= self.buffer_size:
                return self.bpms.calc_average()
            else:
                return 0

    def draw_vitals(self, frame, bpm=0):
        # Make sure a face is detected before drawing anything
        if self.face_rect is not None:
            x, y, w, h = self.face_rect
            self.draw_disconnected_rect(frame, (x, y), (x + w, y + h), COLOR[1], 2)
            test = ""
            if not self.find_faces:
                text = "Press 'S' to stop heart rate estimation"
                if bpm == 0:
                    self.draw_fade_heart(frame, x, y, w, h)
                else:
                    self.draw_pump_heart(frame, x, y, self.alpha)
                    self.draw_bpm_text(frame, bpm, x, y)
            else:
                text = "Press 'S' to begin heart rate estimation"

            textSize = self.ft.getTextSize(text, fontHeight=17, thickness=-1)[0]
            self.ft.putText(
                img=frame,
                text=text,
                org=(10 + 1, 10 + textSize[1] + 1),
                fontHeight=17,
                color=(0, 0, 0),
                thickness=-1,
                line_type=cv2.LINE_AA,
                bottomLeftOrigin=True,
            )
            self.ft.putText(
                img=frame,
                text=text,
                org=(10, 10 + textSize[1]),
                fontHeight=17,
                color=(255, 255, 255),
                thickness=-1,
                line_type=cv2.LINE_AA,
                bottomLeftOrigin=True,
            )
