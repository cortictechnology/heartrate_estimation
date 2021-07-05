# This demo is modified based on this project: https://github.com/thearn/webcam-pulse-detector

from heartrate_estimator import HeartrateEstimator
import cv2
import numpy as np
import time
import sys
import argparse
import depthai as dai
from pathlib import Path
from utils import frame_norm, to_planar


class getVitalApp(object):
    def __init__(self, pixel_threshold):
        self.pressed = 0

        self.processor = HeartrateEstimator(pixel_threshold=pixel_threshold)

        self.key_controls = {"s": self.toggle_search}

        # Create an OAK-D pipeline
        self.pipeline = dai.Pipeline()

        # Setup color camera node
        colorCam = self.pipeline.createColorCamera()
        xoutRgb = self.pipeline.createXLinkOut()
        xoutRgb.setStreamName("rgb")
        colorCam.setPreviewSize(800, 450)
        colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        colorCam.setPreviewKeepAspectRatio(False)
        colorCam.setInterleaved(False)
        colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        colorCam.initialControl.setManualFocus(130)

        # Using ImageManip node to create input to NN input
        manip = self.pipeline.createImageManip()
        manip.initialConfig.setResize(300, 300)
        manip.setKeepAspectRatio(False)
        colorCam.preview.link(manip.inputImage)

        # Create face detection NN node
        detection_nn = self.pipeline.createNeuralNetwork()
        detection_nn.setBlobPath(
            str(
                (
                    Path(__file__).parent
                    / Path(
                        "./models/face-detection-retail-0004_openvino_2021.2_6shave.blob"
                    )
                )
                .resolve()
                .absolute()
            )
        )

        # Feed camera input to NN
        manip.out.link(detection_nn.input)
        colorCam.preview.link(xoutRgb.input)
        xout_det = self.pipeline.createXLinkOut()
        xout_det.setStreamName("det_nn")
        detection_nn.out.link(xout_det.input)

        # Using landmark to locate a skin region is possible too, need more tests.

        # landmarks_nn = self.pipeline.createNeuralNetwork()
        # landmarks_nn.setBlobPath(
        #     str(
        #         (
        #             Path(__file__).parent
        #             / Path(
        #                 "landmarks-regression-retail-0009_openvino_2021.2_6shave.blob"
        #             )
        #         )
        #         .resolve()
        #         .absolute()
        #     )
        # )
        # xin_land = self.pipeline.createXLinkIn()
        # xin_land.setStreamName("land_in")
        # xin_land.out.link(landmarks_nn.input)
        # xout_land = self.pipeline.createXLinkOut()
        # xout_land.setStreamName("land_nn")
        # landmarks_nn.out.link(xout_land.input)

        self.device = dai.Device(self.pipeline)

        # Setup device output queues
        self.previewQueue = self.device.getOutputQueue(
            name="rgb", maxSize=4, blocking=False
        )
        self.q_det = self.device.getOutputQueue(
            name="det_nn", maxSize=4, blocking=False
        )

        # self.land_in = self.device.getInputQueue(
        #     name="land_in", maxSize=4, blocking=False
        # )
        # self.q_land = self.device.getOutputQueue(
        #     name="land_nn", maxSize=4, blocking=False
        # )

    def toggle_search(self):
        """
        Toggles a motion lock on the processor's face detection component.

        Locking the forehead location in place significantly improves
        data quality, once a forehead has been sucessfully isolated.
        """
        # state = self.processor.find_faces.toggle()
        state = self.processor.find_faces_toggle()

    def key_handler(self):
        """
        Handle keystrokes, as set at the bottom of __init__()

        A plotting or camera frame window must have focus for keypresses to be
        detected.
        """

        self.pressed = cv2.waitKey(1) & 255  # wait for keypress for 10 ms
        if self.pressed == 27:  # exit program on 'esc'
            print("Exiting")
            sys.exit()

        for key in self.key_controls.keys():
            if chr(self.pressed) == key:
                self.key_controls[key]()

    def run(self):
        while True:
            # t1 = time.monotonic()
            frame = self.previewQueue.get().getCvFrame()
            bboxes = np.array(self.q_det.get().getFirstLayerFp16())
            bboxes = bboxes.reshape((bboxes.size // 7, 7))
            bboxes = bboxes[bboxes[:, 2] > 0.5][:, 3:7]
            largest_bbox = []
            largest_area = 0
            landmarks = []
            for raw_bbox in bboxes:
                bbox = frame_norm(frame, raw_bbox)
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area > largest_area:
                    largest_area = area
                    largest_bbox = bbox

            # Comment out until more tests on using landmarks for skin reigon are performed
            # if largest_bbox != []:
            #     face_frame = frame[
            #         largest_bbox[1] : largest_bbox[3], largest_bbox[0] : largest_bbox[2]
            #     ]

            #     nn_data = dai.NNData()
            #     nn_data.setLayer("0", to_planar(face_frame, (48, 48)))
            #     self.land_in.send(nn_data)
            #     landmarks = self.q_land.get().getFirstLayerFp16()

            # set current image frame to the processor's input
            # process the image frame to perform all needed analysis
            # t1 = time.monotonic()
            bpm = self.processor.measure_pulse(frame, largest_bbox)

            self.processor.draw_vitals(frame, bpm)
            # print("Pulse Time:", time.monotonic() - t1)

            # show the processed/annotated output frame
            cv2.imshow("Heart Pulse Estimation", frame)

            # handle any key presses
            self.key_handler()
            # print("Time:", time.monotonic() - t1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pixel_thresh",
        default=1.5,
        type=float,
        help="Threshold value for the buffer switching mechanism.",
    )
    args = parser.parse_args()
    App = getVitalApp(pixel_threshold=args.pixel_thresh)
    App.run()
