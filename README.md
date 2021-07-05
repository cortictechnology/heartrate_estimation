# heartrate_estimation
Vision-based heart rate estimation using OAK-D camera

## Install dependencies

On a Raspberry Pi 4B or a PC with Ubuntu/Debian, run in terminal:

```
git clone https://github.com/cortictechnology/heartrate_estimation.git
cd heartrate_estimation
bash install_dependencies.sh
```

## To run

1. Make sure the OAK-D device is plug into the Pi or PC.
2. In the terminal, run
```
python3 main.py
```

To steadily measure the heart rate, we use a threshold value to determine if there is a change in the environment's light settings. You can set this value with
the --pixel_thresh option. 
```
python3 main.py --pixel_thresh 2.0
```
The default value for --pixel_thresh is 1.5. You will need to adjust this value based on your own light settings. If the value is too low, the program may be too sensitive and never obtain a steady heart rate signal. If the value is too high, the program may extract an inaccurate heart rate signal.

## Model description

In the models folder, only one model is provided:

1. face-detection-retail-0004_openvino_2021.2_6shave.blob: Face detection nework from [depthai-experiments](https://github.com/luxonis/depthai-experiments/tree/master/gen2-age-gender)
