# OpenVINO-NCS
 convert keras model to intel IR with openvino and inference with intel NCS/NCS v2

## Directory structure and description

```
.
│
│  # some unimportant files
│
├─installers
│     README.txt            # Website where can download OpenVINO installer.
│
├─pi_code                   # Sample code of inference data on Pi 3
│  ├─face                       # Detect faces using image file.
│  │  ├─code
│  │  │      test.py
│  │  │
│  │  ├─data
│  │  └─model
│  │
│  └─mnist
│      ├─code                   # Detect digits
│      │      static_digit_detection.py         # using image files.
│      │      webcam_digit_detection.py         # using webcam.
│      ├─data
│      └─model
│
└─win_linux_code                # Sample code of Conversion model on x86/64 platforms
        conv_pb_to_bin_xml.sh                   # Conversion PB model to IR (bin/xml) model.
        h5_to_pb_to_vino_file.ipynb             # Conversion keras (h5) model to IR (bin/xml) model notebook.
        h5_to_pb_to_vino_file.py                # ipython interpeter version as above.
        train-mnist-cnn.ipynb                   # train mnist cnn model and 
                                                  generate IR (bin/xml) model at same time.
        train-mnist-cnn.py                      # ipython interpeter version as above.
```
