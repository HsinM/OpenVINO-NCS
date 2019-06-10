# OpenVINO-NCS
 Convert keras model to intel IR with openvino and inference with intel NCS/NCS v2

## Directory structure and description

```
.
│
├─installers
│     README.txt            # Website where can download OpenVINO installer.
│
├─pi_code                   # Sample code of inference data on Pi 3
│  ├─face                        # Detect faces using image file.
│  │  ├─code
│  │  │      test.py
│  │  │
│  │  ├─data
│  │  └─model
│  │
│  └─mnist                       # Detect digits
│      ├─code
│      │      static_digit_detection.py         # using image files.
│      │      webcam_digit_detection.py         # using webcam.
│      ├─data
│      └─model
│
└─win_linux_code             # Sample code of Conversion model on x86/64 platforms
        script_h5_to_xml_bin.py                 # script of convert model (h5 -> xml, bin)
        train-mnist-cnn.ipynb                   # train mnist cnn model and 
                                                  generate IR (bin/xml) model at same time.
        train-mnist-cnn.py                      # ipython interpeter version as above.
```
