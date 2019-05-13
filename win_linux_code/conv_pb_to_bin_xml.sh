python3 /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo.py --input_model ./trans_model/best_model.pb --output_dir ./vino_model --input_shape [1,28,28,1] --data_type FP16
