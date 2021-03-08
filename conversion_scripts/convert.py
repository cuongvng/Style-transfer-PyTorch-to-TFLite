import torch
from torchvision import transforms
import tensorflow as tf
import sys
sys.path.append("../")
from models.pytorch_models.model_arch import TransformerNet
import onnx
import onnxruntime
import argparse
import subprocess

def torch_to_onnx(torch_model_path, onnx_model_path):
    print("Converting to onnx ...")
    state_dict = torch.load(torch_model_path)

    keys_to_del = []

    for i in range(1, 6):
        keys_to_del.extend([
            f"in{i}.running_mean",
            f"in{i}.running_var",
        ])

    for i in range(1,3):
        for r in range(1, 6):
            keys_to_del.extend([
                f"res{r}.in{i}.running_mean",
                f"res{r}.in{i}.running_var"
            ])

    for k in keys_to_del:
        del state_dict[k]
        
    model = TransformerNet()
    model.load_state_dict(state_dict)
    model.eval()
    
    # Just some random number
    batch_size = 1
    height = 224
    width = 224

    x = torch.randn(batch_size, 3, height, width, requires_grad=True)
    output = model(x)

    # Convert the PyTorch model to ONNX format
    torch.onnx.export(model,                     # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      onnx_model_path,           # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for quantization
                      input_names = ["input"],   # the model"s input names
                      output_names = ["output"], # the model"s output names
                      dynamic_axes={"input" : {0: "batch_size", 2: "height", 3: "width"},    # variable lenght axes
                                    "output" : {0: "batch_size",  2: "height", 3: "width"}})
    print("\tDone!")

def onnx_to_tf(onnx_model_path, tf_model_path):
    print("Converting to tf ...")
    subprocess.run(["onnx-tf", "convert", "-i", onnx_model_path, "-o" , tf_model_path])
    print("\tDone!")

def tf_to_tflite(tf_model_path, tflite_model_path, quantization="default"):
    print("Converting to tflite ...")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    
    # tell converter which type of quantization techniques to use
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if quantization == "int8":
        pass
#         def representative_dataset():
#             pass
#         converter.representative_dataset = representative_dataset
#         converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#         converter.inference_input_type = tf.int8  # or tf.uint8
#         converter.inference_output_type = tf.int8  # or tf.uint8

    elif quantization == "float16":
        converter.target_spec.supported_types = [tf.float16]
        
    else:
        pass
        
    tf_lite_model = converter.convert()
    open(tflite_model_path, "wb").write(tf_lite_model)
    print("\tDone!")

def main():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--torch_model_path", type=str, required=True,
                                 help="path to read input pytorch model")
    arg_parser.add_argument("--tflite_model_path", type=str, default="./tflite_models/rain_princess.tflite",
                                 help="path to save output tflite model")
    arg_parser.add_argument("--quantization", type=str, default="default", choices=["default", "int8", "float16"],
                                 help="quantization technique for TF Lite converter")

    args = arg_parser.parse_args()
    torch_model_path = args.torch_model_path
    onnx_model_path = "../models/temp/rain_princess.onnx"
    tf_model_path = "../models/temp/rain_princess.pb"
    tflite_model_path = args.tflite_model_path
    quantization = args.quantization
    
    torch_to_onnx(torch_model_path, onnx_model_path)
    onnx_to_tf(onnx_model_path, tf_model_path)
    tf_to_tflite(tf_model_path, tflite_model_path, quantization)


if __name__ == "__main__":
    main()
