# Style-transfer-PyTorch-to-TFLite
Deploy a PyTorch-pretrained neural style transfer to Android mobile using TFLite

### 1. Convert PyTorch-pretrained model to TF-Lite format

- Using pretrained models from [PyTorch examples](https://github.com/pytorch/examples/tree/master/fast_neural_style)

- Conversion flow:

```
    PyTorch -> ONNX -> TF -> TF-Lite
```

- Usage

```
cd ./conversion_scripts
pip install -r requirements.txt
python convert.py --torch_model_path="../models/pytorch_models/rain_princess.pth" \
                  --tflite_model_path="../models/tflite_models/rain_princess.tflite"
```

- For details of the process please checkout [conversion_scripts/convert.ipynb](conversion_scripts/convert.ipynb)

- Model's size reduction: PyTorch (6.43 MB) -> ONNX (6.43 MB) -> TF (7.4 MB) -> TF-Lite:
    - 1.69 MB (~4x smaller) for default (dynamic range) quantization
    - 3.25 MB (~2x smaller) for float16 quantization 