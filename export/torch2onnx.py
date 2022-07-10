import os
import torch
import torch.nn as nn
import numpy as np
import onnx

# 这里导模型路径
from model import MobileNetV2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = 'MobileNetV2.pth'

def torch2onnx():
    model = MobileNetV2(num_classes=2)

    # step1.加载模型参数
    try:
        print('Loading weights into state dict...')
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)

        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            # net = nn.DataParallel(net)
            # model = model.cuda()

        model.eval()

    except AttributeError as error:
        print(error)

    # step2.转换为onnx
    try:
        # 输入为： n c h w
        # 注意要和模型指定输入一致
        dummy_input1 = torch.randn(1, 3, 224, 224)
        # dummy_input2 = torch.randn(1, 3, 64, 64)
        # dummy_input3 = torch.randn(1, 3, 64, 64)
        input_names = ["input"]
        output_names = ["output"]
        # torch.onnxfile.export(model, (dummy_input1, dummy_input2, dummy_input3), "C3AE.onnxfile", verbose=True, input_names=input_names, output_names=output_names)
        torch.onnx.export(model,
                          dummy_input1,
                          model_path.replace('pth','onnx'),
                          verbose=True,
                          input_names=input_names,
                          output_names=output_names
                          )

    except AttributeError as error:
        print(error)


if __name__ == "__main__":
    torch2onnx()