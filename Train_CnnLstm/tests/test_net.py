import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, out_channels, num_layers, device):
        super(LSTM, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)

        self.out_channels = out_channels

        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        out, _ = self.lstm(x)

        if self.out_channels == 1:
            out = out[:, -1, :]
            out = self.fc(out)
            return out

        return out


batch_size = 1
input_size = 20
seq_len = 5
output_size = 10
num_layers = 2
out_channels = 1

model = LSTM(input_size, output_size, out_channels, num_layers, "cpu")
model.eval()

input_names = ["input"]  # 设定输入接口名称
output_names = ["output"]  # 设定输出接口名称

x = torch.randn((batch_size, seq_len, input_size))
print(x.shape)
y = model(x)
print(y.shape)

torch.onnx.export(model, x, 'lstm.onnx', verbose=True, input_names=input_names, output_names=output_names,
                  dynamic_axes={'input': [0], 'output': [0]})

# import onnx
#
# model = onnx.load("lstm.onnx")
# print("load model done.")
# onnx.checker.check_model(model)
# print(onnx.helper.printable_graph(model.graph))
# print("check model done.")
