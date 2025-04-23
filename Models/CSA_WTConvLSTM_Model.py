import torch
from torch import nn

from Modules.CSA_Module import CSA_Module
from Modules.WTConvLSTM_Modules import WTConvLSTMCell


def check_hidden_dim_num_layers(WTConvLSTM_hidden_dim, WT_num_layers):
    if len(WTConvLSTM_hidden_dim) != WT_num_layers:
        raise RuntimeError('len(WTConvLSTM_hidden_dim) != WT_num_layers !!!!')


class CSA_WTConvLSTM(nn.Module):
    def __init__(self, input_dim, CSA_hidden_dim, CSA_num_layers, WTConvLSTM_hidden_dim, WT_num_layers,
                 height, width, kernel_size, CA_bool=True, SA_bool=True, predict_step=12,
                 batch_first=True, bias=False, channel_second=False, return_all_layers=False,):
        super(CSA_WTConvLSTM, self).__init__()
        self.input_dim = input_dim
        self.CSA_hidden_dim = CSA_hidden_dim
        self.CSA_num_layers = CSA_num_layers

        self.WTConvLSTM_hidden_dim = WTConvLSTM_hidden_dim
        self.WT_num_layers = WT_num_layers
        check_hidden_dim_num_layers(self.WTConvLSTM_hidden_dim, self.WT_num_layers)

        self.height = height
        self.width = width
        self.kernel_size = kernel_size

        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        self.predict_step = predict_step

        self.CA_bool = CA_bool
        self.SA_bool = SA_bool
        self.channel_second = channel_second
        self.CSA_Module = CSA_Module(in_channel=self.input_dim, CSA_hidden_dim=self.CSA_hidden_dim,
                                     CSA_num_layers=self.CSA_num_layers, kernel_size=self.kernel_size,
                                     channel_second=self.channel_second, CA_bool=self.CA_bool, SA_bool=self.SA_bool)

        cell_list = []
        for i in range(0, WT_num_layers):
            cur_input_dim = CSA_hidden_dim[-1] if i == 0 else self.WTConvLSTM_hidden_dim[i - 1]
            cell_list.append(WTConvLSTMCell(in_channel=cur_input_dim,
                                            num_hidden=self.WTConvLSTM_hidden_dim[i],
                                            height=self.height,
                                            width=self.width,
                                            filter_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)
        self.batch_norm = nn.BatchNorm2d(self.WTConvLSTM_hidden_dim[-1])
        self.Conv_last = nn.Conv3d(in_channels=self.WTConvLSTM_hidden_dim[-1], out_channels=1,
                                   kernel_size=self.kernel_size, stride=1, padding=1)
        self.sig = nn.Sigmoid()

    def forward(self, input_tensor, hidden_state=None):
        input_tensor = self.CSA_Module(input_tensor)

        ###################################################################################

        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4).contiguous()

        b, _, _, h, w = input_tensor.size()
        hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)

        cur_layer_input = input_tensor

        for layer_idx in range(self.WT_num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](x_t=cur_layer_input[:, t, :, :, :], h_t=h, c_t=c)
                h = self.batch_norm(h)
                c = self.batch_norm(c)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            last_state_list = last_state_list[-1]

        output = layer_output_list[:, -self.predict_step:, :, :, :]
        if not self.channel_second:
            # b t c h w ---> b c t h w
            output = output.permute(0, 2, 1, 3, 4).contiguous()
        output = self.Conv_last(output)
        if not self.channel_second:
            # b c t h w ---> b t c h w
            output = output.permute(0, 2, 1, 3, 4).contiguous()
        output = self.sig(output)
        return output

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.WT_num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states


if __name__ == '__main__':
    model = CSA_WTConvLSTM(input_dim=1, CSA_hidden_dim=[8], CSA_num_layers=1, WTConvLSTM_hidden_dim=[8, 8],
                           WT_num_layers=2, height=71, width=73, kernel_size=3, CA_bool=True, SA_bool=True).cuda()
    print(model)
    input = torch.randn(10, 24, 1, 71, 73).cuda()  # b s c h w
    out = model(input)
    print(out.size())
