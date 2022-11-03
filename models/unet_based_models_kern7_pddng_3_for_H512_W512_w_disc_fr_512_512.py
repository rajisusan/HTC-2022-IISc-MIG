import torch
import torch.nn as nn

from collections import OrderedDict

# Changelog:
# 1. [17/01/2022]: Added ReLu in _decoder_block_1st to get only non-negative numbers.
# 2. [17/01/2022]:


# Generator Code
class Generator(nn.Module):

    def __init__(self, opt):
        super(Generator, self).__init__()

        num_input_image_channels = opt.nc
        num_channels_enc_1st_conv = opt.nef
        num_features = num_channels_enc_1st_conv

        # Encoder part:

        # input is (nc = 1) x (H = rows = 512) x (W = cols = 512)
        # Conv operation 1
        self.encoder_block_1 = Generator._encoder_block_1st(num_input_image_channels, num_features, block_prefix="enc_block_1")
        # self.encoder_block_1 = nn.DataParallel(self.encoder_block_1)

        # input is (nc = 1) x (H = rows = 512) x (W = cols = 512)
        # MaxPool operation 1
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=(0, 0))

        # input is (nc = 1) x (H = rows = 256) x (W = cols = 256)
        # Conv Operation 2
        self.encoder_block_2 = Generator._encoder_block_except_1st(num_features, 2*num_features, block_prefix="enc_block_2")

        # input is (nc = 1) x (H = rows = 256) x (W = cols = 256)
        # MaxPool operation 2
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # input is (nc = 1) x (H = rows = 128) x (W = cols = 128)
        # Conv Operation 3
        self.encoder_block_3 = Generator._encoder_block_except_1st(2*num_features, 4*num_features, block_prefix="enc_block_3")

        # input is (nc = 1) x (H = rows = 128) x (W = cols = 128)
        # MaxPool operation 3
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=(0, 0))

        # input is (nc = 1) x (H = rows = 64) x (W = cols = 64)
        # Conv Operation 4
        self.encoder_block_4 = Generator._encoder_block_except_1st(4*num_features, 8*num_features, block_prefix="enc_block_4")

        # input is (nc = 1) x (H = rows = 64) x (W = cols = 64)
        # MaxPool operation 4
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # input is (nc = 1) x (H = rows = 32) x (W = cols = 32)
        # Conv Operation 5
        self.encoder_block_5 = Generator._encoder_block_except_1st(8*num_features, 8*num_features, block_prefix="enc_block_5")

        # input is (nc = 1) x (H = rows = 32) x (W = cols = 32)
        # MaxPool operation 5
        self.max_pool_5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=(0, 0))

        # input is (nc = 1) x (H = rows = 16) x (W = cols = 16)
        # Conv Operation 6
        self.encoder_block_6 = Generator._encoder_block_except_1st(8*num_features, 8*num_features, block_prefix="enc_block_6")

        # input is (nc = 1) x (H = rows = 16) x (W = cols = 16)
        # MaxPool operation 6
        self.max_pool_6 = nn.MaxPool2d(kernel_size=2, stride=2, padding=(0, 0))

        # input is (nc = 1) x (H = rows = 8) x (W = cols = 8)
        # Conv Operation 7
        self.encoder_block_7 = Generator._encoder_block_except_1st(8*num_features, 8*num_features, block_prefix="enc_block_7")

        # input is (nc = 1) x (H = rows = 8) x (W = cols = 8)
        # MaxPool operation 7
        self.max_pool_7 = nn.MaxPool2d(kernel_size=2, stride=2, padding=(0, 0))

        # input is (nc = 1) x (H = rows = 4) x (W = cols = 4)
        # Conv Operation 8
        self.encoder_block_8 = Generator._encoder_block_except_1st(8*num_features, 8*num_features, block_prefix="enc_block_8")

        # input is (nc = 1) x (H = rows = 4) x (W = cols = 4)
        # MaxPool operation 8
        self.max_pool_8 = nn.MaxPool2d(kernel_size=2, stride=2, padding=(0, 0))

        # input is (nc = 1) x (H = rows = 2) x (W = cols = 2)
        # Conv Operation 9
        self.encoder_block_9 = Generator._encoder_block_except_1st(8*num_features, 8*num_features, block_prefix="enc_block_9")

        # input is (nc = 1) x (H = rows = 2) x (W = cols = 2)
        # MaxPool operation 9
        self.max_pool_9 = nn.MaxPool2d(kernel_size=2, stride=2)

        # input is (nc = 1) x (H = rows = 1) x (W = cols = 1)
        # Bottleneck
        self.bottleneck = Generator._decoder_block_for_bottleneck_and_9th_to_5th(8*num_features, 8*num_features, block_prefix="bottleneck")

        # Decoder part:

        # input is (nc = 1) x (H = rows = 1) x (W = cols = 1)
        # Upsampling operation 9
        self.upsample_9 = nn.Upsample((2, 2))

        # input is (nc = 1) x (H = rows = 2) x (W = cols = 2)
        # ConvTransp operation 9
        self.decoder_block_9 = Generator._decoder_block_for_bottleneck_and_9th_to_5th(2*8*num_features, 8*num_features, block_prefix="dec_block_9")

        # input is (nc = 1) x (H = rows = 2) x (W = cols = 2)
        # Upsampling operation 8
        self.upsample_8 = nn.Upsample((4, 4))

        # input is (nc = 1) x (H = rows = 4) x (W = cols = 4)
        # ConvTransp operation 8
        self.decoder_block_8 = Generator._decoder_block_for_bottleneck_and_9th_to_5th(2*8*num_features, 8*num_features, block_prefix="dec_block_8")

        # input is (nc = 1) x (H = rows = 4) x (W = cols = 4)
        # Upsampling operation 7
        self.upsample_7 = nn.Upsample((8, 8))

        # input is (nc = 1) x (H = rows = 8) x (W = cols = 8)
        # ConvTransp operation 7
        self.decoder_block_7 = Generator._decoder_block_for_bottleneck_and_9th_to_5th(2*8*num_features, 8*num_features, block_prefix="dec_block_7")

        # input is (nc = 1) x (H = rows = 8) x (W = cols = 8)
        # Upsampling operation 6
        self.upsample_6 = nn.Upsample((16, 16))

        # input is (nc = 1) x (H = rows = 16) x (W = cols = 16)
        # ConvTransp operation 6
        self.decoder_block_6 = Generator._decoder_block_for_bottleneck_and_9th_to_5th(2*8*num_features, 8*num_features, block_prefix="dec_block_6")

        # input is (nc = 1) x (H = rows = 16) x (W = cols = 16)
        # Upsampling operation 5
        self.upsample_5 = nn.Upsample((32, 32))

        # input is (nc = 1) x (H = rows = 32) x (W = cols = 32)
        # ConvTransp operation 5
        self.decoder_block_5 = Generator._decoder_block_for_bottleneck_and_9th_to_5th(2*8*num_features, 8*num_features, block_prefix="dec_block_5")

        # input is (nc = 1) x (H = rows = 32) x (W = cols = 32)
        # Upsampling operation 4
        self.upsample_4 = nn.Upsample((64, 64))

        # input is (nc = 1) x (H = rows = 64) x (W = cols = 64)
        # ConvTransp operation 4
        self.decoder_block_4 = Generator._decoder_block_4th_to_2nd(2*8*num_features, 4*num_features, block_prefix="dec_block_4")

        # input is (nc = 1) x (H = rows = 64) x (W = cols = 64)
        # Upsampling operation 3
        self.upsample_3 = nn.Upsample((128, 128))

        # input is (nc = 1) x (H = rows = 128) x (W = cols = 128)
        # ConvTransp operation 3
        self.decoder_block_3 = Generator._decoder_block_4th_to_2nd(2*4*num_features, 2*num_features, block_prefix="dec_block_3")

        # input is (nc = 1) x (H = rows = 128) x (W = cols = 128)
        # Upsampling operation 2
        self.upsample_2 = nn.Upsample((256, 256))

        # input is (nc = 1) x (H = rows = 256) x (W = cols = 256)
        # ConvTransp operation 2
        self.decoder_block_2 = Generator._decoder_block_4th_to_2nd(2*2*num_features, num_features, block_prefix="dec_block_2")

        # input is (nc = 1) x (H = rows = 256) x (W = cols = 256)
        # Upsampling operation 1
        self.upsample_1 = nn.Upsample((512, 512))

        # input is (nc = 1) x (H = rows = 512) x (W = cols = 512)
        # ConvTransp operation 1
        self.decoder_block_1 = Generator._decoder_block_1st(2*num_features, num_input_image_channels, block_prefix="dec_block_1")

    def forward(self, input):

        enc1_output = self.encoder_block_1(input)
        enc2_output = self.encoder_block_2(self.max_pool_1(enc1_output))
        enc3_output = self.encoder_block_3(self.max_pool_2(enc2_output))
        enc4_output = self.encoder_block_4(self.max_pool_3(enc3_output))
        enc5_output = self.encoder_block_5(self.max_pool_4(enc4_output))
        enc6_output = self.encoder_block_6(self.max_pool_5(enc5_output))
        enc7_output = self.encoder_block_7(self.max_pool_6(enc6_output))
        enc8_output = self.encoder_block_8(self.max_pool_7(enc7_output))
        enc9_output = self.encoder_block_9(self.max_pool_8(enc8_output))        # output dim: (batchsize, 1, 4, 4)

        bottleneck_output = self.bottleneck(self.max_pool_9(enc9_output))       # output dim: (batchsize, 1, 2, 2)

        dec9_input = torch.cat((self.upsample_9(bottleneck_output), enc9_output), dim=1)
        dec9_output = self.decoder_block_9(dec9_input)

        dec8_input = torch.cat((self.upsample_8(dec9_output), enc8_output), dim=1)
        dec8_output = self.decoder_block_8(dec8_input)

        dec7_input = torch.cat((self.upsample_7(dec8_output), enc7_output), dim=1)
        dec7_output = self.decoder_block_7(dec7_input)

        dec6_input = torch.cat((self.upsample_6(dec7_output), enc6_output), dim=1)
        dec6_output = self.decoder_block_6(dec6_input)

        dec5_input = torch.cat((self.upsample_5(dec6_output), enc5_output), dim=1)
        dec5_output = self.decoder_block_5(dec5_input)

        dec4_input = torch.cat((self.upsample_4(dec5_output), enc4_output), dim=1)
        dec4_output = self.decoder_block_4(dec4_input)

        dec3_input = torch.cat((self.upsample_3(dec4_output), enc3_output), dim=1)
        dec3_output = self.decoder_block_3(dec3_input)

        dec2_input = torch.cat((self.upsample_2(dec3_output), enc2_output), dim=1)
        dec2_output = self.decoder_block_2(dec2_input)

        dec1_input = torch.cat((self.upsample_1(dec2_output), enc1_output), dim=1)
        dec1_output = self.decoder_block_1(dec1_input)

        # https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
        # print("\tIn Model: input size", input.size(),
        #   "output size", dec1_output.size())

        # return nn.Tanh(dec1_output)
        return dec1_output.clone()
        # return output

    # First block in the encoder part
    #
    @staticmethod
    def _encoder_block_1st(num_in_channels, num_out_channels, block_prefix):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        block_prefix + "_conv2d",
                        nn.Conv2d(
                            in_channels=num_in_channels,
                            out_channels=num_out_channels,
                            kernel_size=7,
                            padding=3,
                            stride=1,
                            bias=False
                        )
                    ),

                    (
                        block_prefix + "_leaky_relu",
                        nn.LeakyReLU(inplace=True)
                    )

                ]

            )
        )

    # For all blocks, except the first one, in the encoder part
    #
    @staticmethod
    def _encoder_block_except_1st(num_in_channels, num_out_channels, block_prefix):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        block_prefix + "_conv2d",
                        nn.Conv2d(
                            in_channels=num_in_channels,
                            out_channels=num_out_channels,
                            kernel_size=7,
                            padding=3,
                            stride=1,
                            bias=False
                        )
                    ),

                    (
                     block_prefix + "_batch_norm",
                     nn.BatchNorm2d(num_features=num_out_channels)
                    ),

                    (
                        block_prefix + "_leaky_relu",
                        nn.LeakyReLU(inplace=True)
                    )

                ]

            )
        )

    # For blocks 9th to 5th in the decoder part
    #
    @staticmethod
    def _decoder_block_for_bottleneck_and_9th_to_5th(num_in_channels, num_out_channels, block_prefix):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        block_prefix + "_convTransp2d",
                        nn.ConvTranspose2d(
                            in_channels=num_in_channels,
                            out_channels=num_out_channels,
                            kernel_size=7,
                            stride=1,
                            padding=3,
                            dilation=1,
                            bias=False
                        )
                    ),

                    # Following commented for batch size == 1
                    #   File "/home/miguser1/Documents/Project/Programming/DL_sinogram_completion/2022_05_09_Model_sinogram_extension_by_deconvolution_resumed_from_remote/models/
                    #     unet_based_models_for_H512_W512.py", line 186, in forward
                    #     bottleneck_output = self.bottleneck(self.max_pool_9(enc9_output))       # output dim: (batchsize, 1, 2, 2)
                    #   File "/home/miguser1/anaconda3/envs/env_for_PyTorch_with_Python_3_8/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
                    #     return forward_call(*input, **kwargs)
                    #   File "/home/miguser1/anaconda3/envs/env_for_PyTorch_with_Python_3_8/lib/python3.8/site-packages/torch/nn/modules/container.py", line 141, in forward
                    #     input = module(input)
                    #   File "/home/miguser1/anaconda3/envs/env_for_PyTorch_with_Python_3_8/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
                    #     return forward_call(*input, **kwargs)
                    #   File "/home/miguser1/anaconda3/envs/env_for_PyTorch_with_Python_3_8/lib/python3.8/site-packages/torch/nn/modules/batchnorm.py", line 168, in forward
                    #     return F.batch_norm(
                    #   File "/home/miguser1/anaconda3/envs/env_for_PyTorch_with_Python_3_8/lib/python3.8/site-packages/torch/nn/functional.py", line 2280, in batch_norm
                    #     _verify_batch_size(input.size())
                    #   File "/home/miguser1/anaconda3/envs/env_for_PyTorch_with_Python_3_8/lib/python3.8/site-packages/torch/nn/functional.py", line 2248, in _verify_batch_size
                    #     raise ValueError("Expected more than 1 value per channel when training, got input size {}".format(size))
                    # ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 512, 1, 1])
                    (
                     block_prefix + "_batch_norm",
                     nn.BatchNorm2d(num_features=num_out_channels)
                    ),

                    (
                      block_prefix + "_dropout",
                      nn.Dropout2d(p=0.2)
                    ),

                    (
                        block_prefix + "_relu",
                        nn.ReLU(inplace=True)
                    )

                ]

            )
        )

    # For blocks 4th to 1st in the decoder part
    #
    @staticmethod
    def _decoder_block_4th_to_2nd(num_in_channels, num_out_channels, block_prefix):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        block_prefix + "_convTransp2d",
                        nn.ConvTranspose2d(
                            in_channels=num_in_channels,
                            out_channels=num_out_channels,
                            kernel_size=7,
                            stride=1,
                            padding=3,
                            dilation=1,
                            bias=False
                        )
                    ),

                    (
                     block_prefix + "_batch_norm",
                     nn.BatchNorm2d(num_features=num_out_channels)
                    ),

                    (
                        block_prefix + "_relu",
                        nn.ReLU(inplace=True)
                    )

                ]

            )
        )

    # For block 1st in the decoder part
    #
    @staticmethod
    def _decoder_block_1st(num_in_channels, num_out_channels, block_prefix):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        block_prefix + "_convTransp2d",
                        nn.ConvTranspose2d(
                            in_channels=num_in_channels,
                            out_channels=num_out_channels,
                            kernel_size=7,
                            stride=1,
                            padding=3,
                            dilation=1,
                            bias=False
                        )
                    ),

                    (
                        block_prefix + "_relu",
                        nn.ReLU(inplace=True)
                    )

                ]

            )
        )


#######################################################################################################################################
#######################################################################################################################################
