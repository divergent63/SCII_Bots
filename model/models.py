import math
import numpy as np
# import keras
# from keras import backend as K
# from keras.models import Sequential
# from keras.layers import Dense, Conv1D, Conv2D, Dropout, Flatten, Activation, MaxPool1D, MaxPooling2D, Lambda
# from keras.optimizers import Adam, RMSprop
#
# import tensorflow as tf
# from keras import backend as keras_backend
# from tensorflow.python import debug as tf_debug
# from tensorflow.python.debug.lib.debug_data import has_inf_or_nan

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

# # 后端包装 session
# def set_debugger_session():
#     sess = K.get_session()
#     sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#     sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)
#     K.set_session(sess)
#
#
# dbg = False
# if dbg:
#     set_debugger_session()
# else:
#     pass

# keras_backend.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))

# sess = K.get_session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# K.set_session(sess)

LOSS_V = .5
np.random.seed(1)


# class FullyConv:
#     """This class implements the fullyconv agent network from DeepMind paper
#
#     Args
#     ----
#     eta : the entropy regularization hyperparameter
#     expl_rate : the multiplication factor for the policy output Dense layer, used to favorise exploration. Set to 0 if
#         not exploring but exploiting
#     model (Keras.model): the actual keras model"""
#
#     def __init__(self, eta, expl_rate, categorical_actions, spatial_actions):
#         self.eta = eta
#         self.expl_rate = expl_rate
#         self.categorical_actions = categorical_actions
#         self.spatial_actions = spatial_actions
#         self.model = None
#         self.initialize_layers(eta, expl_rate)
#
#     def initialize_layers(self, eta, expl_rate):
#         """Initializes the keras model"""
#
#         # value input
#         # actual_value = keras.layers.Input(shape=(12,), name='actual__value')
#         actual_value = keras.layers.Input(shape=(1,), name='actual__value')
#         # actual_non_spatial = keras.layers.Input(shape=(12,), name='actual__non_spatial')
#         # actual_spatial = keras.layers.Input(shape=(4096,), name='actual__spatial')
#
#         def value_loss():
#             def val_loss(y_true, y_pred):
#                 advantage = y_true - (y_pred+1e-5)
#                 return K.mean(LOSS_V * K.square(advantage))
#
#             return val_loss
#
#         def policy_loss(actual_value, predicted_value):
#             # TODO: the shape of the actual_value are not same as the shape of the predicted_value
#             advantage = actual_value - predicted_value
#
#             def pol_loss(y_true, y_pred):
#                 log_prob = K.log(K.sum((y_pred+1e-5) * y_true, axis=1, keepdims=True) + 1e-5)
#                 return -log_prob * K.stop_gradient(advantage)
#
#             return pol_loss
#
#         input_player = keras.layers.Input(shape=(11, ), name='input_player_info')
#         model_player_info = input_player
#
#         # map conv
#         input_map = keras.layers.Input(shape=(27, 64, 64), name='input_map')
#         model_view_map = Conv2D(16, kernel_size=(5, 5), data_format='channels_first', input_shape=(17, 64, 64),
#                                 kernel_initializer="he_uniform")(input_map)
#         model_view_map = Activation('relu')(model_view_map)
#         model_view_map = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first')(
#             model_view_map)
#         model_view_map = Conv2D(32, kernel_size=(3, 3), data_format='channels_first', kernel_initializer="he_uniform")(
#             model_view_map)
#         model_view_map = Activation('relu')(model_view_map)
#         model_view_map = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first')(
#             model_view_map)
#
#         # minimap conv
#         input_mini = keras.layers.Input(shape=(11, 64, 64), name='input_mini')
#         model_view_mini = Conv2D(16, kernel_size=(5, 5), data_format='channels_first', input_shape=(7, 64, 64),
#                                  kernel_initializer="he_uniform")(input_mini)
#         model_view_mini = Activation('relu')(model_view_mini)
#         model_view_mini = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first')(
#             model_view_mini)
#         model_view_mini = Conv2D(32, kernel_size=(3, 3), data_format='channels_first', kernel_initializer="he_uniform")(
#             model_view_mini)
#         model_view_mini = Activation('relu')(model_view_mini)
#         model_view_mini = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first')(
#             model_view_mini)
#
#         # concatenate
#         concat = keras.layers.concatenate([model_view_map, model_view_mini])
#
#         # value estimate and Action policy
#         intermediate = Flatten()(concat)
#         concat_f = keras.layers.concatenate([intermediate, model_player_info])
#
#         intermediate_f = keras.layers.Dense(256, activation='relu', kernel_initializer="he_uniform")(concat_f)
#
#         out_value = keras.layers.Dense(1)(intermediate_f)
#         out_value = Activation('linear', name='value_output')(out_value)
#
#         out_non_spatial = keras.layers.Dense(len(self.categorical_actions) + len(self.spatial_actions),
#                                              kernel_initializer="he_uniform"
#                                              )(intermediate)
#         out_non_spatial = Lambda(lambda x: self.expl_rate * x)(out_non_spatial)
#         out_non_spatial = Activation('softmax', name='non_spatial_output')(out_non_spatial)
#
#         # spatial policy output
#         out_spatial = Conv2D(1, kernel_size=(1, 1), data_format='channels_first', kernel_initializer="he_uniform",
#                              name='out_spatial')(concat)
#         out_spatial = Flatten()(out_spatial)
#
#         # out_spatial = Dense(12, activation='softmax', kernel_initializer="he_uniform")(out_spatial)
#         out_spatial = Dense(4096, activation='softmax', kernel_initializer="he_uniform")(out_spatial)
#
#         out_spatial = Activation('softmax', name='spatial_output')(out_spatial)
#
#         # compile
#         model = keras.models.Model(inputs=[input_map, input_mini, input_player, actual_value],
#                                    outputs=[out_value, out_non_spatial, out_spatial])
#         model.summary()
#         # losses = {
#         #     "value_output": value_loss(),
#         #     "non_spatial_output": policy_loss(actual_value=actual_non_spatial, predicted_value=out_non_spatial),
#         #     "spatial_output": policy_loss(actual_value=actual_spatial, predicted_value=out_spatial)  # ,
#         #     # 'actual__value': 'mae'
#         # }
#         losses = {
#             "value_output": value_loss(),
#             "non_spatial_output": policy_loss(actual_value=actual_value, predicted_value=out_value),
#             "spatial_output": policy_loss(actual_value=actual_value, predicted_value=out_value)  # ,
#             # 'actual__value': 'mae'
#         }
#
#         # lossWeights = {"value_output": 1.0, "non_spatial_output": 1.0, "spatial_output": 1.0
#         #                # ,
#         #                # "actual__value":1.0
#         #                }
#
#         lossWeights = {"value_output": 1.0, "non_spatial_output": 0.0, "spatial_output": 0.0
#                        # ,
#                        # "actual__value":1.0
#                        }
#         model.compile(loss=losses, loss_weights=lossWeights, optimizer=RMSprop(lr=0.1))
#         self.model = model
#
#     def predict(self, *args, **kwargs):
#         """wrapper for keras model predict function"""
#         return self.model.predict(*args, **kwargs)
#
#     def fit(self, *args, **kwargs):
#         """wrapper for keras model fit function"""
#         return self.model.fit(*args, **kwargs)
#
#     def load_weights(self, *args, **kwargs):
#         """wrapper for keras model load_weights function"""
#         return self.model.load_weights(*args, **kwargs)
#
#     def save_weights(self, *args, **kwargs):
#         """wrapper for keras model save_weights function"""
#         return self.model.save_weights(*args, **kwargs)


class SimpleConvNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleConvNet, self).__init__()
        '''
        这是对继承自父类的属性进行初始化。而且是用父类的初始化方法来初始化继承的属性。
        也就是说，子类继承了父类的所有属性和方法，父类属性自然会用父类方法来进行初始化。
        '''
        # 定义网络结构
        """
        nn.Conv2d(in_channels, 8, 5)：
            W：in_channels: 输入大小  64*64
            F：卷积核大小 5*5
            P：填充值的大小    0默认值
            S：步长大小  1默认值
            N：输出大小
                N=(W-F+2P)/S+1=(64-5 + 2*0)/1 + 1 = 60
            output_size: 输出通道数 8
            输出为：8(out_channels)*60*60           # 有小数时向下取整
        """
        self.conv1 = nn.Conv2d(in_channels=input_size[0] + input_size[1], out_channels=8, kernel_size=5)

        """
        nn.MaxPool2d(2, 2):
            W：输入大小  60*60
            F：kernel_size: 卷积核大小 2*2
            P：填充值的大小    0默认值
            S：stride: 步长大小  2
            N：输出大小          # 有小数时向上取整
                N=(W-F+2P)/S+1=(60-2 + 2*0)/2 + 1 = 30
            output_size: 输出通道数 8
            输出为：8*30*30
        """
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.fc1 = nn.Linear(16*12*12, 32)
        self.fc1 = nn.Linear(16 * 13 * 13, 32)
        self.fc2 = nn.Linear(43, 16)

        self.fc_o1 = nn.Linear(16, output_size[0])  # len(categorical_actions)
        self.fc_o2 = nn.Linear(16, output_size[1])  # 4096
        self.fc_o3 = nn.Linear(16, output_size[3])  # 4096

    def forward(self, input_x):  # 输入图片大小为：input = Variable(torch.randn(1, 1, 28, 28)) 即28*28的单通道图片
        input0 = torch.unsqueeze(Variable(torch.tensor(input_x[0])), 0) if len(input_x[0].shape) == 3 else Variable(
            torch.tensor(input_x[0]))  # input.shape = (1, 27, 64, 64)
        input0 = input0.cuda() if torch.cuda.is_available() else input0
        input1 = torch.unsqueeze(Variable(torch.tensor(input_x[1])), 0) if len(input_x[1].shape) == 3 else Variable(
            torch.tensor(input_x[1]))  # input.shape = (1, 11, 64, 64)
        input1 = input1.cuda() if torch.cuda.is_available() else input1

        input_concat = torch.cat([input0, input1], dim=1)
        x = self.pool1(F.relu(self.conv1(input_concat.float())))
        # x = torch.flatten(x)
        x = self.pool2(F.relu(self.conv2(x)))
        # x = self.fc1(x.flatten())
        x = self.fc1(x.view(x.size(0), -1))  # x.view(x.size(0), -1): 自动识别batch维度

        input2 = torch.unsqueeze(Variable(torch.tensor(input_x[2])), 0) if len(input_x[2].shape) == 1 else Variable(
            torch.tensor(input_x[2]))  # input.shape = (1, 11, 64, 64)
        input2 = input2.cuda() if torch.cuda.is_available() else input2  # input.shape = (11, )

        x = self.fc2(torch.cat([x, input2.float()], 1))

        x1 = self.fc_o1(x)
        x2 = self.fc_o2(x)

        p1 = torch.softmax(x1, 1)
        p2 = torch.softmax(x2, 1)
        v = torch.relu(x)
        return p1, p2, v

    def test(self):
        # 网络结构：conv2d--maxpool2d--conv2d--maxpool2d--fullyconnect--fullyconnect
        net = self.SimpleConvNet()
        print(net)
        input = Variable(torch.randn(1, 38, 64, 64))
        out = net(input)
        print(out.size())

    def structure(self):
        params = list(self.SimpleConvNet.parameters())
        for i in params:
            print('the structure of this layer is ' + str(list(i.size())))


class SimpleConvNet_prob(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleConvNet_prob, self).__init__()
        '''
        这是对继承自父类的属性进行初始化。而且是用父类的初始化方法来初始化继承的属性。
        也就是说，子类继承了父类的所有属性和方法，父类属性自然会用父类方法来进行初始化。
        '''
        # 定义网络结构
        """
        nn.Conv2d(in_channels, 8, 5)：
            W：in_channels: 输入大小  64*64
            F：卷积核大小 5*5
            P：填充值的大小    0默认值
            S：步长大小  1默认值
            N：输出大小
                N=(W-F+2P)/S+1=(64-5 + 2*0)/1 + 1 = 60
            output_size: 输出通道数 8
            输出为：8(out_channels)*60*60           # 有小数时向下取整
        """
        self.conv1 = nn.Conv2d(in_channels=input_size[0] + input_size[1], out_channels=8, kernel_size=5)

        """
        nn.MaxPool2d(2, 2):
            W：输入大小  60*60
            F：kernel_size: 卷积核大小 2*2
            P：填充值的大小    0默认值
            S：stride: 步长大小  2
            N：输出大小          # 有小数时向上取整
                N=(W-F+2P)/S+1=(60-2 + 2*0)/2 + 1 = 30
            output_size: 输出通道数 8
            输出为：8*30*30
        """
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.fc1 = nn.Linear(16*12*12, 32)
        self.fc1 = nn.Linear(16 * 13 * 13, 32)
        self.fc2 = nn.Linear(43, 16)

        self.fc_o1 = nn.Linear(16, output_size[0])  # len(categorical_actions)
        self.fc_o2 = nn.Linear(16, output_size[1])  # 4096

    def forward(self, input_x):  # 输入图片大小为：input = Variable(torch.randn(1, 1, 28, 28)) 即28*28的单通道图片
        input0 = torch.unsqueeze(Variable(torch.tensor(input_x[0])), 0) if len(input_x[0].shape) == 3 else Variable(
            torch.tensor(input_x[0]))  # input.shape = (1, 27, 64, 64)
        input0 = input0.cuda() if torch.cuda.is_available() else input0
        input1 = torch.unsqueeze(Variable(torch.tensor(input_x[1])), 0) if len(input_x[1].shape) == 3 else Variable(
            torch.tensor(input_x[1]))  # input.shape = (1, 11, 64, 64)
        input1 = input1.cuda() if torch.cuda.is_available() else input1

        input_concat = torch.cat([input0, input1], dim=1)
        x = self.pool1(F.relu(self.conv1(input_concat.float())))
        # x = torch.flatten(x)
        x = self.pool2(F.relu(self.conv2(x)))
        # x = self.fc1(x.flatten())
        x = self.fc1(x.view(x.size(0), -1))  # x.view(x.size(0), -1): 自动识别batch维度

        input2 = torch.unsqueeze(Variable(torch.tensor(input_x[2])), 0) if len(input_x[2].shape) == 1 else Variable(
            torch.tensor(input_x[2]))  # input.shape = (1, 11, 64, 64)
        input2 = input2.cuda() if torch.cuda.is_available() else input2  # input.shape = (11, )

        x = self.fc2(torch.cat([x, input2.float()], 1))

        x1 = self.fc_o1(x)
        x2 = self.fc_o2(x)

        p1 = torch.softmax(x1, 1)
        p2 = torch.softmax(x2, 1)
        return p1, p2


class SimpleConvNet_val(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleConvNet_val, self).__init__()
        # 定义网络结构
        """
        nn.Conv2d(in_channels, 8, 5)：
            W：in_channels: 输入大小  64*64
            F：卷积核大小 5*5
            P：填充值的大小    0默认值
            S：步长大小  1默认值
            N：输出大小
                N=(W-F+2P)/S+1=(64-5 + 2*0)/1 + 1 = 60
            out_channels: 输出通道数 8
            输出为：8(out_channels)*60*60           # 有小数时向下取整
        """
        self.conv1 = nn.Conv2d(in_channels=input_size[0] + input_size[1], out_channels=8, kernel_size=5)

        """
        nn.MaxPool2d(2, 2):
            W：输入大小  60*60
            F：kernel_size: 卷积核大小 2*2
            P：填充值的大小    0默认值
            S：stride: 步长大小  2
            N：输出大小          # 有小数时向上取整
                N=(W-F+2P)/S+1=(60-2 + 2*0)/2 + 1 = 30
            out_channels: 输出通道数 8
            输出为：8*30*30
        """
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.fc1 = nn.Linear(16*12*12, 32)
        self.fc1 = nn.Linear(16 * 13 * 13, 32)
        self.fc2 = nn.Linear(43, 16)

        self.fc_o1 = nn.Linear(16, output_size[0])  # len(categorical_actions)
        self.fc_o2 = nn.Linear(16, output_size[1])  # len(categorical_actions)

    def forward(self, input_x):
        input0 = torch.unsqueeze(Variable(torch.tensor(input_x[0])), 0) if len(input_x[0].shape) == 3 else Variable(
            torch.tensor(input_x[0]))  # input.shape = (1, 27, 64, 64)
        input0 = input0.cuda() if torch.cuda.is_available() else input0
        input1 = torch.unsqueeze(Variable(torch.tensor(input_x[1])), 0) if len(input_x[1].shape) == 3 else Variable(
            torch.tensor(input_x[1]))  # input.shape = (1, 11, 64, 64)
        input1 = input1.cuda() if torch.cuda.is_available() else input1
        # input2 = input2.cuda() if torch.cuda.is_available() else input2          # input.shape = (11, )

        input_concat = torch.cat([input0, input1], dim=1)
        x = self.pool1(F.relu(self.conv1(input_concat.float())))
        # x = torch.flatten(x)
        x = self.pool2(F.relu(self.conv2(x)))
        # x = self.fc1(x.flatten())
        x = self.fc1(x.view(x.size(0), -1))  # x.view(x.size(0), -1): 自动识别batch维度

        input2 = torch.unsqueeze(Variable(torch.tensor(input_x[2])), 0) if len(input_x[2].shape) == 1 else Variable(torch.tensor(input_x[2]))  # input.shape = (1, 11, 64, 64)
        input2 = input2.cuda() if torch.cuda.is_available() else input2  # input.shape = (11, )

        x = self.fc2(torch.cat([x, input2.float()], 1))

        xo1 = self.fc_o1(x)
        xo2 = self.fc_o2(x)

        v1 = torch.relu(xo1)
        v2 = torch.relu(xo2)

        # v = torch.softmax(xo, 1)
        return [v1, v2]


class NeuralAgent_val_rec(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralAgent_val_rec, self).__init__()
        # 定义网络结构
        """
        nn.Conv2d(in_channels, 8, 5)：
            W：in_channels: 输入大小  64*64
            F：卷积核大小 5*5
            P：填充值的大小    0默认值
            S：步长大小  1默认值
            N：输出大小
                N=(W-F+2P)/S+1=(64-5 + 2*0)/1 + 1 = 60
            out_channels: 输出通道数 8
            输出为：8(out_channels)*60*60           # 有小数时向下取整
        """
        self.conv1 = nn.Conv2d(in_channels=input_size[0] + input_size[1], out_channels=8, kernel_size=5)

        """
        nn.MaxPool2d(2, 2):
            W：输入大小  60*60
            F：kernel_size: 卷积核大小 2*2
            P：填充值的大小    0默认值
            S：stride: 步长大小  2
            N：输出大小          # 有小数时向上取整
                N=(W-F+2P)/S+1=(60-2 + 2*0)/2 + 1 = 30
            out_channels: 输出通道数 8
            输出为：8*30*30
        """
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.fc1 = nn.Linear(16*12*12, 32)
        self.fc1 = nn.Linear(16 * 13 * 13, 32)
        self.fc2 = nn.Linear(43, 16)

        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc_o1 = nn.Linear(16, output_size[0])  # len(categorical_actions)
        self.fc_o2 = nn.Linear(16, output_size[1])  # len(categorical_actions)

    def forward(self, input_x):
        input0 = torch.unsqueeze(Variable(torch.tensor(input_x[0])), 0) if len(input_x[0].shape) == 3 else Variable(
            torch.tensor(input_x[0]))  # input.shape = (1, 27, 64, 64)
        input0 = input0.cuda() if torch.cuda.is_available() else input0
        input1 = torch.unsqueeze(Variable(torch.tensor(input_x[1])), 0) if len(input_x[1].shape) == 3 else Variable(
            torch.tensor(input_x[1]))  # input.shape = (1, 11, 64, 64)
        input1 = input1.cuda() if torch.cuda.is_available() else input1
        # input2 = input2.cuda() if torch.cuda.is_available() else input2          # input.shape = (11, )

        input_concat = torch.cat([input0, input1], dim=1)
        x = self.pool1(F.relu(self.conv1(input_concat.float())))
        # x = torch.flatten(x)
        x = self.pool2(F.relu(self.conv2(x)))
        # x = self.fc1(x.flatten())
        x = self.fc1(x.view(x.size(0), -1))  # x.view(x.size(0), -1): 自动识别batch维度

        input2 = torch.unsqueeze(Variable(torch.tensor(input_x[2])), 0) if len(input_x[2].shape) == 1 else Variable(torch.tensor(input_x[2]))  # input.shape = (1, 11, 64, 64)
        input2 = input2.cuda() if torch.cuda.is_available() else input2  # input.shape = (11, )

        x = self.fc2(torch.cat([x, input2.float()], 1))

        xo1 = self.fc_o1(x)
        xo2 = self.fc_o2(x)

        v1 = torch.relu(xo1)
        v2 = torch.relu(xo2)

        # v = torch.softmax(xo, 1)
        return [v1, v2]
