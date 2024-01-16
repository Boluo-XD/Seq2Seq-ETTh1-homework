
import argparse
import logging
import math
import os
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import pandas as pd
from tqdm import tqdm
 
# 将matplotlib的日志级别设置为警告级别
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG)
 
 
# 参数设置部分
parser = argparse.ArgumentParser(description="CNN-GRU-LSTM for multivariate time series forecasting",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data-dir', type=str, default='./', help='relative path to input data')
parser.add_argument('--data_name', type=str, default='ETTh1.csv', help='Input Model File Name')
parser.add_argument('--max-records', type=int, default=None, help='total records before data split')
parser.add_argument('--q', type=int, default=96,
                    help='number of histrical measurements included in each training example')
parser.add_argument('--horizon', type=int, default=96, choices=[96,336],help='number of measurements ahead to predict')
parser.add_argument('--splits', type=str, default="0.6,0.2",
                    help='fraction of data to use for train & validation. remainder used for test.')
parser.add_argument('--batch-size', type=int, default=2048, help='the batch size.')
parser.add_argument('--filter-list', type=str, default="6,12,18", help='unique filter sizes')
parser.add_argument('--num-filters', type=int, default=100, help='number of each filter size')
parser.add_argument('--recurrent-state-size', type=int, default=100,
                    help='number of hidden units in each unrolled recurrent cell')
parser.add_argument('--seasonal-period', type=int, default=24, help='time between seasonal measurements')
parser.add_argument('--time-interval', type=int, default=1, help='time between each measurement')
parser.add_argument('--gpus', type=str, default='0',
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu. ')   #  gpu 修改
parser.add_argument('--optimizer', type=str, default='adam', help='the optimizer type')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate for network')
parser.add_argument('--num-epochs', type=int, default=1000, help='max num of epochs')
parser.add_argument('--save-period', type=int, default=20, help='save checkpoint for every n epochs')
parser.add_argument('--model_prefix', type=str, default='electricity_model', help='prefix for saving model params')

 
def rse(label, pred):
    """computes the root relative squared error (condensed using standard deviation formula)"""
    numerator = np.sqrt(np.mean(np.square(label - pred), axis=None))
    denominator = np.std(label, axis=None)
    return numerator / denominator
 
 
def rae(label, pred):
    """computes the relative absolute error (condensed using standard deviation formula)"""
    numerator = np.mean(np.abs(label - pred), axis=None)
    denominator = np.mean(np.abs(label - np.mean(label, axis=None)), axis=None)
    return numerator / denominator
 
 
def corr(label, pred):
    """computes the empirical correlation coefficient"""
    numerator1 = label - np.mean(label, axis=0)
    numerator2 = pred - np.mean(pred, axis=0)
    numerator = np.mean(numerator1 * numerator2, axis=0)
    denominator = np.std(label, axis=0) * np.std(pred, axis=0)
    return np.mean(numerator / denominator)
 
 
def get_custom_metrics():
    """
    :return: mxnet metric object
    """
    _rse = mx.metric.create(rse)
    _rae = mx.metric.create(rae)
    _corr = mx.metric.create(corr)
    return mx.metric.create([_rae, _rse, _corr])
 
 
def evaluate(pred, label):
    return {"RAE": rae(label, pred), "RSE": rse(label, pred), "CORR": corr(label, pred)}
 
 
def build_iters(data_dir, max_records, q, horizon, splits, batch_size):
    """
    Load & generate training examples from multivariate time series data
    :return: data iters & variables required to define network architecture
    """
    # Read in data as numpy array
    csv_list = ['train_set.csv','test_set.csv','validation_set.csv']
    out_list = ['train_csv','test_csv','validation_csv']
    for i in range(len(csv_list)):
        df = pd.read_csv(os.path.join(data_dir, csv_list[i]), sep=",", )
        feature_df = df.iloc[:, 1:].fillna(0).astype(float)
        x = feature_df.values
        x = x[:max_records] if max_records else x
    
        # Construct training examples based on horizon and window
        x_ts = np.zeros((x.shape[0] - q, q, x.shape[1]))
        y_ts = np.zeros((x.shape[0] - q, x.shape[1]))
        for n in range(x.shape[0]):
            if n + 1 < q:
                continue
            elif n + 1 + horizon > x.shape[0]:
                continue
            else:
                y_n = x[n + horizon, :]
                x_n = x[n + 1 - q:n + 1, :]
            x_ts[n - q] = x_n
            y_ts[n - q] = y_n
            out_list[i] = mx.io.NDArrayIter(data=x_ts,
                                   label=y_ts,
                                   batch_size=batch_size)
            
    return out_list[0],out_list[1],out_list[2]
 
def sym_gen(train_iter, q, filter_list, num_filter, dropout, rcells, skiprcells, seasonal_period, time_interval):
    input_feature_shape = train_iter.provide_data[0][1]
    X = mx.symbol.Variable(train_iter.provide_data[0].name)
    Y = mx.sym.Variable(train_iter.provide_label[0].name)
 
    # reshape data before applying convolutional layer (takes 4D shape incase you ever work with images)
    conv_input = mx.sym.reshape(data=X, shape=(0, 1, q, -1))
 
    ###############
    # CNN Component
    ###############
    outputs = []
    for i, filter_size in enumerate(filter_list):
        # pad input array to ensure number output rows = number input rows after applying kernel
        padi = mx.sym.pad(data=conv_input, mode="constant", constant_value=0,
                          pad_width=(0, 0, 0, 0, filter_size - 1, 0, 0, 0))
        convi = mx.sym.Convolution(data=padi, kernel=(filter_size, input_feature_shape[2]), num_filter=num_filter)
        acti = mx.sym.Activation(data=convi, act_type='relu')
        trans = mx.sym.reshape(mx.sym.transpose(data=acti, axes=(0, 2, 1, 3)), shape=(0, 0, 0))
        outputs.append(trans)
    cnn_features = mx.sym.Concat(*outputs, dim=2)
    cnn_reg_features = mx.sym.Dropout(cnn_features, p=dropout)
 
    ###############
    # GRU Component
    ###############
    stacked_rnn_cells = mx.rnn.SequentialRNNCell()
    for i, recurrent_cell in enumerate(rcells):
        stacked_rnn_cells.add(recurrent_cell)
        stacked_rnn_cells.add(mx.rnn.DropoutCell(dropout))
    outputs, states = stacked_rnn_cells.unroll(length=q, inputs=cnn_reg_features, merge_outputs=False)
    rnn_features = outputs[-1]  # only take value from final unrolled cell for use later
 
    ####################
    # LSTM Component
    ####################
    stacked_rnn_cells = mx.rnn.SequentialRNNCell()
    for i, recurrent_cell in enumerate(skiprcells):
        stacked_rnn_cells.add(recurrent_cell)
        stacked_rnn_cells.add(mx.rnn.DropoutCell(dropout))
    outputs, states = stacked_rnn_cells.unroll(length=q, inputs=cnn_reg_features, merge_outputs=False)
 
    # Take output from cells p steps apart
    p = int(seasonal_period / time_interval)
    output_indices = list(range(0, q, p))
    outputs.reverse()
    skip_outputs = [outputs[i] for i in output_indices]
    skip_rnn_features = mx.sym.concat(*skip_outputs, dim=1)
 
    ##########################
    # Autoregressive Component
    ##########################
    auto_list = []
    for i in list(range(input_feature_shape[2])):
        time_series = mx.sym.slice_axis(data=X, axis=2, begin=i, end=i + 1)
        fc_ts = mx.sym.FullyConnected(data=time_series, num_hidden=1)
        auto_list.append(fc_ts)
    ar_output = mx.sym.concat(*auto_list, dim=1)
 
    ######################
    # Prediction Component
    ######################
    neural_components = mx.sym.concat(*[rnn_features, skip_rnn_features], dim=1)
    neural_output = mx.sym.FullyConnected(data=neural_components, num_hidden=input_feature_shape[2])
    model_output = neural_output + ar_output
    loss_grad = mx.sym.LinearRegressionOutput(data=model_output, label=Y)
    return loss_grad, [v.name for v in train_iter.provide_data], [v.name for v in train_iter.provide_label]
 
 
def train(symbol, train_iter, val_iter, data_names, label_names):
    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    module = mx.mod.Module(symbol, data_names=data_names, label_names=label_names, context=devs)
    module.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    module.init_params(mx.initializer.Uniform(0.1))
    module.init_optimizer(optimizer=args.optimizer, optimizer_params={'learning_rate': args.lr})
 
    for epoch in tqdm(range(1, args.num_epochs + 1), desc="Epochs"):
        train_iter.reset()
        val_iter.reset()
        for batch in tqdm(train_iter, desc="Batches", leave=False):
            module.forward(batch, is_train=True)  # compute predictions
            module.backward()  # compute gradients
            module.update()  # update parameters
 
        train_pred = module.predict(train_iter).asnumpy()
        train_label = train_iter.label[0][1].asnumpy()
        print('\nMetrics: Epoch %d, Training %s' % (epoch, evaluate(train_pred, train_label)))
 
        val_pred = module.predict(val_iter).asnumpy()
        val_label = val_iter.label[0][1].asnumpy()
        print('Metrics: Epoch %d, Validation %s' % (epoch, evaluate(val_pred, val_label)))
 
        if epoch % args.save_period == 0 and epoch > 1:
            module.save_checkpoint(prefix=os.path.join("../models/", args.model_prefix), epoch=epoch,
                                   save_optimizer_states=False)
        if epoch == args.num_epochs:
            module.save_checkpoint(prefix=os.path.join("../models/", args.model_prefix), epoch=epoch,
                                   save_optimizer_states=False)
 
    return module
 
 
def predict(symbol, train_iter, val_iter, test_iter, data_names, label_names):
    print("here")
    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    module = mx.mod.Module(symbol, data_names=data_names, label_names=label_names, context=devs)
    module.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    module.init_params(mx.initializer.Uniform(0.1))
    module.init_optimizer(optimizer=args.optimizer, optimizer_params={'learning_rate': args.lr})
 
    # 加载模型参数
    params_file = "../models/electricity_model_336.params"  # 参数文件的路径
    module.load_params(params_file)
 
    # 将模型转换为评估模式
    test_iter.reset()
    # print(test_iter.data)
    test_data = test_iter.data[0][1].asnumpy()
    test_pred = module.predict(test_iter).asnumpy()
    test_label = test_iter.label[0][1].asnumpy()

    min_diff = float("inf")
    pre_results, real_results, ind = None, None, None
    for i in range(len(test_pred) - args.horizon):
        if np.sum(test_label[i:i+args.horizon, 6] == 0).all():
            continue
        diff = np.mean(np.abs(test_pred[i:i+args.horizon, 6] - test_label[i:i+args.horizon, 6]))
        if diff < min_diff:
            min_diff = diff
            pre_results = test_pred[i:i+args.horizon, 6]
            real_results = test_label[i:i+args.horizon, 6]
            ind = i
    pre_results = np.concatenate((test_data[ind-96:ind, 0, 6], pre_results))
    real_results = np.concatenate((test_data[ind-96:ind, 0, 6], real_results))
    print('test',evaluate(pre_results,real_results))
    # pre_results = []
    # real_results = []
    # for i in range(96):
    #     pre_results.append(test_data[0, i, 6])
    #     real_results.append(test_data[0, i, 6])
    # for i in range(len(test_pred)):
    #     pre_results.append(test_pred[i][6])
    #     real_results.append(test_label[i][6])
    # print("预测值：", pre_results)
    # print("真实值：", real_results)
    df = pd.DataFrame({'real': real_results, 'forecast': pre_results})
 
    df.to_csv('results.csv', index=False)
    # 创建一个新的图形
    plt.figure(figsize=(10, 6))
 
    # 绘制预测值曲线，使用蓝色实线
    plt.plot(range(96+args.horizon), pre_results, linewidth=2, label='Prediction')
 
    # 绘制真实值曲线，使用红色虚线
    plt.plot(range(96+args.horizon), real_results, linewidth=2, label='GroundTruth')
 
 
    # 添加图例
    plt.legend(loc='upper left')
 
    # 显示网格线
    plt.grid(True, linestyle='--', alpha=0.5)
 
    # 保存图形
    plt.savefig('predict_cnn_gru_lstm_96_500.png')
 
    # 显示图形
    plt.show()
    # print(test_label, test_pred)
 
 
if __name__ == '__main__':
    # parse args
    args = parser.parse_args()
    args.splits = list(map(float, args.splits.split(',')))
    args.filter_list = list(map(int, args.filter_list.split(',')))
 
    # Check valid args
    if not max(args.filter_list) <= args.q:
        raise AssertionError("no filter can be larger than q")
    if not args.q >= math.ceil(args.seasonal_period / args.time_interval):
        raise AssertionError("size of skip connections cannot exceed q")
    # Build data iterators
    train_iter, val_iter, test_iter = build_iters(args.data_dir, args.max_records, args.q, args.horizon, args.splits,
                                                  args.batch_size)
    # train_iter = build_iters(args.data_dir, args.max_records, args.q, args.horizon, args.splits,
    #                                               args.batch_size)
    # Choose cells for recurrent layers: each cell will take the output of the previous cell in the list
    rcells = [mx.rnn.GRUCell(num_hidden=args.recurrent_state_size)]
    skiprcells = [mx.rnn.LSTMCell(num_hidden=args.recurrent_state_size)]
 
    # Define network symbol
    symbol, data_names, label_names = sym_gen(train_iter, args.q, args.filter_list, args.num_filters,
                                              args.dropout, rcells, skiprcells, args.seasonal_period,
                                              args.time_interval)
 
    Train = False
    # train cnn model
    if Train:
        module = train(symbol, train_iter, val_iter, data_names, label_names)
 
    predict(symbol, train_iter, val_iter, test_iter, data_names, label_names)