from data.preprocess import *
from model.LSTM import LSTMModel
from train import Trainer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

if __name__ == '__main__':
    # data = preprocess_data('GOOG', 5)
    # plot_data(data, 'GOOGLE', './Images/GOOG.png')

    # train_size = int(data.shape[0] * 0.8)
    # input_size = 1
    # num_layers = 2
    # hidden_size = 64
    # model = LSTMModel

    # train = Trainer('GOOG', train_size, data, data.shape[1] - 2, model)

    # train.train_model(100, './model/GOOGLE_path.pth')

    data = preprocess_data('META', 5)
    plot_data(data, 'META', './Images/META.png')

    train_size = int(data.shape[0] * 0.8)
    input_size = 1
    num_layers = 2
    hidden_size = 64
    model = LSTMModel

    train = Trainer('META', train_size, data, data.shape[1] - 2, model)

    train.train_model(100, './model/META_path.pth')