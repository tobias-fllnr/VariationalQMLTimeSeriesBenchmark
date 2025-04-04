from handling_data import DataHandling
import models
from trainer import Trainer
from analyzer import Analyzer
import time
import argparse

def train_and_analyse():
    if model_name == "vqc":
        model = models.VQC(num_qubits=num_qubits, seq_length=seq_length, ansatz=ansatz, data_label=data_label, random_id=random_id)
    elif model_name == "qlstm_paper":
        model = models.QLSTM_Paper(num_qubits=num_qubits, ansatz=ansatz, data_label=data_label, random_id=random_id)
    elif model_name == "qlstm_linear_enhanced_paper":
        model = models.QLSTM_Linerar_Enhanced_Paper(num_qubits=num_qubits, hidden_size=hidden_size, ansatz=ansatz, data_label=data_label, random_id=random_id)
    elif model_name == "qrnn_paper":
        model = models.QRNN_Paper(num_qubits=num_qubits, num_qubits_hidden=hidden_size, seq_length=seq_length, ansatz=ansatz, data_label=data_label, random_id=random_id)
    elif model_name == "lstm":
        model = models.LSTM(hidden_size=hidden_size, ansatz=ansatz, data_label=data_label, random_id=random_id)
    elif model_name == "rnn":
        model = models.RNN(hidden_size=hidden_size, ansatz=ansatz, data_label=data_label, random_id=random_id)
    elif model_name == "mlp":
        model = models.MLP(seq_length=seq_length, ansatz=ansatz, data_label=data_label, random_id=random_id)
    data_handler = DataHandling(data_label=data_label, seq_length=seq_length, prediction_step=prediction_step)
    trainer = Trainer(model=model, random_id=random_id, learning_rate=learning_rate, batch_size=batch_size)
    inputs_training, labels_training, inputs_validation, labels_validation, inputs_testing, labels_testing = data_handler.get_training_and_test_data()
    
    cost_training, cost_validation, cost_testing, trained_model, best_validation_model, total_time = trainer.train(inputs_training, labels_training, 
                                                                                           inputs_validation, labels_validation, 
                                                                                           inputs_testing, labels_testing)
    analyzer = Analyzer(version=version, model=model, trainer=trainer, data_handler=data_handler)
    analyzer.create_directory()
    analyzer.save_training_output(trained_model, best_validation_model, cost_training=cost_training, cost_validation=cost_validation, cost_testing=cost_testing)
    analyzer.plot_cost(cost_training=cost_training, cost_validation=cost_validation, cost_testing=cost_testing)
    analyzer.evaluate_trained_model(inputs_testing, labels_testing, inputs_validation, labels_validation)
    analyzer.save_training_info(cost_training=cost_training, cost_validation=cost_validation, cost_testing=cost_testing, total_time=total_time)

if __name__ == '__main__':
    start = time.time()
    def none_or_type(type_):
        def convert(value):
            if value == "None":
                return None
            return type_(value)
        return convert

    parser = argparse.ArgumentParser(description='Train with specific arguments')

    parser.add_argument('-version', '--version', type=none_or_type(int), help='version of the code, in case of change of models, training or data')
    parser.add_argument('-model', '--model_name', type=none_or_type(str), help='model to train e.g. vqc, qlstm_paper, qlstm_linear_enhanced_paper, qrnn_paper, lstm, rnn, mlp')
    parser.add_argument('-data', '--data_label', type=none_or_type(str), help='data e.g. lorenz_1000, mackey_glass_1000_default, henon_map_1000_default')
    parser.add_argument('-id', '--random_id', type=none_or_type(int), help='random id to initialize the weights')
    parser.add_argument('-lr', '--learning_rate', type=none_or_type(float), help='step size of the Adam optimizer')
    parser.add_argument('-num_qubits', '--number_qubits', type=none_or_type(int), help='number of qubits used')
    parser.add_argument('-hidden_size', '--hidden_size', type=none_or_type(int), help='hidden size of the rnn, lstm, qlstm_linear_enhanced_paper, qrnn_paper')
    parser.add_argument('-ansatz', '--type_of_ansatz', type=none_or_type(str), help='type of quantum circuit ansatz')
    parser.add_argument('-seq_length', '--sequence_length', type=none_or_type(int), help='sequence length of data')
    parser.add_argument('-pred_step', '--prediction_step', type=none_or_type(int), help='step into the future on which the models are trained on')
    parser.add_argument('-batch_size', '--batch_size', type=none_or_type(int), help='batch size for training')

    args = parser.parse_args()
    if args.model_name is not None:
        version = args.version
        model_name = args.model_name
        random_id = args.random_id
        learning_rate = args.learning_rate
        ansatz = args.type_of_ansatz
        data_label = args.data_label
        seq_length = args.sequence_length
        prediction_step = args.prediction_step
        num_qubits = args.number_qubits
        hidden_size = args.hidden_size
        batch_size = args.batch_size
        train_and_analyse()
    else:
        version = 64
        model_name = "lstm"
        random_id = 103
        learning_rate = 0.001
        ansatz = "layers_1"
        data_label = "henon_1000"
        seq_length = 4
        prediction_step = 1
        num_qubits = 4
        hidden_size = 32
        batch_size = 64
        train_and_analyse()
    
    end = time.time()
    print("total_time for training and analyzing= ", end-start, flush=True)