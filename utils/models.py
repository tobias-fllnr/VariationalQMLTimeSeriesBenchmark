import pennylane as qml
import torch.nn as nn
import torch
import random


def print_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.data} {param.size()}")
def print_total_parameters(model):
    total_parameters = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_parameters += param.numel()
    print(total_parameters)


class VQC(nn.Module):
    """Variational Quantum Classifier model using PyTorch and PennyLane"""
    def __init__(self, num_qubits, seq_length, ansatz, data_label, random_id=42, backend="default.qubit", diff_method="best", evaluation=False):
        super(VQC, self).__init__()
        self.num_qubits = num_qubits
        self.seq_length = seq_length
        self.ansatz = ansatz
        self.data_label = data_label
        self.random_id = random_id
        self.backend = backend
        self.diff_method = diff_method
        self.evaluation = evaluation

        # set torch seed for reproducibility of initial weights
        torch.manual_seed(self.random_id)
        self.weight_init = {"weights": lambda x: torch.nn.init.uniform_(x, 0, 2 * torch.pi)}
        
        self.dev = qml.device(self.backend, wires=num_qubits)
        self.vqc_torch_layer = self.vqc()
        if self.data_label.startswith("lorenz"):
            self.output_layer = nn.Linear(num_qubits, 3)
        elif self.data_label.startswith("henon"):
            self.output_layer = nn.Linear(num_qubits, 2)
        else:
            self.output_layer = nn.Linear(num_qubits, 1)

        self.ansatz_input_layer_start = ("paper_rivera-ruiz_with_inputlayer_")
        if self.ansatz.startswith(self.ansatz_input_layer_start):
            if self.data_label.startswith("lorenz"):
                self.input_layer = nn.Linear(3*self.seq_length, self.num_qubits)
            elif self.data_label.startswith("henon"):
                self.input_layer = nn.Linear(2*self.seq_length, self.num_qubits)
            else:
                self.input_layer = nn.Linear(self.seq_length, self.num_qubits)

        self.vqc_torch_layer.weights.requires_grad = True      # set to False for fixed weights
    def vqc(self):
        if self.ansatz.startswith("paper_rivera-ruiz_with_inputlayer_"):
            num_layers = int(self.ansatz.split("_")[-1])
            @qml.qnode(self.dev, diff_method=self.diff_method, interface="torch")
            def circuit(inputs, weights):
                for i in range(self.num_qubits):
                    qml.RY(torch.pi*inputs[:, i], wires=i)
                for j in range(num_layers):
                    for i in range(self.num_qubits):
                        qml.RX(weights[i][j][0], wires=i)
                        qml.RY(weights[i][j][1], wires=i)
                        qml.RZ(weights[i][j][2], wires=i)
                    for i in range(self.num_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    qml.CNOT(wires=[self.num_qubits - 1, 0])
                return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
            weight_shapes = {"weights": (self.num_qubits, num_layers, 3)}
            return qml.qnn.TorchLayer(circuit, weight_shapes, init_method=self.weight_init)

        elif self.ansatz.startswith("ruexp_"):
            strings = self.ansatz.split('_')[1:]
            parameter_blocks_count = sum(1 for string in strings if not string.startswith('E'))
            def built_circuit_block(inputs, weights, i):
                enc_gate_map = {'X': qml.RX, 'Y': qml.RY, 'Z': qml.RZ}
                param_num_count = 0
                for string in strings:
                    if string.startswith('E'):                            
                        c1, c2 = string[1], string[2]
                        input_col = None
                            
                        # Determine input column based on data_label and c1
                        if self.data_label.startswith("lorenz"):
                            offset = {'X': 0, 'Y': 1, 'Z': 2}.get(c1)
                            if offset is not None:
                                input_col = 3 * i + offset
                        elif self.data_label.startswith("henon"):
                            offset = {'X': 0, 'Y': 1}.get(c1)
                            if offset is not None:
                                input_col = 2 * i + offset
                        else:
                            if c1 == 'X':
                                input_col = i
                        
                        # Get the rotation gate based on c2
                        rotation_gate = enc_gate_map.get(c2)
                        
                        if input_col is not None and rotation_gate is not None:
                            for j in range(self.num_qubits):
                                angle = torch.pi * inputs[:, input_col] * (3 ** (j - self.num_qubits))
                                rotation_gate(angle, wires=j)
                    elif string == 'X':
                        for j in range(self.num_qubits):
                            qml.RX(weights[i][j][param_num_count], wires=j)
                        param_num_count += 1
                    elif string == 'Y':
                        for j in range(self.num_qubits):
                            qml.RY(weights[i][j][param_num_count], wires=j)
                        param_num_count += 1
                    elif string == 'Z':
                        for j in range(self.num_qubits):
                            qml.RZ(weights[i][j][param_num_count], wires=j)
                        param_num_count += 1
                    elif string == 'CX':
                        for j in range(self.num_qubits - 1):
                            qml.CRX(weights[i][j][param_num_count], wires=[j, j + 1])
                        qml.CRX(weights[i][self.num_qubits-1][param_num_count], wires=[self.num_qubits - 1, 0])
                        param_num_count += 1
                    elif string == 'CY':
                        for j in range(self.num_qubits - 1):
                            qml.CRY(weights[i][j][param_num_count], wires=[j, j + 1])
                        qml.CRY(weights[i][self.num_qubits-1][param_num_count], wires=[self.num_qubits - 1, 0])
                        param_num_count += 1
                    elif string == 'CZ':
                        for j in range(self.num_qubits - 1):
                            qml.CRZ(weights[i][j][param_num_count], wires=[j, j + 1])
                        qml.CRZ(weights[i][self.num_qubits-1][param_num_count], wires=[self.num_qubits - 1, 0])
                        param_num_count += 1

            @qml.qnode(self.dev, diff_method=self.diff_method, interface="torch")
            def circuit(inputs, weights):
                for i in range(self.seq_length):
                    built_circuit_block(inputs, weights, i)
                return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
            weight_shapes = {"weights": (self.seq_length, self.num_qubits, parameter_blocks_count)}
            return qml.qnn.TorchLayer(circuit, weight_shapes, init_method=self.weight_init)

    def forward(self, x):
        x = torch.reshape(x, (x.size(0), -1))
        if self.ansatz.startswith(self.ansatz_input_layer_start):
            x = self.input_layer(x)
        vqc_output = self.vqc_torch_layer(x)
        scaled_output = self.output_layer(vqc_output) 
        return scaled_output

class QLSTM_Paper(nn.Module):
    """Quantum LSTM model using PyTorch and PennyLane"""
    def __init__(self, num_qubits, ansatz, data_label, random_id=42, backend="default.qubit", diff_method="best"):
        super(QLSTM_Paper, self).__init__()
        self.num_qubits = num_qubits
        self.ansatz = ansatz
        self.data_label = data_label
        self.random_id = random_id
        self.backend = backend
        self.diff_method = diff_method

        if self.data_label.startswith("lorenz"):
            self.output_layer = nn.Linear(num_qubits, 3)
        elif self.data_label.startswith("henon"):
            self.output_layer = nn.Linear(num_qubits, 2)
        else:
            self.output_layer = nn.Linear(num_qubits, 1)

        # set torch seed for reproducibility of initial weights
        torch.manual_seed(self.random_id)
        self.weight_init = {"weights": lambda x: torch.nn.init.uniform_(x, 0, 2 * torch.pi)}
        
        self.dev = qml.device(self.backend, wires=num_qubits)

        if self.ansatz.startswith("original_"):
            num_layers = int(self.ansatz.split("_")[-1])
            def VQC(inputs, weights):
                for i in range(self.num_qubits):
                    qml.Hadamard(wires=i)
                    qml.RY(torch.arctan(inputs[:, i]), wires=i)
                    qml.RZ(torch.arctan(inputs[:, i]**2), wires=i)
                for j in range(num_layers):
                    for i in range(self.num_qubits-1):
                        qml.CNOT(wires=[i, i + 1])
                    qml.CNOT(wires=[self.num_qubits - 1, 0])
                    for i in range(self.num_qubits-1):
                        qml.CNOT(wires=[i, i + 1])
                    qml.CNOT(wires=[self.num_qubits - 1, 0])
                    for i in range(self.num_qubits):
                        qml.Rot(weights[i][j][0], weights[i][j][1], weights[i][j][2], wires=i)
                return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.num_qubits)]
            weight_shapes = {"weights": (self.num_qubits, num_layers, 3)}
        
        self.VQC1 = qml.QNode(VQC, self.dev, diff_method=self.diff_method, interface="torch")
        self.VQC2 = qml.QNode(VQC, self.dev, diff_method=self.diff_method, interface="torch")
        self.VQC3 = qml.QNode(VQC, self.dev, diff_method=self.diff_method, interface="torch")
        self.VQC4 = qml.QNode(VQC, self.dev, diff_method=self.diff_method, interface="torch")
        self.VQC5 = qml.QNode(VQC, self.dev, diff_method=self.diff_method, interface="torch")
        self.VQC6 = qml.QNode(VQC, self.dev, diff_method=self.diff_method, interface="torch")
        
        self.vqc1 = qml.qnn.TorchLayer(self.VQC1, weight_shapes, init_method=self.weight_init)
        self.vqc2 = qml.qnn.TorchLayer(self.VQC2, weight_shapes, init_method=self.weight_init)
        self.vqc3 = qml.qnn.TorchLayer(self.VQC3, weight_shapes, init_method=self.weight_init)
        self.vqc4 = qml.qnn.TorchLayer(self.VQC4, weight_shapes, init_method=self.weight_init)
        self.vqc5 = qml.qnn.TorchLayer(self.VQC5, weight_shapes, init_method=self.weight_init)
        self.vqc6 = qml.qnn.TorchLayer(self.VQC6, weight_shapes, init_method=self.weight_init)


    def forward(self, x):
        batch_size, seq_length, features_size = x.size()
        h_t = torch.zeros(batch_size, self.num_qubits-features_size)  # hidden state 
        c_t = torch.zeros(batch_size, self.num_qubits)  # cell state
        for t in range(seq_length):
            x_t = x[:, t, :]
            v_t = torch.cat((x_t, h_t), dim=1)
            f_t = torch.sigmoid(self.vqc1(v_t))  # forget block
            i_t = torch.sigmoid(self.vqc2(v_t))  # input block
            g_t = torch.tanh(self.vqc3(v_t))  # update block
            o_t = torch.sigmoid(self.vqc4(v_t))  # output block
            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = self.vqc5(o_t * torch.tanh(c_t))[:, :self.num_qubits-features_size]
        
        y_t = self.vqc6(o_t * torch.tanh(c_t))
        y_t = self.output_layer(y_t)
        return y_t


class QLSTM_Linerar_Enhanced_Paper(nn.Module):
    """Quantum LSTM model using PyTorch and PennyLane"""
    def __init__(self, num_qubits, hidden_size, ansatz, data_label, random_id=42, backend="default.qubit", diff_method="best"):
        super(QLSTM_Linerar_Enhanced_Paper, self).__init__()
        self.num_qubits = num_qubits
        self.hidden_size = hidden_size
        self.ansatz = ansatz
        self.data_label = data_label
        self.random_id = random_id
        self.backend = backend
        self.diff_method = diff_method

        if self.data_label.startswith("lorenz"):
            self.output_layer = nn.Linear(self.hidden_size, 3)
            self.input_size = 3
        elif self.data_label.startswith("henon"):
            self.output_layer = nn.Linear(self.hidden_size, 2)
            self.input_size = 2
        else:
            self.output_layer = nn.Linear(self.hidden_size, 1)
            self.input_size = 1

        # set torch seed for reproducibility of initial weights
        torch.manual_seed(self.random_id)
        self.weight_init = {"weights": lambda x: torch.nn.init.uniform_(x, 0, 2 * torch.pi)}
        
        self.dev = qml.device(self.backend, wires=num_qubits)

        if self.ansatz.startswith("original_"):
            num_layers = int(self.ansatz.split("_")[-1])
            def VQC(inputs, weights):
                for i in range(self.num_qubits):
                    qml.Hadamard(wires=i)
                    qml.RY(torch.arctan(inputs[:, i]), wires=i)
                    qml.RZ(torch.arctan(inputs[:, i]**2), wires=i)
                for j in range(num_layers):
                    for i in range(self.num_qubits-1):
                        qml.CNOT(wires=[i, i + 1])
                    qml.CNOT(wires=[self.num_qubits - 1, 0])
                    for i in range(self.num_qubits-1):
                        qml.CNOT(wires=[i, i + 1])
                    qml.CNOT(wires=[self.num_qubits - 1, 0])
                    for i in range(self.num_qubits):
                        qml.Rot(weights[i][j][0], weights[i][j][1], weights[i][j][2], wires=i)
                return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.num_qubits)]
            weight_shapes = {"weights": (self.num_qubits, num_layers, 3)}

        self.VQC1 = qml.QNode(VQC, self.dev, diff_method=self.diff_method, interface="torch")
        self.VQC2 = qml.QNode(VQC, self.dev, diff_method=self.diff_method, interface="torch")
        self.VQC3 = qml.QNode(VQC, self.dev, diff_method=self.diff_method, interface="torch")
        self.VQC4 = qml.QNode(VQC, self.dev, diff_method=self.diff_method, interface="torch")

        
        self.vqc1 = qml.qnn.TorchLayer(self.VQC1, weight_shapes, init_method=self.weight_init)
        self.vqc2 = qml.qnn.TorchLayer(self.VQC2, weight_shapes, init_method=self.weight_init)
        self.vqc3 = qml.qnn.TorchLayer(self.VQC3, weight_shapes, init_method=self.weight_init)
        self.vqc4 = qml.qnn.TorchLayer(self.VQC4, weight_shapes, init_method=self.weight_init)

        self.linear_in = nn.Linear(self.hidden_size + self.input_size, self.num_qubits)
        self.linear_out_1 = nn.Linear(self.num_qubits, self.hidden_size)
        self.linear_out_2 = nn.Linear(self.num_qubits, self.hidden_size)
        self.linear_out_3 = nn.Linear(self.num_qubits, self.hidden_size)
        self.linear_out_4 = nn.Linear(self.num_qubits, self.hidden_size)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size)  # hidden state 
        c_t = torch.zeros(batch_size, self.hidden_size)  # cell state
        for t in range(seq_length):
            x_t = x[:, t, :]
            v_t = torch.cat((x_t, h_t), dim=1)
            v_t = self.linear_in(v_t)
            f_t = torch.sigmoid(self.linear_out_1(self.vqc1(v_t)))  # forget block
            i_t = torch.sigmoid(self.linear_out_2(self.vqc2(v_t)))  # input block
            g_t = torch.tanh(self.linear_out_3(self.vqc3(v_t)))  # update block
            o_t = torch.sigmoid(self.linear_out_4(self.vqc4(v_t)))  # output block
            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)
        
        y_t = self.output_layer(h_t)
        return y_t



class QRNN_Paper(nn.Module):
    """Quantum RNN model from the paper Quantum Recurrent Neural Networks for Sequential Learning"""
    def __init__(self, num_qubits, num_qubits_hidden, seq_length, ansatz, data_label, random_id=42, backend="default.qubit", diff_method="best"):
        super(QRNN_Paper, self).__init__()
        self.num_qubits = num_qubits
        self.seq_length = seq_length
        self.ansatz = ansatz
        self.data_label = data_label
        self.random_id = random_id
        self.backend = backend
        self.diff_method = diff_method

        self.num_qubits_hidden = num_qubits_hidden
        self.num_qubits_data = self.num_qubits - self.num_qubits_hidden

        if self.data_label.startswith("lorenz"):
            self.output_layer = nn.Linear(self.num_qubits_data, 3)
        elif self.data_label.startswith("henon"):
            self.output_layer = nn.Linear(self.num_qubits_data, 2)
        else:
            self.output_layer = nn.Linear(self.num_qubits_data, 1)

        # set torch seed for reproducibility of initial weights
        torch.manual_seed(self.random_id)
        self.weight_init = {"weights": lambda x: torch.nn.init.uniform_(x, 0, 2 * torch.pi)}
        
        self.dev = qml.device(self.backend, wires=num_qubits)
        self.vqc_torch_layer = self.vqc()

    def vqc(self):
        if self.ansatz == "paper_no_reset":
            if self.data_label.startswith("lorenz"):
                @qml.qnode(self.dev, diff_method=self.diff_method, interface="torch")
                def circuit(inputs, weights):
                    for i in range(self.seq_length):
                        for j in range(self.num_qubits_data):
                            qml.RY(torch.arccos(inputs[:, 3*i]), wires=j)
                            qml.RX(torch.arccos(inputs[:, 1+3*i]), wires=j)
                            qml.RY(torch.arccos(inputs[:, 2+3*i]), wires=j)
                        for j in range(self.num_qubits):
                            qml.RX(weights[j][0], wires=j)
                            qml.RZ(weights[j][1], wires=j)
                            qml.RX(weights[j][2], wires=j)
                        for j in range(self.num_qubits - 1):
                            qml.CNOT(wires=[j, j + 1])
                            qml.RZ(weights[j+1][3], wires=j+1)
                            qml.CNOT(wires=[j, j + 1])
                        qml.CNOT(wires=[self.num_qubits - 1, 0])
                        qml.RZ(weights[0][3], wires=0)
                        qml.CNOT(wires=[self.num_qubits - 1, 0])
                    return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits_data)]
                
                weight_shapes = {"weights": (self.num_qubits, 4)}
                return qml.qnn.TorchLayer(circuit, weight_shapes, init_method=self.weight_init)
            elif self.data_label.startswith("henon"):
                @qml.qnode(self.dev, diff_method=self.diff_method, interface="torch")
                def circuit(inputs, weights):
                    for i in range(self.seq_length):
                        for j in range(self.num_qubits_data):
                            qml.RX(torch.arccos(inputs[:, 2*i]), wires=j)
                            qml.RY(torch.arccos(inputs[:, 1+2*i]), wires=j)
                        for j in range(self.num_qubits):
                            qml.RX(weights[j][0], wires=j)
                            qml.RZ(weights[j][1], wires=j)
                            qml.RX(weights[j][2], wires=j)
                        for j in range(self.num_qubits - 1):
                            qml.CNOT(wires=[j, j + 1])
                            qml.RZ(weights[j+1][3], wires=j+1)
                            qml.CNOT(wires=[j, j + 1])
                        qml.CNOT(wires=[self.num_qubits - 1, 0])
                        qml.RZ(weights[0][3], wires=0)
                        qml.CNOT(wires=[self.num_qubits - 1, 0])
                    return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits_data)]
                
                weight_shapes = {"weights": (self.num_qubits, 4)}
                return qml.qnn.TorchLayer(circuit, weight_shapes, init_method=self.weight_init)
            else:
                @qml.qnode(self.dev, diff_method=self.diff_method, interface="torch")
                def circuit(inputs, weights):
                    for i in range(self.seq_length):
                        for j in range(self.num_qubits_data):
                            qml.RY(torch.arccos(inputs[:, i]), wires=j)
                        for j in range(self.num_qubits):
                            qml.RX(weights[j][0], wires=j)
                            qml.RZ(weights[j][1], wires=j)
                            qml.RX(weights[j][2], wires=j)
                        for j in range(self.num_qubits - 1):
                            qml.CNOT(wires=[j, j + 1])
                            qml.RZ(weights[j+1][3], wires=j+1)
                            qml.CNOT(wires=[j, j + 1])
                        qml.CNOT(wires=[self.num_qubits - 1, 0])
                        qml.RZ(weights[0][3], wires=0)
                        qml.CNOT(wires=[self.num_qubits - 1, 0])

                    return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits_data)]
                
                weight_shapes = {"weights": (self.num_qubits, 4)}
                return qml.qnn.TorchLayer(circuit, weight_shapes, init_method=self.weight_init)
        if self.ansatz == "paper_reset":
            if self.data_label.startswith("lorenz"):
                @qml.qnode(self.dev, diff_method=self.diff_method, interface="torch")
                def circuit(inputs, weights):
                    for i in range(self.seq_length):
                        for j in range(self.num_qubits_data):
                            qml.RY(torch.arccos(inputs[:, 3*i]), wires=j)
                            qml.RX(torch.arccos(inputs[:, 1+3*i]), wires=j)
                            qml.RY(torch.arccos(inputs[:, 2+3*i]), wires=j)
                        for j in range(self.num_qubits):
                            qml.RX(weights[j][0], wires=j)
                            qml.RZ(weights[j][1], wires=j)
                            qml.RX(weights[j][2], wires=j)
                        for j in range(self.num_qubits - 1):
                            qml.CNOT(wires=[j, j + 1])
                            qml.RZ(weights[j+1][3], wires=j+1)
                            qml.CNOT(wires=[j, j + 1])
                        qml.CNOT(wires=[self.num_qubits - 1, 0])
                        qml.RZ(weights[0][3], wires=0)
                        qml.CNOT(wires=[self.num_qubits - 1, 0])
                        if i == self.seq_length - 1:
                            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits_data)]
                        for j in range(self.num_qubits_data):
                            qml.measure(wires=j, reset=True)
                
                weight_shapes = {"weights": (self.num_qubits, 4)}
                return qml.qnn.TorchLayer(circuit, weight_shapes, init_method=self.weight_init)
            elif self.data_label.startswith("henon"):
                @qml.qnode(self.dev, diff_method=self.diff_method, interface="torch")
                def circuit(inputs, weights):
                    for i in range(self.seq_length):
                        for j in range(self.num_qubits_data):
                            qml.RX(torch.arccos(inputs[:, 2*i]), wires=j)
                            qml.RY(torch.arccos(inputs[:, 1+2*i]), wires=j)
                        for j in range(self.num_qubits):
                            qml.RX(weights[j][0], wires=j)
                            qml.RZ(weights[j][1], wires=j)
                            qml.RX(weights[j][2], wires=j)
                        for j in range(self.num_qubits - 1):
                            qml.CNOT(wires=[j, j + 1])
                            qml.RZ(weights[j+1][3], wires=j+1)
                            qml.CNOT(wires=[j, j + 1])
                        qml.CNOT(wires=[self.num_qubits - 1, 0])
                        qml.RZ(weights[0][3], wires=0)
                        qml.CNOT(wires=[self.num_qubits - 1, 0])
                        if i == self.seq_length - 1:
                            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits_data)]
                        for j in range(self.num_qubits_data):
                            qml.measure(wires=j, reset=True)
                
                weight_shapes = {"weights": (self.num_qubits, 4)}
                return qml.qnn.TorchLayer(circuit, weight_shapes, init_method=self.weight_init)
            else:
                @qml.qnode(self.dev, diff_method=self.diff_method, interface="torch")
                def circuit(inputs, weights):
                    for i in range(self.seq_length):
                        for j in range(self.num_qubits_data):
                            qml.RY(torch.arccos(inputs[:, i]), wires=j)
                        for j in range(self.num_qubits):
                            qml.RX(weights[j][0], wires=j)
                            qml.RZ(weights[j][1], wires=j)
                            qml.RX(weights[j][2], wires=j)
                        for j in range(self.num_qubits - 1):
                            qml.CNOT(wires=[j, j + 1])
                            qml.RZ(weights[j+1][3], wires=j+1)
                            qml.CNOT(wires=[j, j + 1])
                        qml.CNOT(wires=[self.num_qubits - 1, 0])
                        qml.RZ(weights[0][3], wires=0)
                        qml.CNOT(wires=[self.num_qubits - 1, 0])
                        if i == self.seq_length - 1:
                            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits_data)]
                        for j in range(self.num_qubits_data):
                            qml.measure(wires=j, reset=True)
                
                weight_shapes = {"weights": (self.num_qubits, 4)}
                return qml.qnn.TorchLayer(circuit, weight_shapes, init_method=self.weight_init)
    def forward(self, x):
        if self.data_label.startswith(("lorenz", "henon")):
            x = torch.reshape(x, (x.size(0), -1))
        else:
            x = x.squeeze(-1)
        vqc_output = self.vqc_torch_layer(x)
        scaled_output = self.output_layer(vqc_output) 
        return scaled_output
    
class LSTM(nn.Module):
    def __init__(self, hidden_size, ansatz, data_label, random_id=42):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.ansatz = ansatz
        self.data_label = data_label
        self.random_id = random_id
        self.num_layers = int(self.ansatz.split("_")[-1])
        torch.manual_seed(self.random_id)
        if self.data_label.startswith("lorenz"):
            self.lstm = nn.LSTM(3, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
            self.fc = nn.Linear(self.hidden_size, 3)
        elif self.data_label.startswith("henon"):
            self.lstm = nn.LSTM(2, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
            self.fc = nn.Linear(self.hidden_size, 2)
        else:
            self.lstm = nn.LSTM(1, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
            self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class RNN(nn.Module):
    def __init__(self, hidden_size, ansatz, data_label, random_id=42):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.ansatz = ansatz
        self.data_label = data_label
        self.random_id = random_id
        self.num_layers = int(self.ansatz.split("_")[-1])
        torch.manual_seed(self.random_id)
        if self.data_label.startswith("lorenz"):
            self.rnn = nn.RNN(3, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
            self.fc = nn.Linear(self.hidden_size, 3)
        elif self.data_label.startswith("henon"):
            self.rnn = nn.RNN(2, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
            self.fc = nn.Linear(self.hidden_size, 2)
        else:
            self.rnn = nn.RNN(1, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
            self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out
    

class MLP(nn.Module):
    def __init__(self, seq_length, ansatz, data_label, random_id=42):
        super(MLP, self).__init__()
        self.seq_length = seq_length
        self.ansatz = ansatz
        self.data_label = data_label
        self.random_id = random_id

        torch.manual_seed(self.random_id)
        self.activation, self.hidden_sizes = self.extract_hidden_sizes()
        if self.data_label.startswith("lorenz"):
            self.input_size = 3*self.seq_length
            self.output_size = 3
        elif self.data_label.startswith("henon"):
            self.input_size = 2*self.seq_length
            self.output_size = 2
        else: 
            self.input_size = self.seq_length
            self.output_size = 1
        if self.activation == "tanh":
            self.activation_func = nn.Tanh()
        elif self.activation == "sigmoid":
            self.activation_func = nn.Sigmoid()
        elif self.activation == "relu":
            self.activation_func = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
        
        layers = []
        layers.append(nn.Linear(self.input_size, self.hidden_sizes[0]))
        layers.append(self.activation_func) 
        
        for i in range(1, len(self.hidden_sizes)):
            layers.append(nn.Linear(self.hidden_sizes[i-1], self.hidden_sizes[i]))
            layers.append(self.activation_func) 
        
        layers.append(nn.Linear(self.hidden_sizes[-1], self.output_size))
        
        self.mlp_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.reshape(x, (x.size(0), -1))
        x = self.mlp_layers(x)
        return x
    
    def extract_hidden_sizes(self):
        parts = self.ansatz.split('_')
        activation = parts[0]
        numbers = parts[1:]
        numbers = [int(num) for num in numbers]
        return activation, numbers

