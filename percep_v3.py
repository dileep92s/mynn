import numpy as np

class NN():
    def __init__(self, shape, actfn):
        self.n_inputs = shape[0]
        self.n_hlayers = shape[1]
        self.n_outputs = shape[2]

        self.i = None
        self.w_h1 = np.random.uniform(-1, 1, (self.n_hlayers, self.n_inputs))
        self.b_h1 = np.random.uniform(-1, 1, (self.n_hlayers))
        self.h1 = None
        self.w_o = np.random.uniform(-1, 1, (self.n_outputs, self.n_hlayers))
        self.b_o = np.random.uniform(-1, 1, (self.n_outputs))
        self.o = None

        self.lr = 0.05

        self.actfn = None
        if actfn == "relu":
            self.actfn = self.act_relu
        elif actfn == "sigmoid":
            self.actfn = self.act_sigmoid
      
    def feedforward(self, input_m):
        self.i = input_m
        self.h1 = self.w_h1.dot(self.i) + self.b_h1
        self.h1 = self.actfn(self.h1)
        self.o = self.w_o.dot(self.h1) + self.b_o
        self.o = self.actfn(self.o)
        return self.o

    def backpropagate(self, error_m):
        e_o = self.w_o.T.dot(error_m)
        self.w_o += e_o * self.h1 * self.lr
        self.b_o += error_m * self.lr

        e_h1 = self.w_h1.T.dot(e_o)
        self.w_h1 += e_h1 * self.i * self.lr
        self.b_h1 += e_o * self.lr

    def train(self, input_m, label_m):
        output_m = self.feedforward(input_m)
        error_m = label_m - output_m
        self.backpropagate(error_m)
        return output_m

    def test(self, input_m):
        output_m = self.feedforward(input_m)
        return output_m

    def act_relu(self, wsum):
        return wsum.clip(0)

    def act_sigmoid(self, wsum):
        return 1/(1+np.exp(-wsum))

if __name__ == "__main__":

    nn = NN((2, 6, 1), "sigmoid")
    input_m = np.array([1,2])
    label_m = np.array([1])

    for _ in range(1000):
        output_m = nn.train(input_m, label_m)
        print(output_m)
        