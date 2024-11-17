#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

class NeuralNetwork {
public:
    vector<vector<float>> weights1, weights2;
    vector<float> bias1, bias2;
    float learning_rate;

    NeuralNetwork(int input_size, int hidden_size, int output_size, float learning_rate = 0.1) {
        this->learning_rate = learning_rate;

        // Initialize weights and biases
        weights1.resize(input_size, vector<float>(hidden_size));
        weights2.resize(hidden_size, vector<float>(output_size));
        bias1.resize(hidden_size);
        bias2.resize(output_size);

        srand(time(0));

        // Random initialization of weights and biases
        for (int i = 0; i < input_size; i++) {
            for (int j = 0; j < hidden_size; j++) {
                weights1[i][j] = static_cast<float>(rand()) / RAND_MAX;
            }
        }
        for (int i = 0; i < hidden_size; i++) {
            for (int j = 0; j < output_size; j++) {
                weights2[i][j] = static_cast<float>(rand()) / RAND_MAX;
            }
        }
        for (int i = 0; i < hidden_size; i++) {
            bias1[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        for (int i = 0; i < output_size; i++) {
            bias2[i] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Sigmoid activation function
    float sigmoid(float x) {
        return 1.0 / (1.0 + exp(-x));
    }

    // Derivative of sigmoid function
    float sigmoid_derivative(float x) {
        return x * (1.0 - x);
    }

    // Forward pass
    vector<float> forward(const vector<float>& input) {
        // Input to hidden layer
        vector<float> hidden(hidden_size);
        for (int i = 0; i < hidden_size; i++) {
            hidden[i] = 0.0;
            for (int j = 0; j < input.size(); j++) {
                hidden[i] += input[j] * weights1[j][i];
            }
            hidden[i] += bias1[i];
            hidden[i] = sigmoid(hidden[i]);
        }

        // Hidden to output layer
        vector<float> output(output_size);
        for (int i = 0; i < output_size; i++) {
            output[i] = 0.0;
            for (int j = 0; j < hidden.size(); j++) {
                output[i] += hidden[j] * weights2[j][i];
            }
            output[i] += bias2[i];
            output[i] = sigmoid(output[i]);
        }

        return output;
    }

    // Backpropagation
    void backpropagate(const vector<float>& input, const vector<float>& target) {
        vector<float> hidden(hidden_size);
        for (int i = 0; i < hidden_size; i++) {
            hidden[i] = 0.0;
            for (int j = 0; j < input.size(); j++) {
                hidden[i] += input[j] * weights1[j][i];
            }
            hidden[i] += bias1[i];
            hidden[i] = sigmoid(hidden[i]);
        }

        vector<float> output(output_size);
        for (int i = 0; i < output_size; i++) {
            output[i] = 0.0;
            for (int j = 0; j < hidden.size(); j++) {
                output[i] += hidden[j] * weights2[j][i];
            }
            output[i] += bias2[i];
            output[i] = sigmoid(output[i]);
        }

        // Calculate error
        vector<float> output_error(output_size);
        for (int i = 0; i < output_size; i++) {
            output_error[i] = target[i] - output[i];
        }

        // Backpropagate to the second layer
        vector<float> output_delta(output_size);
        for (int i = 0; i < output_size; i++) {
            output_delta[i] = output_error[i] * sigmoid_derivative(output[i]);
        }

        // Backpropagate to the first layer
        vector<float> hidden_error(hidden_size);
        for (int i = 0; i < hidden_size; i++) {
            hidden_error[i] = 0.0;
            for (int j = 0; j < output_size; j++) {
                hidden_error[i] += output_delta[j] * weights2[i][j];
            }
        }

        vector<float> hidden_delta(hidden_size);
        for (int i = 0; i < hidden_size; i++) {
            hidden_delta[i] = hidden_error[i] * sigmoid_derivative(hidden[i]);
        }

        // Update weights and biases
        for (int i = 0; i < hidden_size; i++) {
            for (int j = 0; j < output_size; j++) {
                weights2[i][j] += learning_rate * output_delta[j] * hidden[i];
            }
        }

        for (int i = 0; i < output_size; i++) {
            bias2[i] += learning_rate * output_delta[i];
        }

        for (int i = 0; i < input.size(); i++) {
            for (int j = 0; j < hidden_size; j++) {
                weights1[i][j] += learning_rate * hidden_delta[j] * input[i];
            }
        }

        for (int i = 0; i < hidden_size; i++) {
            bias1[i] += learning_rate * hidden_delta[i];
        }
    }

private:
    int input_size, hidden_size, output_size;
};

int main() {
    NeuralNetwork nn(2, 2, 1);  // 2 inputs, 2 hidden neurons, 1 output
    vector<vector<float>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<float> targets = {0, 1, 1, 0};  // XOR target

    // Train the network
    for (int epoch = 0; epoch < 10000; epoch++) {
        for (int i = 0; i < inputs.size(); i++) {
            nn.backpropagate(inputs[i], {targets[i]});
        }
    }

    // Test the network
    for (int i = 0; i < inputs.size(); i++) {
        vector<float> output = nn.forward(inputs[i]);
        cout << "Input: (" << inputs[i][0] << ", " << inputs[i][1] << ") -> Output: " << output[0] << endl;
    }

    return 0;
}
