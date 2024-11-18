# Neural Network in Julia for XOR Problem
using Random

# Activation function and its derivative
function sigmoid(x)
    1 / (1 + exp(-x))
end

function sigmoid_derivative(x)
    x * (1 - x)
end

# Initialize random weights
Random.seed!(1234) # For reproducibility
weights_input_hidden = rand(2, 2)
weights_hidden_output = rand(1, 2)

# XOR data
inputs = [0 0; 0 1; 1 0; 1 1]
expected_output = [0, 1, 1, 0]'

# Hyperparameters
learning_rate = 0.1
epochs = 10000

# Training process
for epoch in 1:epochs
    # Forward propagation
    hidden_layer_input = inputs * weights_input_hidden'
    hidden_layer_output = sigmoid.(hidden_layer_input)

    final_layer_input = hidden_layer_output * weights_hidden_output'
    final_output = sigmoid.(final_layer_input)

    # Backpropagation
    output_error = expected_output' - final_output
    d_output = output_error .* sigmoid_derivative.(final_output)

    hidden_error = d_output * weights_hidden_output
    d_hidden = hidden_error .* sigmoid_derivative.(hidden_layer_output)

    # Update weights
    weights_hidden_output += learning_rate .* (d_output' * hidden_layer_output)
    weights_input_hidden += learning_rate .* (d_hidden' * inputs)
end

# Testing the network
println("Trained weights:")
println("Input to Hidden Layer: ", weights_input_hidden)
println("Hidden to Output Layer: ", weights_hidden_output)

println("Final Output:")
hidden_layer_output = sigmoid.(inputs * weights_input_hidden')
final_output = sigmoid.(hidden_layer_output * weights_hidden_output')
println(final_output)
