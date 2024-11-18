import Foundation

func sigmoid(_ x: Double) -> Double {
    return 1.0 / (1.0 + exp(-x))
}

func sigmoidDerivative(_ x: Double) -> Double {
    return x * (1.0 - x)
}

class NeuralNetwork {
    var weights: [[Double]]
    var learningRate: Double

    init(inputNodes: Int, outputNodes: Int, learningRate: Double) {
        self.learningRate = learningRate
        self.weights = (0..<inputNodes).map { _ in
            (0..<outputNodes).map { _ in Double.random(in: 0...1) }
        }
    }

    func feedForward(inputs: [Double]) -> [Double] {
        return weights.map { row in
            let sum = zip(row, inputs).reduce(0.0) { $0 + $1.0 * $1.1 }
            return sigmoid(sum)
        }
    }

    func train(inputs: [Double], targets: [Double]) {
        let outputs = feedForward(inputs: inputs)
        let errors = zip(targets, outputs).map { $0 - $1 }

        for i in 0..<weights.count {
            for j in 0..<weights[i].count {
                weights[i][j] += learningRate * errors[i] * sigmoidDerivative(outputs[i]) * inputs[j]
            }
        }
    }
}

let nn = NeuralNetwork(inputNodes: 2, outputNodes: 1, learningRate: 0.1)
let trainingData = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0])
]

for _ in 0..<10000 {
    for (inputs, targets) in trainingData {
        nn.train(inputs: inputs, targets: targets)
    }
}

let test = [1.0, 0.0]
let result = nn.feedForward(inputs: test)
print("Output for inputs \(test): \(result)")
