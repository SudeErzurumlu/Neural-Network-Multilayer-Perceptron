<?php
class NeuralNetwork {
    private $weights;
    private $learningRate;

    public function __construct($inputNodes, $outputNodes, $learningRate) {
        $this->weights = array_fill(0, $inputNodes, array_fill(0, $outputNodes, rand() / getrandmax()));
        $this->learningRate = $learningRate;
    }

    private function sigmoid($x) {
        return 1 / (1 + exp(-$x));
    }

    private function sigmoidDerivative($x) {
        return $x * (1 - $x);
    }

    public function feedForward($inputs) {
        $outputs = [];
        foreach ($this->weights as $i => $weightsRow) {
            $outputs[$i] = $this->sigmoid(array_sum(array_map(fn($w, $inp) => $w * $inp, $weightsRow, $inputs)));
        }
        return $outputs;
    }

    public function train($inputs, $targets) {
        $outputs = $this->feedForward($inputs);
        $errors = array_map(fn($target, $output) => $target - $output, $targets, $outputs);

        foreach ($this->weights as $i => &$weightsRow) {
            foreach ($weightsRow as $j => &$weight) {
                $weight += $this->learningRate * $errors[$i] * $this->sigmoidDerivative($outputs[$i]) * $inputs[$j];
            }
        }
    }
}

// Example usage
$nn = new NeuralNetwork(2, 1, 0.1);
$trainingData = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
];

for ($epoch = 0; $epoch < 10000; $epoch++) {
    foreach ($trainingData as [$inputs, $target]) {
        $nn->train($inputs, $target);
    }
}

$test = [1, 0];
$result = $nn->feedForward($test);
echo "Output for " . implode(", ", $test) . ": " . implode(", ", $result) . "\n";
?>
