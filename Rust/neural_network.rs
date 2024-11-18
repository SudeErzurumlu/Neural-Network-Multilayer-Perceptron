use rand::Rng;

#[derive(Debug)]
struct NeuralNetwork {
    weights: Vec<Vec<f64>>,
    learning_rate: f64,
}

impl NeuralNetwork {
    fn new(input_nodes: usize, output_nodes: usize, learning_rate: f64) -> NeuralNetwork {
        let mut rng = rand::thread_rng();
        let weights = (0..input_nodes)
            .map(|_| {
                (0..output_nodes)
                    .map(|_| rng.gen_range(0.0..1.0))
                    .collect::<Vec<f64>>()
            })
            .collect();

        NeuralNetwork {
            weights,
            learning_rate,
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn sigmoid_derivative(x: f64) -> f64 {
        x * (1.0 - x)
    }

    fn feed_forward(&self, inputs: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .map(|weights_row| {
                let sum: f64 = weights_row
                    .iter()
                    .zip(inputs)
                    .map(|(w, inp)| w * inp)
                    .sum();
                NeuralNetwork::sigmoid(sum)
            })
            .collect()
    }

    fn train(&mut self, inputs: &[f64], targets: &[f64]) {
        let outputs = self.feed_forward(inputs);
        let errors: Vec<f64> = targets
            .iter()
            .zip(&outputs)
            .map(|(target, output)| target - output)
            .collect();

        for (i, weights_row) in self.weights.iter_mut().enumerate() {
            for (j, weight) in weights_row.iter_mut().enumerate() {
                *weight += self.learning_rate
                    * errors[i]
                    * NeuralNetwork::sigmoid_derivative(outputs[i])
                    * inputs[j];
            }
        }
    }
}

fn main() {
    let mut nn = NeuralNetwork::new(2, 1, 0.1);

    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    for _ in 0..10000 {
        for (inputs, targets) in &training_data {
            nn.train(inputs, targets);
        }
    }

    let test = vec![1.0, 0.0];
    let result = nn.feed_forward(&test);

    println!(
        "Output for inputs {:?}: {:?}",
        test,
        result
    );
}
