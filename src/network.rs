use ndarray::{Array, Array1, Array2, ArrayView2, Axis};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;

pub struct NeuralNetwork {
    pub W: Vec<Array2<f32>>, // Weights
    pub B: Vec<Array1<f32>>, // Biases
    pub A: Vec<Array2<f32>>, // Hidden layer output
}

impl NeuralNetwork {
    pub fn new(layers: Vec<usize>, n_inputs: usize) -> NeuralNetwork {
        let mut weights: Vec<Array2<f32>> = vec![];
        let mut activations: Vec<Array2<f32>> = vec![];
        let mut bias: Vec<Array1<f32>> = vec![];

        // Initialize empty activation matrices
        for _ in 0..layers.len() {
            activations.push(Array2::zeros((0, 0)));
        }

        // Fix the RNG seed
        let mut rng = StdRng::seed_from_u64(0);

        // Initialize weights
        let mut prev: usize = n_inputs;
        for l in layers {
            let shape = (prev, l);
            weights.push(Array::random_using(shape, StandardNormal, &mut rng));
            bias.push(Array::random_using(l, StandardNormal, &mut rng));

            prev = l;
        }

        NeuralNetwork {
            W: weights,
            B: bias,
            A: activations,
        }
    }

    pub fn forward(&mut self, X: ArrayView2<f32>) -> Array2<f32> {
        /*
         * Forward input through the network
         *
         * X: Array2: [bs, 784]
         */

        // Input layer
        let mut l1 = X.dot(&self.W[0]) + &self.B[0];
        l1 = l1.mapv(|x| sigmoid(x));
        *&mut self.A[0] = l1.clone();

        // Hidden layers
        for i in 1..(self.W.len() - 1) {
            *&mut self.A[i] = (&self.A[i - 1].dot(&self.W[i]) + &self.B[i]).mapv(|x| sigmoid(x));
        }

        // Output layer
        let out_idx: usize = self.A.len() - 1;
        let out = &self.A[out_idx - 1].dot(&self.W[out_idx]) + &self.B[out_idx];

        softmax(&out)
    }

    pub fn backward(
        &self,
        X: ArrayView2<f32>,
        Y: ArrayView2<f32>,
        outputs: ArrayView2<f32>,
    ) -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {
        /*
         * Perform a backward pass through the network that calculates the gradients
         * by backpropagation.
         *
         * X:           [bs, 784]
         * Y:           [bs, 10]
         * outputs:     [bs, 10]
         */

        let bs: f32 = X.shape()[0] as f32;
        let n_layers: usize = self.W.len();

        // Error in the output layer
        let output_error = &outputs - &Y;

        let mut errors: Vec<Array2<f32>> = vec![];
        errors.push(output_error);

        for i in 0..(n_layers - 1) {
            let err = &errors[i].dot(&self.W[n_layers - (i + 1)].t())
                * (&self.A[n_layers - (i + 2)] * (1.0 - &self.A[n_layers - (i + 2)]));

            errors.push(err);
        }

        // First element is the error of the first layer
        errors.reverse();

        // Weight gradients
        let mut grads: Vec<Array2<f32>> = vec![];
        grads.push((X.t().dot(&errors[0])) / bs);

        for i in 0..(n_layers - 1) {
            grads.push((self.A[i].t().dot(&errors[i + 1])) / bs);
        }

        // Bias gradients
        let mut b_grads: Vec<Array1<f32>> = vec![];
        for i in 0..n_layers {
            b_grads.push(errors[i].sum_axis(Axis(0)) / bs);
        }

        (grads, b_grads)
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax(x: &Array2<f32>) -> Array2<f32> {
    let mut s = x.mapv(f32::exp);

    for mut row in s.rows_mut() {
        row /= row.sum();
    }
    s
}
