use ndarray::{Array, Array2, array, s, ArrayView2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;

pub struct NeuralNetwork {
    pub W: Vec<Array2<f32>>, // Weights of the network
    pub A: Vec<Array2<f32>>, // Hidden layer output
}

impl NeuralNetwork {
    pub fn new(layers: Vec<usize>, n_inputs: usize) -> NeuralNetwork{

        let mut weights: Vec<Array2<f32>> = vec![];
        let mut activations: Vec<Array2<f32>> = vec![];

        // Initialize empty activation matrices
        for i in 0..3 {
            activations.push(Array2::zeros((0,0)));
        }

        // Initialize weights
        let mut prev: usize = n_inputs;
        for l in layers {
            let shape = (prev, l);
            println!("{:?}", shape);
            weights.push(Array::random(shape, StandardNormal));

            prev = l;
        }

        NeuralNetwork {
            W: weights,
            A: activations,
        }
    }

    pub fn forward(&mut self, X: ArrayView2<f32>) -> Array2<f32> {
        /*
         * Forward input through the network
         * X: Array2: [bs, 784]
         */         

        let mut l1 = X.dot(&self.W[0]);  // [bs, 64]
        l1 = l1.mapv(|x| sigmoid(x));
        *&mut self.A[0] = l1.clone();

        let mut l2 = l1.dot(&self.W[1]); // [bs, 32]
        l2 = l2.mapv(|x| sigmoid(x));
        *&mut self.A[1] = l2.clone();

        let mut l3 = l2.dot(&self.W[2]); // [bs, 10]
        let out = softmax(&l3);

        out
    }

    pub fn backward(&self, X: ArrayView2<f32>, Y: ArrayView2<f32>, outputs: ArrayView2<f32>) -> Vec<Array2<f32>> {
        /*
         * Perform a backward pass through the network that calculates the gradients
         * by backpropagation.
         *
         * X:           [bs, 784]
         * Y:           [bs, 10]
         * outputs:     [bs, 10]
         */

        let bs: f32 = X.shape()[0] as f32;

        // Error in the output layer
        let output_error = &outputs - &Y; // [bs, 10]

        // Errors in hidden layers
        let l2_error = &output_error.dot(&self.W[2].t()) * (&self.A[1] * (1.0-&self.A[1]));
        let l1_error = &l2_error.dot(&self.W[1].t()) * (&self.A[0] * (1.0-&self.A[0]));

        // Calculate gradients
        let mut grads: Vec<Array2<f32>> = vec![];

        grads.push((X.t().dot(&l1_error))/bs);
        grads.push((self.A[0].t().dot(&l2_error))/bs);
        grads.push((self.A[1].t().dot(&output_error))/bs);

        grads
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0/(1.0+(-x).exp())
}

fn softmax(x: &Array2<f32>) -> Array2<f32> {
    // TODO: This could very well be very slow

    let mut s = x.mapv(f32::exp);

    for mut row in s.rows_mut() {
        row /= row.sum();
    }

    s
}
