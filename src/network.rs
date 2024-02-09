use ndarray::{Array, Array2, array, s, ArrayView2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

pub struct NeuralNetwork {
    pub W: Vec<Array2<f32>>
}

impl NeuralNetwork {
    pub fn new(layers: Vec<usize>, n_inputs: usize) -> NeuralNetwork{

        let mut weights: Vec<Array2<f32>> = vec![];

        let mut prev: usize = n_inputs;
        for l in layers {
            let shape = (prev, l);

            weights.push(Array::random(shape, Uniform::new(-1.,1.))); // TODO: currently initializing uniformly; change to gaussian?
            prev = l;
        }

        NeuralNetwork {
            W: weights
        }
    }

    pub fn forward(&self, X: ArrayView2<f32>) -> Array2<f32> {
        /*
         * Forward the input through the network
         * X: Array2 of size [bs, 784]
         */         

        let mut out = X.dot(&self.W[0]);

        out.dot(&self.W[1])

        // TODO: assert output is of size [BS, 10]

    }
}

fn sigmoid(z: f32) -> f32 {
    1.0/(1.0+(-z).exp())
}
