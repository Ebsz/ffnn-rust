use ndarray::{Array, Array2, array, s, ArrayView2, Axis};
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
         * Forward input through the network
         * X: Array2: [bs, 784]
         */         

        let mut l1 = X.dot(&self.W[0]);  // [bs, 64]
        l1 = l1.mapv(|x| sigmoid(x));

        let mut l2 = l1.dot(&self.W[1]); // [bs, 32]
        l2 = l2.mapv(|x| sigmoid(x));

        let mut l3 = l2.dot(&self.W[2]); // [bs, 10]

        let out = softmax(&l3);

        out
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

    e
}
