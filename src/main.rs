pub mod network;
pub mod mnist;

use network::NeuralNetwork;
use mnist::load_mnist;

use ndarray::{Array, Array2, array, s, ArrayView2};


const BATCH_SIZE: usize = 50;
const N_INPUTS: usize = 784;

fn create_network() -> NeuralNetwork {
    println!("Initializing network");

    let layers: Vec<usize> = vec!(64, 10);

    NeuralNetwork::new(layers, N_INPUTS)
}

fn train_step(model: &NeuralNetwork, X: ArrayView2<f32>) -> Array2<f32> {
    /*
     * Perform a training step
     */
    let outputs = model.forward(X);
    outputs
}


fn train(model: NeuralNetwork, dataset: (Array2<f32>, Vec<u8>)) {
    println!("Training..");

    //let img_array = to_arrays(dataset.0);
    let img_array = dataset.0;

    let its_per_epoch: usize = dataset.1.len() / BATCH_SIZE;


    let mut its = 0;
    for i in 0..its_per_epoch {
        let batch = img_array.slice(s![i..i+BATCH_SIZE,..]);

        train_step(&model, batch);

        its+= 1;
        if its % 120 == 0 {
            println!("{}/{}", its, its_per_epoch);
        }
    }
}


fn main() {
    let dataset: (Array2<f32>, Vec<u8>) = load_mnist();
    let mut network: NeuralNetwork = create_network();

    train(network, dataset);
}
