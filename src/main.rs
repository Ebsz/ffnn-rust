pub mod network;
pub mod mnist;
pub mod plots;

use std::time::Instant;

use ndarray::{Array2, ArrayView1, ArrayView2, array, s,  Axis};

use network::NeuralNetwork;
use mnist::load_mnist;
use plots::plot_loss;


const BATCH_SIZE: usize = 50;
const N_INPUTS: usize = 784;
const LEARNING_RATE: f32 = 0.005;

const REPORT_FREQ: usize = 30; // Report info during training every # batches


fn cross_entropy_loss(a: ArrayView2<f32>, y: ArrayView2<f32>) -> f32 {
    /*
     * Calculate cross-entropy loss, returning the average over the batch.
     *
     * This is calculated as: -sum(y * ln(a)) / batch_size
     *
     * a: [bs, 10] predictions
     * y: [bs, 10] ground truth
     */

    -(&y * &(a.mapv(f32::ln)).view()).sum() / (BATCH_SIZE as f32)
}

fn train_step(model: &mut NeuralNetwork, x_batch: ArrayView2<f32>, y_batch: ArrayView2<f32>) -> f32 {
    /*
     * Perform a training step, including forward, backward, and gradient descent.
     *
     * X: [bs, 784] array of images to train on
     * Y: [bs, 10] array of labels corresponding to each image
     */
    let outputs = model.forward(x_batch);
    assert!(outputs.shape() == [BATCH_SIZE,10]);

    let grads = model.backward(x_batch, y_batch, outputs.view());

    // Gradient descent, using the gradients to update the weights
    for i in 0..grads.len() {
        model.W[i] = &model.W[i] - LEARNING_RATE * &grads[i];
    }

    cross_entropy_loss(outputs.view(), y_batch)
}

fn train(mut model: NeuralNetwork, dataset: (Array2<f32>, Array2<f32>)) -> Vec<f32> {
    println!("Training..");

    let img_array = dataset.0;
    let labels = dataset.1;

    let its_per_epoch: usize = labels.shape()[0] / BATCH_SIZE;

    let mut its = 0;

    let mut loss_history: Vec<f32> = Vec::new();

    let mut now = Instant::now();
    for i in 0..its_per_epoch {
        let x_batch = img_array.slice(s![i..i+BATCH_SIZE,..]);
        let y_batch = labels.slice(s![i..i+BATCH_SIZE,..]);

        let loss: f32 = train_step(&mut model, x_batch, y_batch);
        loss_history.push(loss);

        its+= 1;
        if its % REPORT_FREQ == 0 {
            let elapsed = now.elapsed().as_secs_f32();

            println!("Loss: {}\t\t  {} its/s  - \t {}/{}", loss, (REPORT_FREQ as f32)/elapsed, its, its_per_epoch);

            now = Instant::now();
        }
    }

    loss_history
}

fn create_network() -> NeuralNetwork {
    println!("Initializing network");

    let layers: Vec<usize> = vec!(64, 32, 10);

    NeuralNetwork::new(layers, N_INPUTS)
}



fn main() {
    // The dataset consists of a [60000, 784] array containing the image data,
    // and a [60000, 10] array containing one-hot encoded labels
    let dataset: (Array2<f32>, Array2<f32>) = load_mnist();
    let mut network: NeuralNetwork = create_network();

    let loss_history = train(network, dataset);

    plot_loss(loss_history);
}
