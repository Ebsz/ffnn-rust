pub mod network;
pub mod mnist;
pub mod plots;

use std::time::Instant;

use ndarray::{Array2, ArrayView2, ArrayView1, Axis, s};

use network::NeuralNetwork;
use mnist::load_mnist;
use plots::plot_loss;


const BATCH_SIZE: usize = 50;
const N_INPUTS: usize = 784;
const LEARNING_RATE: f32 = 0.01;

const VALIDATE_FREQ: usize = 240;   // Validate the model every # batches
const EPOCHS: usize = 1;            // Number of epochs to train for


fn accuracy(y: ArrayView2<f32>, y_hat: ArrayView2<f32>) -> f32 {
    /*
     * Calculate the accuracy of the model ie. the number of accurate predictions
     *
     * y:       [bs, 10] array of model predictions
     * y_hat:   [bs, 10] array of one-hot encoded labels
     */

    fn argmax(a: ArrayView1<f32>) -> usize {
        /* Returns the index of the max value in the array */
        let max = a.iter().reduce(|x, max| if x > max {x} else {max}).unwrap();
        let pos = a.iter().position(|x| x == max).unwrap();
        pos
    }

    let predictions = y_hat.map_axis(Axis(1), |x| argmax(x));
    let label = y.map_axis(Axis(1), |x| argmax(x));

    let correct = predictions.iter().zip(label.iter()).filter(|(a, b)| a == b).count();

    correct as f32 / y.shape()[0] as f32
}

fn validate(model: &mut NeuralNetwork, x_batch: ArrayView2<f32>, y_batch: ArrayView2<f32>) -> f32 {
    /*
     * Validate the model by testing on data outside the training set
     *
     * TODO: Currently does not actually validate, but only calculates
     *       accuracy on the training set; implement
     */
    let outputs = model.forward(x_batch);
    let accuracy = accuracy(y_batch, outputs.view());

    accuracy
}

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

fn train(mut model: NeuralNetwork, dataset: (Array2<f32>, Array2<f32>), epochs: usize) -> (Vec<f32>, Vec<f32>) {
    println!("Training..");

    let x_dataset = dataset.0;
    let y_dataset = dataset.1;

    let its_per_epoch: usize = y_dataset.shape()[0] / BATCH_SIZE;

    let mut loss_history: Vec<f32> = Vec::new();
    let mut accuracy_history: Vec<f32> = Vec::new();

    let mut its = 0;
    let mut now = Instant::now();

    for e in 0..epochs {
        for i in 0..its_per_epoch {
            let x_batch = x_dataset.slice(s![i..i+BATCH_SIZE,..]);
            let y_batch = y_dataset.slice(s![i..i+BATCH_SIZE,..]);

            let loss: f32 = train_step(&mut model, x_batch, y_batch);
            loss_history.push(loss);

            its+= 1;

            if its % VALIDATE_FREQ == 0 {
                let accuracy: f32 = validate(&mut model, x_batch, y_batch);

                accuracy_history.push(accuracy);

                println!("Loss: {:.3}\t Accuracy: {}", loss_history[loss_history.len()-1], accuracy);
            }
        }

        let elapsed = now.elapsed().as_secs_f32();
        println!("***** Epoch {} \t ({:.2} steps/s) *****", e, (its_per_epoch as f32) / elapsed);

        now = Instant::now();
    }

    (loss_history, accuracy_history)
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
    let network: NeuralNetwork = create_network();

    let (loss_history, accuracy_history) = train(network, dataset, EPOCHS);

    let plot_ok = plot_loss(loss_history);

    match plot_ok {
        Ok(_) => (),
        Err(e) => println!("Error creating plot: {:?}", e),
    }
}
