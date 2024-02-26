pub mod network;
pub mod mnist;
pub mod plots;

use std::time::Instant;

use ndarray::{Array2, ArrayView2, ArrayView1, Axis, s};

use network::NeuralNetwork;
use mnist::load_mnist;
use plots::plot_loss;


const N_INPUTS: usize = 784;

const BATCH_SIZE: usize = 50;
const LEARNING_RATE: f32 = 0.01;
const EPOCHS: usize = 20;

const VALIDATION_SIZE: usize = 5000; // Number of examples from the dataset used for validation
const VALIDATE_FREQ: usize = 240;    // Validate the model every # number of batches


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

fn validate(model: &mut NeuralNetwork, x_batch: ArrayView2<f32>, y_batch: ArrayView2<f32>, x_val: ArrayView2<f32>, y_val: ArrayView2<f32>) -> (f32, f32, f32) {
    /*
     * Validate the model by testing on data outside the training set
     */

    let train_accuracy = accuracy(y_batch, model.forward(x_batch).view());

    let val_outputs = model.forward(x_val);

    let val_accuracy = accuracy(y_val, val_outputs.view());
    let val_loss = cross_entropy_loss(val_outputs.view(), y_val);

    (val_loss, val_accuracy, train_accuracy)
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

    let bs: f32 = a.shape()[0] as f32;

    -(&y * &(a.mapv(f32::ln)).view()).sum() / bs
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

    let (grads, b_grads) = model.backward(x_batch, y_batch, outputs.view());

    // Gradient descent, using the gradients to update the weights and biases of the network
    for i in 0..grads.len() {
        model.W[i] = &model.W[i] - LEARNING_RATE * &grads[i];
        model.B[i] = &model.B[i] - LEARNING_RATE * &b_grads[i];
    }

    cross_entropy_loss(outputs.view(), y_batch)
}

fn split_dataset(dataset: &(Array2<f32>, Array2<f32>), val_size: usize) ->
        (ArrayView2<f32>, ArrayView2<f32>, ArrayView2<f32>, ArrayView2<f32>) {
    /*
     *  Separate the entire dataset into validation and train sets
     */

    let cutoff = dataset.0.shape()[0] - val_size;

    let x_train = dataset.0.slice(s![..cutoff, ..]);
    let y_train = dataset.1.slice(s![..cutoff, ..]);

    let x_validate = dataset.0.slice(s![cutoff.., ..]);
    let y_validate = dataset.1.slice(s![cutoff.., ..]);

    (x_train, y_train, x_validate, y_validate)
}

fn train(mut model: NeuralNetwork, dataset: (Array2<f32>, Array2<f32>), epochs: usize) -> (Vec<f32>, Vec<(f32, f32)>) {
    println!("Training.. ({} epochs)", epochs);

    let (x_train, y_train, x_val, y_val) = split_dataset(&dataset, VALIDATION_SIZE);

    let steps_per_epoch: usize = y_train.shape()[0] / BATCH_SIZE;

    let mut train_loss_record: Vec<f32> = Vec::new();
    let mut val_loss_record: Vec<(f32, f32)> = Vec::new();

    // Total number of training steps completed
    let mut steps = 0;

    let mut now = Instant::now();

    for e in 0..epochs {
        println!("********** Epoch {} **********", e);

        for i in 0..steps_per_epoch {
            let x_batch = x_train.slice(s![i..i+BATCH_SIZE,..]);
            let y_batch = y_train.slice(s![i..i+BATCH_SIZE,..]);

            let loss: f32 = train_step(&mut model, x_batch, y_batch);
            train_loss_record.push(loss);

            steps+= 1;

            if steps % VALIDATE_FREQ == 0 {
                let (val_loss, val_accuracy, train_accuracy) = validate(&mut model, x_batch, y_batch, x_val, y_val);

                val_loss_record.push((steps as f32, val_loss));

                println!("Train - loss: {:.3}, accuracy: {:.2}\t | Validation - loss: {:.3}, accuracy: {:.2}",
                    train_loss_record[train_loss_record.len()-1], train_accuracy, val_loss, val_accuracy);
            }
        }

        let elapsed = now.elapsed().as_secs_f32();
        println!("({:.2} steps/s)\n", (steps_per_epoch as f32) / elapsed);

        now = Instant::now();
    }

    (train_loss_record, val_loss_record)
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

    let (train_loss, validation_loss) = train(network, dataset, EPOCHS);

    let plot_ok = plot_loss(train_loss, validation_loss);

    match plot_ok {
        Ok(_) => (),
        Err(e) => println!("Error creating plot: {:?}", e),
    }
}
