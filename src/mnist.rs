use std::fs::File;
use std::io::Read;

use image::{ImageBuffer};

use ndarray::{Array, Array2, array, s, ArrayView};

pub const IMG_SIZE: usize = 784;

const LABEL_HEADER_LEN: usize = 8;
const IMG_HEADER_LEN: usize = 16;
const N_IMAGES_TRAIN: usize = 60000;


pub fn load_mnist() -> (Array2<f32>, Array2<f32>) {
    println!("Loading mnist dataset");

    let mut training_data: (Vec<u8>, Vec<u8>) = load_training_set();
    let img: Array2<f32> = normalize_img(data_to_arrays(training_data.0));
    let labels: Array2<f32> = one_hot_encode(training_data.1);

    assert!(img.shape() == [N_IMAGES_TRAIN, IMG_SIZE]);

    (img, labels)
}

fn one_hot_encode(labels: Vec<u8>) -> Array2<f32> {
    let mut L: Array2<f32> = Array2::zeros((labels.len(), 10));

    for (n, l) in labels.into_iter().enumerate() {
        L.row_mut(n)[(l as usize)] = 1.0;
    }

    L
}

fn normalize_img(data: Array2<f32>) -> Array2<f32> {
    // Normalize each point in the images from [0,255] to (-1,1)
    data * (2.0/255.0) - 1.0
}

fn data_to_arrays(data: Vec<u8>) -> Array2<f32> {
    /* Convert the raw image data to arrays that can be batched
     * Returns an Array2 of shape [60000, 784] */

    Array::from_shape_vec((60000,IMG_SIZE), data).unwrap().mapv(|x| f32::from(x))
}

fn load_training_set() -> (Vec<u8>, Vec<u8>) {

    let mut img = read_file("mnist/train-images-idx3-ubyte");
    let mut labels = read_file("mnist/train-labels-idx1-ubyte");

    // Remove file headers
    img.drain(..IMG_HEADER_LEN);
    labels.drain(..LABEL_HEADER_LEN);

    assert_eq!(img.len(), N_IMAGES_TRAIN*IMG_SIZE);
    assert_eq!(labels.len(), N_IMAGES_TRAIN);

    (img, labels)
}

fn read_file(path: &'static str) -> Vec::<u8> {
    // TODO: this is a wonky way of handling the Result(); improve
    if let Ok(mut file) = File::open(path) {
        let mut buffer: Vec<u8> = Vec::new();

        file.read_to_end(&mut buffer);

        buffer

    } else {
        Vec::new()
    }
}


fn save_image(buf: &[u8] ) {
    image::save_buffer("img.png", buf, 28,28, image::ColorType::L8);
}
