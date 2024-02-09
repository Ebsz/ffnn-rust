use std::fs::File;
use std::io::Read;

use image::{ImageBuffer};

use ndarray::{Array, Array2, array, s, ArrayView};

pub const IMG_SIZE: usize = 784;

const LABEL_HEADER_LEN: usize = 8;
const IMG_HEADER_LEN: usize = 16;
const N_IMAGES_TRAIN: usize = 60000;


pub fn load_mnist() -> (Array2<f32>, Vec<u8>) {
    println!("Loading mnist dataset");

    let mut training_data: (Vec<u8>, Vec<u8>) = load_training_set();

    let img_array: Array2<f32> = data_to_arrays(training_data.0);

    (img_array, training_data.1)

}

fn data_to_arrays(data: Vec<u8>) -> Array2<f32> {
    /* Convert the raw image data to arrays that can be batched
     * Returns an Array2 of shape [60000, 784] */

    Array::from_shape_vec((60000,IMG_SIZE), data).unwrap().mapv(|x| f32::from(x))
}

fn load_training_set() -> (Vec<u8>, Vec<u8>) {

    let mut img = read_file("mnist/train-images-idx3-ubyte");
    let mut labels = read_file("mnist/train-labels-idx1-ubyte");

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


//TODO: obsolete
//fn parse_images(mut data: Vec::<u8>) -> Vec<u8> {
//    const OFFSET: usize = 16; // Header size.
//    const IMG_SIZE: usize = 784; // Number of bytes per image.
//    const N: usize = 60000; 
//
//    // Discard the header.
//    data.drain(..OFFSET);
//
//    //let img: Vec<&[u8]> = data.chunks(IMG_SIZE).map(|k| k.into()).collect();
//
//    assert_eq!(data.len(), N*IMG_SIZE);
//
//    data
//}
//
//
//fn parse_labels(mut data: Vec::<u8>) -> Vec<u8> {
//    const OFFSET: usize = 8; // Header size.
//    const N: usize = 60000; 
//
//    // Discard the header.
//    data.drain(..OFFSET);
//
//    for i in 0..10 {
//        println!("{}", data[i]);
//    }
//
//    println!("{}", data.len());
//
//
//
//    data
//}
