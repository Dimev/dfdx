use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

use curl::easy::Easy;

use crate::prelude::{HasArrayData, SubsetIterator, Tensor2D, TensorCreator};

pub struct Cifar10 {
    pixels: Vec<f32>,
    labels: Vec<u8>,
}

pub const CLASS_NAMES: [&str; 10] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
];

const DIR_NAME: &str = "cifar-10-batches-bin";
const TRAIN_FILES: [&str; 5] = [
    "data_batch_1.bin",
    "data_batch_2.bin",
    "data_batch_3.bin",
    "data_batch_4.bin",
    "data_batch_5.bin",
];

const TEST_FILES: [&str; 1] = ["test_batch.bin"];

impl Cifar10 {
    pub fn train_data<P: AsRef<Path>>(root: P) -> Result<Self, CifarDownloadError> {
        let root = root.as_ref();

        let data_dir = root.join(DIR_NAME);
        if !data_dir.exists() {
            download_all(root)?;
        }
        let mut pixels = Vec::new();
        let mut labels = Vec::new();
        for f in TRAIN_FILES {
            let f_path = data_dir.join(f);
            if !f_path.exists() {
                download_all(root)?;
            }
            load_bin(f_path, &mut pixels, &mut labels)?;
        }

        Ok(Self { pixels, labels })
    }

    pub fn test_data<P: AsRef<Path>>(root: P) -> Result<Self, CifarDownloadError> {
        let root = root.as_ref();

        let data_dir = root.join(DIR_NAME);
        if !data_dir.exists() {
            download_all(root)?;
        }
        let mut pixels = Vec::new();
        let mut labels = Vec::new();
        for f in TEST_FILES {
            let f_path = data_dir.join(f);
            if !f_path.exists() {
                download_all(root)?;
            }
            load_bin(f_path, &mut pixels, &mut labels)?;
        }

        Ok(Self { pixels, labels })
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn get_batch<const B: usize>(
        &self,
        idxs: [usize; B],
    ) -> (Tensor2D<B, { 3 * 32 * 32 }>, Tensor2D<B, 10>) {
        let mut img = Tensor2D::zeros();
        let mut lbl = Tensor2D::zeros();
        let img_data = img.mut_data();
        let lbl_data = lbl.mut_data();
        for (batch_i, &img_idx) in idxs.iter().enumerate() {
            let start = (3 * 32 * 32) * img_idx;
            img_data[batch_i].copy_from_slice(&self.pixels[start..start + (3 * 32 * 32)]);
            lbl_data[batch_i][self.labels[img_idx] as usize] = 1.0;
        }
        (img, lbl)
    }

    pub fn batches<R: rand::Rng, const B: usize>(
        &self,
        rng: &mut R,
    ) -> impl '_ + Iterator<Item = (Tensor2D<B, { 3 * 32 * 32 }>, Tensor2D<B, 10>)> {
        SubsetIterator::<B>::shuffled(self.len(), rng).map(|i| self.get_batch(i))
    }
}

const URL: &str = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
const MD5: &str = "c32a1d4ab5d03f1284b67883e8d87530";

fn load_bin<P: AsRef<Path>>(
    path: P,
    pixels: &mut Vec<f32>,
    labels: &mut Vec<u8>,
) -> Result<(), std::io::Error> {
    let f = File::open(path)?;
    assert_eq!(f.metadata()?.len(), 3073 * 10_000);

    let mut r = BufReader::new(f);
    let mut buf: Vec<u8> = vec![0; 3073];
    for _ in 0..10_000 {
        r.read_exact(&mut buf)?;
        labels.push(buf[0]);
        pixels.extend(buf[1..].iter().map(|&x| x as f32 / 255.0));
    }
    Ok(())
}

fn download_all<P: AsRef<Path>>(root: P) -> Result<(), CifarDownloadError> {
    download(root, URL, MD5)
}

fn download<P: AsRef<Path>>(root: P, url: &str, md5: &str) -> Result<(), CifarDownloadError> {
    let root = root.as_ref();
    std::fs::create_dir_all(root)?;

    let mut compressed = Vec::new();
    let mut easy = Easy::new();
    easy.url(url).unwrap();
    easy.progress(true).unwrap();

    println!("Downloading {url}");
    {
        let mut dl = easy.transfer();
        let pb = indicatif::ProgressBar::new(1);
        dl.progress_function(move |total_dl, cur_dl, _, _| {
            pb.set_length(total_dl as u64);
            pb.set_position(cur_dl as u64);
            true
        })?;
        dl.write_function(|data| {
            compressed.extend_from_slice(data);
            Ok(data.len())
        })?;
        dl.perform()?;
    }

    println!("Verifying hash is {md5}");
    let digest = md5::compute(&compressed);
    if format!("{:?}", digest) != md5 {
        return Err(CifarDownloadError::Md5Mismatch);
    }

    println!("Deflating {} bytes", compressed.len());
    let mut uncompressed = Vec::new();
    let mut decoder = flate2::read::GzDecoder::new(&compressed[..]);
    decoder.read_to_end(&mut uncompressed)?;

    let mut archive = tar::Archive::new(&uncompressed[..]);
    archive.unpack(root)?;

    Ok(())
}

#[derive(Debug)]
pub enum CifarDownloadError {
    IoError(std::io::Error),
    CurlError(curl::Error),
    Md5Mismatch,
}

impl std::fmt::Display for CifarDownloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?}", self))
    }
}

impl std::error::Error for CifarDownloadError {}

impl From<std::io::Error> for CifarDownloadError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

impl From<curl::Error> for CifarDownloadError {
    fn from(e: curl::Error) -> Self {
        Self::CurlError(e)
    }
}
