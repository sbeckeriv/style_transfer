// This code is not mine own.
//
// First the style transfer code is a copy from [1] tch-rs. The maintainer has been an amazing
// help to me getting going and answering lots of questions. I tweaked the code to take a path for
// the styles. I also catch panics and move on. If it panics most likely do not enough cuda memory.
//
// Second after much reading about two thoughts on keeping the color during style transfer I found
// kmeans_colors. Kmeans provides a bin that can do a color transfer. I ripped that up in two a few
// functions. [2]
//
//
//
// [1] https://github.com/LaurentMazare/tch-rs/blob/488d7e2d71395eb864c8296a21f865e989c70b3f/examples/neural-style-transfer/main.rs
// [2] https://github.com/okaneco/kmeans-colors
extern crate anyhow;
extern crate image;
extern crate kmeans_colors;
extern crate palette;
extern crate structopt;
extern crate tch;

use anyhow::{Result};
use kmeans_colors::{
    get_kmeans, get_kmeans_hamerly, Calculate, CentroidData, Kmeans, MapColor, Sort,
};
use palette::{Lab, Pixel, Srgb, Srgba};
use std::error::Error;
use std::fmt::Write;
use std::fs::File;
use std::io::BufWriter;
use std::panic;
use std::path::{Path, PathBuf};
use structopt::StructOpt;
use tch::vision::{imagenet, vgg};
use tch::{nn, nn::OptimizerConfig, Device, Tensor};

const STYLE_WEIGHT: f64 = 1e12;
const LEARNING_RATE: f64 = 1e-1;

const STYLE_LOW: [usize; 5] = [0, 2, 5, 7, 10];
const STYLE_LOWER: [usize; 8] = [0, 2, 5, 7, 10, 13, 16, 19];
const STYLE_TOLOW: [usize; 12] = [0, 2, 5, 7, 10, 13, 16, 19, 21, 24, 27, 30];
const CONTENT_INDEXES: [usize; 2] = [7, 10];
//const CONTENT_INDEXES: [usize; 1] = [7];

#[derive(StructOpt, Debug)]
pub struct Opt {
    #[structopt(short = "r", long = "runs", default_value = "1500")]
    runs: usize,
    #[structopt(long = "save_runs", default_value = "1000")]
    save_runs: usize,
    #[structopt(short = "d", long = "debug")]
    debug: bool,
    #[structopt(short = "k", long = "keepcolor")]
    keep_color: bool,
    #[structopt(long = "colordepth", default_value = "40")]
    color_depth: usize,
    #[structopt(long = "depth", default_value = "2")]
    style_depth: usize,
    #[structopt(long = "style")]
    style: String,
    #[structopt(long = "source")]
    source: String,
    #[structopt(long = "vgg")]
    vgg: String,
}

/// Parse hex string to Rgb color.
pub fn parse_color(c: &str) -> Result<Srgb<u8>, &str> {
    let red = u8::from_str_radix(
        match &c.get(0..2) {
            Some(x) => x,
            None => {
                eprintln!("Invalid color: {}", c);
                return Err("erong");
            }
        },
        16,
    )
    .unwrap();
    let green = u8::from_str_radix(
        match &c.get(2..4) {
            Some(x) => x,
            None => {
                eprintln!("Invalid color: {}", c);
                return Err("wrong");
            }
        },
        16,
    )
    .unwrap();
    let blue = u8::from_str_radix(
        match &c.get(4..6) {
            Some(x) => x,
            None => {
                eprintln!("Invalid color: {}", c);
                return Err("wrong");
            }
        },
        16,
    )
    .unwrap();
    Ok(Srgb::new(red, green, blue))
}

pub fn save_image(
    imgbuf: &[u8],
    imgx: u32,
    imgy: u32,
    title: &PathBuf,
) -> Result<(), Box<dyn Error>> {
    let mut w = BufWriter::new(File::create(title)?);
    if title.extension().unwrap() == "png" {
        let encoder = image::png::PNGEncoder::new_with_quality(
            w,
            image::png::CompressionType::Best,
            image::png::FilterType::NoFilter,
        );

        // Clean up if file is created but there's a problem writing to it
        match encoder.encode(imgbuf, imgx, imgy, image::ColorType::Rgb8) {
            Ok(_) => {}
            Err(err) => {
                eprintln!("Error: {}.", err);
                std::fs::remove_file(title)?;
            }
        }
    } else {
        let mut encoder = image::jpeg::JPEGEncoder::new_with_quality(&mut w, 90);

        match encoder.encode(imgbuf, imgx, imgy, image::ColorType::Rgb8) {
            Ok(_) => {}
            Err(err) => {
                eprintln!("Error: {}.", err);
                std::fs::remove_file(title)?;
            }
        }
    };

    Ok(())
}

pub fn color_list(file: &PathBuf, color_depth: usize) -> String {
    let img = image::open(file).unwrap().to_rgba();
    let img_vec = img.into_raw();
    let converge = 5.0;

    let runs = 3;

    // Read image buffer into Srgb format
    let rgb: Vec<Srgb> = Srgba::from_raw_slice(&img_vec)
        .iter()
        .map(|x| x.into_format().into())
        .collect();

    // Iterate over amount of runs keeping best results
    let mut result = Kmeans::new();
    (0..runs).for_each(|i| {
        let run_result = get_kmeans_hamerly(color_depth, 20, converge, false, &rgb, i as u64);
        if run_result.score < result.score {
            result = run_result;
        }
    });

    let res = Srgb::sort_indexed_colors(&result.centroids, &result.indices);

    print_colors(&res)
}

/// Prints colors and percentage of their appearance in an image buffer.
pub fn print_colors<C: Calculate + Copy + Into<Srgb>>(colors: &Vec<CentroidData<C>>) -> String {
    let mut col = String::new();
    if let Some((last, elements)) = colors.split_last() {
        for elem in elements {
            write!(&mut col, "{:x},", elem.centroid.into().into_format::<u8>()).unwrap();
        }
        write!(&mut col, "{:x}\n", last.centroid.into().into_format::<u8>()).unwrap();
    }
    col
}

pub fn find_colors(
    input: &str,
    tmp_file: &str,
    colors: Vec<String>,
    max_iter: usize,
    runs: usize,
) -> Result<(), Box<dyn Error>> {
    let converge = 5.0;
    let seed = 0;
    // Initialize user centroids
    let mut centroids: Vec<Lab> = Vec::with_capacity(colors.len());
    for c in colors {
        centroids.push(
            (parse_color(c.trim_start_matches('#'))?)
                .into_format()
                .into(),
        );
    }

    let img = image::open(&PathBuf::from(input)).unwrap().to_rgba();
    let (imgx, imgy) = (img.dimensions().0, img.dimensions().1);
    let img_vec = img.into_raw();

    let lab: Vec<Lab>;
    lab = Srgba::from_raw_slice(&img_vec)
        .iter()
        .map(|x| x.into_format().into())
        .collect();

    // Replace the k-means colors case
    let mut result = Kmeans::new();
    let k = centroids.len();
    if k > 1 {
        (0..runs).for_each(|i| {
            let run_result =
                get_kmeans_hamerly(k, max_iter, converge, false, &lab, seed + i as u64);
            if run_result.score < result.score {
                result = run_result;
            }
        });
    } else {
        (0..runs).for_each(|i| {
            let run_result = get_kmeans(k, max_iter, converge, false, &lab, seed + i as u64);
            if run_result.score < result.score {
                result = run_result;
            }
        });
    }

    let mut res = Lab::sort_indexed_colors(&result.centroids, &result.indices);
    res.iter_mut()
        .zip(&centroids)
        .for_each(|(s, c)| s.centroid = *c);

    // Sorting the centroids now
    res.sort_unstable_by(|a, b| (a.index).cmp(&b.index));
    let sorted: Vec<Lab> = res.iter().map(|x| x.centroid).collect();

    let rgb_centroids = &sorted
        .iter()
        .map(|x| Srgb::from(*x).into_format())
        .collect::<Vec<Srgb<u8>>>();
    let lab: Vec<Srgb<u8>> = Srgb::map_indices_to_centroids(&rgb_centroids, &result.indices);
    save_image(
        Srgb::into_raw_slice(&lab),
        imgx,
        imgy,
        &PathBuf::from(tmp_file),
    )
    .unwrap();

    Ok(())
}

fn gram_matrix(m: &Tensor) -> Tensor {
    let (a, b, c, d) = m.size4().unwrap();
    let m = m.view([a * b, c * d]);
    let g = m.matmul(&m.tr());
    g / (a * b * c * d)
}

fn style_loss(m1: &Tensor, m2: &Tensor) -> Tensor {
    gram_matrix(m1).mse_loss(&gram_matrix(m2), tch::Reduction::Mean)
}

pub fn main() -> Result<()> {
    let opt = Opt::from_args();
    if opt.debug {
        println!("{:#?}", opt);
    }
    let style_img = opt.style;
    let content_img = opt.source;
    let weights = opt.vgg;
    let save_runs = opt.save_runs;
    let device = Device::cuda_if_available();
    let style_path = Path::new(&style_img);
    let content_path = Path::new(&content_img);
    let runs = opt.runs;
    let mut style_list = if style_path.is_dir() {
        style_path
            .read_dir()
            .expect("dir read")
            .map(|file| {
                format!(
                    "{}{}",
                    style_img,
                    file.expect("file").file_name().into_string().unwrap()
                )
            })
            .collect::<Vec<_>>()
    } else {
        vec![style_img.clone()]
    };
    style_list.sort();
    let mut color_array = String::from("");
    if opt.keep_color {
        if opt.debug {
            println!("start color_array ");
        }
        color_array = color_list(&PathBuf::from(&content_img), opt.color_depth);

        if opt.debug {
            println!("done color_array {}", color_array);
        }
    }
    for mut style_img in style_list {
        if opt.keep_color {
            let style_path = Path::new(&style_img);
            let tmp_file = format!(
                "/tmp/{}-{}-recolor.jpg",
                content_path.file_stem().unwrap().to_str().unwrap(),
                style_path.file_stem().unwrap().to_str().unwrap(),
            );
            if opt.debug {
                dbg!(&tmp_file);
            }
            find_colors(
                &style_img,
                &tmp_file,
                color_array.split(',').map(|x| x.to_string()).collect(),
                20,
                3,
            )
            .expect("could not create recolored image");
            if opt.debug {
                println!("done recolor");
            }

            style_img = tmp_file;
        }
        let result = panic::catch_unwind(|| {
            let mut net_vs = tch::nn::VarStore::new(device);
            let net = if weights.contains("19") {
                vgg::vgg19(&net_vs.root(), imagenet::CLASS_COUNT)
            } else {
                vgg::vgg16(&net_vs.root(), imagenet::CLASS_COUNT)
            };
            net_vs
                .load(weights.clone())
                .expect("Could not load weights file");
            net_vs.freeze();
            let style_path = Path::new(&style_img);
            let new_file = format!(
                "{}-{}",
                content_path.file_stem().unwrap().to_str().unwrap(),
                style_path.file_stem().unwrap().to_str().unwrap(),
            );
            dbg!(&new_file, &style_img);
            let style_img = imagenet::load_image(style_img)
                .expect("Could not load style file")
                .unsqueeze(0)
                .to_device(device);
            let content_img = imagenet::load_image(content_img.clone())
                .expect("Could not load content file")
                .unsqueeze(0)
                .to_device(device);
            let max_layer = STYLE_LOWER.iter().max().unwrap() + 1;
            let style_layers = net.forward_all_t(&style_img, false, Some(max_layer));
            let content_layers = net.forward_all_t(&content_img, false, Some(max_layer));

            let vs = nn::VarStore::new(device);
            let input_var = vs.root().var_copy("img", &content_img);
            let mut opt_nn = nn::Adam::default()
                .build(&vs, LEARNING_RATE)
                .expect("learrate");

            for step_idx in 1..(1 + runs) {
                let input_layers = net.forward_all_t(&input_var, false, Some(max_layer));
                let style_loss: Tensor = STYLE_LOWER
                    .iter()
                    .map(|&i| style_loss(&input_layers[i], &style_layers[i]))
                    .sum();
                let content_loss: Tensor = CONTENT_INDEXES
                    .iter()
                    .map(|&i| input_layers[i].mse_loss(&content_layers[i], tch::Reduction::Mean))
                    .sum();
                let loss = style_loss * STYLE_WEIGHT + content_loss;
                opt_nn.backward_step(&loss);
                if step_idx % save_runs == 0 && step_idx <= runs {
                    println!("{} {}", step_idx, f64::from(loss));
                    imagenet::save_image(&input_var, &format!("{}-{}.jpg", new_file, step_idx))
                        .expect("file");
                }
            }

            println!("done ");
            imagenet::save_image(&input_var, &format!("{}-done.jpg", new_file)).expect("file");
        });
        if result.is_err() {
            println!("error");
        }
    }
    Ok(())
}
