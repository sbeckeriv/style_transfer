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

use anyhow::Result;
use std::fs;
use std::panic;
use std::path::{Path, PathBuf};
use structopt::StructOpt;
use tch::vision::{imagenet, vgg};
use tch::{nn, nn::OptimizerConfig, Device, Tensor};

use image::{imageops, GenericImageView, ImageBuffer, RgbImage};
mod util;
use util::*;

const STYLE_WEIGHT: f64 = 1e12;
const LEARNING_RATE: f64 = 1e-1;

const STYLE_LOW: [usize; 5] = [0, 2, 5, 7, 10];
const STYLE_LOWER: [usize; 8] = [0, 2, 5, 7, 10, 13, 16, 19];
const STYLE_TOLOW: [usize; 12] = [0, 2, 5, 7, 10, 13, 16, 19, 21, 24, 27, 30];
const CONTENT_INDEXES: [usize; 2] = [7, 10];
//const CONTENT_INDEXES: [usize; 1] = [7];

#[derive(StructOpt, Debug)]
pub struct Opt {
    #[structopt(long = "splitsize", default_value = "800")]
    split_size: u32,
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

pub fn main() -> Result<()> {
    let opt = Opt::from_args();
    if opt.debug {
        println!("{:#?}", opt);
        println!("Cuda available: {}", tch::Cuda::is_available());
    }
    let style_img = opt.style;
    let split_size = opt.split_size;
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
        let original_style_image = style_img.clone();
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
        let debug = opt.debug;

        let counts = save_crops(&content_img, split_size);
        //save_crops_style(&content_img, &style_img, split_size);

        let new_file = format!(
            "{}-{}",
            content_path.file_stem().unwrap().to_str().unwrap(),
            style_path.file_stem().unwrap().to_str().unwrap(),
        );
        for x in 0..counts.0 {
            for y in 0..counts.1 {
                let split_out = format!("{}-{}-{}-done.jpg", x, y, new_file);
                if Path::new(&split_out).exists() {
                    dbg!(split_out);
                    continue;
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
                    let split_content = format!("{}-{}-{}", x, y, content_img);

                    let mut img = image::open(&split_content).unwrap();
                    let dimensions = img.dimensions();
                    resize(&style_img, dimensions.0, dimensions.1);
                    let split_style = format!("{}-{}-{}", dimensions.0, dimensions.1, style_img);

                    if debug {
                        dbg!(&new_file, &style_img);
                    }
                    let style_image_net = imagenet::load_image(split_style.clone())
                        .expect("Could not load style file")
                        .unsqueeze(0)
                        .to_device(device);
                    let content_img = imagenet::load_image(split_content.clone())
                        .expect("Could not load content file")
                        .unsqueeze(0)
                        .to_device(device);
                    let max_layer = STYLE_LOWER.iter().max().unwrap() + 1;
                    let style_layers = net.forward_all_t(&style_image_net, false, Some(max_layer));
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
                            .map(|&i| {
                                input_layers[i].mse_loss(&content_layers[i], tch::Reduction::Mean)
                            })
                            .sum();
                        let loss = style_loss * STYLE_WEIGHT + content_loss;
                        opt_nn.backward_step(&loss);
                        if step_idx % save_runs == 0 && step_idx <= runs {
                            println!("{} {}", step_idx, f64::from(loss));
                            imagenet::save_image(
                                &input_var,
                                &format!("{}-{}-{}-{}.jpg", x, y, new_file, step_idx),
                            )
                            .expect("file");
                        }
                    }

                    println!("done ");
                    imagenet::save_image(&input_var, &split_out).expect("file");
                });
                if result.is_err() {
                    println!("error");
                }
                if original_style_image != style_img {
                    //fs::remove_file(style_img).expect("Could not delete style image");
                }
            }
        }

        paste(&format!("{}-done.jpg", new_file), &content_img, split_size);
    }
    Ok(())
}
