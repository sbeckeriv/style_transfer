use anyhow::Result;
use image::{imageops, GenericImageView, ImageBuffer, RgbImage};
use kmeans_colors::{
    get_kmeans, get_kmeans_hamerly, Calculate, CentroidData, Kmeans, MapColor, Sort,
};
use palette::{Lab, Pixel, Srgb, Srgba};
use std::error::Error;
use std::fmt::Write;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::path::PathBuf;
use tch::Tensor;

const SPLIT_SIZE: u32 = 600;
pub fn paste(file: &str, base: &str) {
    let mut img = image::open(base).unwrap();
    let dimensions = img.dimensions();
    let size = SPLIT_SIZE;
    let xs = (dimensions.0 / size) + 1;
    let ys = (dimensions.1 / size) + 1;
    for x in 0..xs {
        for y in 0..ys {
            let part = &format!("{}-{}-{}", x, y, file);
            dbg!(part);
            let mut on_top = image::open(part).unwrap();
            image::imageops::overlay(&mut img, &on_top, x * size, y * size);
        }
    }
    img.save(format!("final-{}", file));
}

pub fn save_crops(file: &str) -> (u32, u32) {
    let size = SPLIT_SIZE;
    let mut img = image::open(file).unwrap();
    let dimensions = img.dimensions();
    let xs = (dimensions.0 / size) + 1;
    let ys = (dimensions.1 / size) + 1;
    for x in 0..xs {
        for y in 0..ys {
            let split_out = format!("{}-{}-{}", x, y, file);
            if Path::new(&split_out).exists() {
                continue;
            }
            let subimg = imageops::crop(&mut img, x * size, y * size, size, size);
            subimg.to_image().save(&split_out);
        }
    }
    (xs, ys)
}

pub fn resize(style_file: &str, x: u32, y: u32) {
    let split_out = format!("{}-{}-{}", x, y, style_file);
    if !Path::new(&split_out).exists() {
        let mut img_style = image::open(style_file).unwrap();
        let mut img_style =
            imageops::resize(&img_style, x, y, image::imageops::FilterType::Lanczos3);
        img_style.save(split_out);
    } else {
    }
}

pub fn save_crops_style(content_file: &str, style_file: &str) {
    let size = SPLIT_SIZE;
    let mut img_content = image::open(content_file).unwrap();
    let mut img_style = image::open(style_file).unwrap();
    let dimensions = img_content.dimensions();
    let mut img_style = imageops::resize(
        &img_style,
        dimensions.0,
        dimensions.1,
        image::imageops::FilterType::Lanczos3,
    );
    let xs = (dimensions.0 / size) + 1;
    let ys = (dimensions.1 / size) + 1;
    for x in 0..xs {
        for y in 0..ys {
            let split_out = format!("{}-{}-{}", x, y, style_file);
            if Path::new(&split_out).exists() {
                dbg!(split_out);
                continue;
            }
            let subimg = imageops::crop(&mut img_style, x * size, y * size, size, size);
            subimg.to_image().save(&split_out);
        }
    }
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
            image::codecs::png::CompressionType::Best,
            image::codecs::png::FilterType::NoFilter,
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

pub fn gram_matrix(m: &Tensor) -> Tensor {
    let (a, b, c, d) = m.size4().unwrap();
    let m = m.view([a * b, c * d]);
    let g = m.matmul(&m.tr());
    g / (a * b * c * d)
}

pub fn style_loss(m1: &Tensor, m2: &Tensor) -> Tensor {
    gram_matrix(m1).mse_loss(&gram_matrix(m2), tch::Reduction::Mean)
}
