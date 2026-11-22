//! Export real down projections for the standalone Maxwell study.
use anyhow::{ensure, Context, Result};
use clap::Parser;
use m40_llm::{
    gguf::{load_gguf, GgmlDType},
    infer::q4_channels::Q4ChannelWeights,
};
use serde_json::json;
use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
    time::Instant,
};

const INPUT: usize = 11008;
const OUTPUT: usize = 2048;

#[derive(Parser)]
struct Arguments {
    model: PathBuf,
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Arguments::parse();
    let gguf = load_gguf(&args.model)?;
    let mut file = File::open(&args.model)?;
    fs::create_dir(&args.output).context("specify a new fixture directory")?;
    for layer in [0, 17, 35] {
        let name = format!("blk.{layer}.ffn_down.weight");
        let tensor = gguf
            .tensors
            .iter()
            .find(|t| t.name == name)
            .context("down tensor missing")?;
        if tensor.dtype != GgmlDType::Q4_0 || tensor.shape != [INPUT as u64, OUTPUT as u64] {
            println!(
                "{}",
                json!({"layer":layer,"name":name,"status":"unsupported","dtype":tensor.dtype,"shape":tensor.shape})
            );
            continue;
        }
        let offset = gguf
            .data_offset
            .checked_add(tensor.offset)
            .context("offset overflow")?;
        file.seek(SeekFrom::Start(offset))?;
        let mut native = vec![0; OUTPUT * INPUT / 32 * 18];
        file.read_exact(&mut native)?;
        let prepare = Instant::now();
        let packed = Q4ChannelWeights::pack(&native, INPUT, OUTPUT)?;
        let packed_ms = prepare.elapsed().as_secs_f64() * 1000.0;
        ensure!(
            packed.unpack() == native,
            "lossless layout round trip failed"
        );
        let prepare = Instant::now();
        let (alternate, rms, max) = packed.repack_zigzag()?;
        let zigzag_ms = prepare.elapsed().as_secs_f64() * 1000.0;
        let folder = args.output.join(layer.to_string());
        fs::create_dir(&folder)?;
        let [scales, codes] = packed.planes();
        for (filename, bytes) in [
            ("native", native.as_slice()),
            ("scales", scales),
            ("codes", codes),
            ("zigzag", &alternate),
        ] {
            fs::write(folder.join(filename), bytes)?;
        }
        println!(
            "{}",
            json!({"layer":layer,"name":name,"status":"prepared","dtype":tensor.dtype,
            "shape":tensor.shape,"offset":offset,"native_bytes":native.len(),"packed_bytes":packed.bytes(),
            "zigzag_bytes":alternate.len(),"packed_ms":packed_ms,"zigzag_from_packed_ms":zigzag_ms,
            "native_to_zigzag_weight_rmse":rms,"native_to_zigzag_weight_max_abs":max})
        );
    }
    Ok(())
}
