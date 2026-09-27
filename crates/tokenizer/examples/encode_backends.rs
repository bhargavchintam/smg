//! Encode-backend benchmark: compares SMG's tokenizer backends with fastokens.
//!
//! Run it with `--release` so it is built with the profile SMG ships.
#![expect(clippy::print_stdout, reason = "benchmark CLI prints its results")]

#[global_allocator]
static GLOBAL_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("version") => {
            println!(
                "encode_backends: fastokens {}",
                std::any::type_name::<fastokens::Tokenizer>()
            );
            Ok(())
        }
        _ => anyhow::bail!("usage: encode_backends <render|parity|time|throughput|load> ..."),
    }
}
