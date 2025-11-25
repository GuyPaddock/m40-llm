fn main() {
    let ctx = m40_llm::cuda::CudaContext::new(-1).unwrap();
    let p = ctx.current_device_props().unwrap();
    println!(
        "name={} major={} minor={} device_id={}",
        p.name, p.major, p.minor, p.device_id
    );
}
