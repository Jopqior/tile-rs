fn main() -> Result<(), Box<dyn std::error::Error>> {
    // FIXME: lib cannot be compiled separately without this call.
    tile_kernel_builder_config::add_ascend_link_args()?;
    Ok(())
}
