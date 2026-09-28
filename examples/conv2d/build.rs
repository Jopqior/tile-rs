fn main() -> Result<(), Box<dyn std::error::Error>> {
    tile_kernel_builder::add_ascend_link_args()?;
    tile_kernel_builder::build_operator("test_data/config/conv2d.json", "test_data/op_models")?;
    Ok(())
}
