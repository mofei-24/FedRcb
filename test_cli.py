from fedrcb.cli import main


def test_smoke_aggregate_and_plot_from_a_saved_configuration(tmp_path):
    source = tmp_path / "small.toml"
    source.write_text("[experiment]\nnum_clients=4\nnum_actions=4\nglobal_dim=2\nlocal_dim=2\nepoch_length=10\ncheckpoint_interval=10\nlibrary_size=12\ngossip_rounds=2\nbit_width=8\nheterogeneity_values=[0.1]\n", encoding="utf-8")
    output = tmp_path / "output"
    assert main(["smoke", "--config", str(source), "--horizon", "20", "--output", str(output)]) == 0
    assert main(["aggregate", "--output", str(output)]) == 0
    assert main(["plot", "--output", str(output)]) == 0
    assert (output / "summary.csv").is_file()
    assert (output / "trajectory.csv").is_file()
    assert (output / "figures/fedrcb.pdf").is_file()
