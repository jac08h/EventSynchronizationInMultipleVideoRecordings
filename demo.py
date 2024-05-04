# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


from pathlib import Path

from tests.config.config_wrapper import ConfigWrapper
from tests.test_utils import plot_synchronization, visualize_synchronized_clips
from video_snapping.videosnapping import synchronize_clips

if __name__ == '__main__':
    config = ConfigWrapper(Path("tests/config.yaml"))
    config.output_directory.mkdir(exist_ok=True, parents=True)
    for cost_matrix in config.cost_matrices():
        predicted, calculated_cost_matrix = synchronize_clips(cost_matrix,
                                                              config.clip_0(),
                                                              config.clip_1(),
                                                              config.max_size(),
                                                              config.fps(),
                                                              config.initialize_dijkstra(),
                                                              config.cache_directory() if config.use_cache() else None
                                                              )
        if config.save_plots():
            graph_path = config.output_directory / f"{cost_matrix.get_feature_name()}.{config.plot_ext()}"
            plot_synchronization([], predicted, calculated_cost_matrix, graph_path, colormap=config.colormap())

        visualize_synchronized_clips(cost_matrix,
                                     config.visualization(),
                                     cost_matrix.get_feature_name(),
                                     config.output_directory,
                                     config.clip_0(),
                                     config.clip_1(),
                                     predicted,
                                     config.max_size(),
                                     config.clip_ext()
                                     )
