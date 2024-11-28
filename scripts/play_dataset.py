from lotus.libero import benchmark, get_libero_path
import os

import h5py
from lotus.libero.utils.dataset_utils import get_dataset_info
import imageio

benchmark_name = "libero_90"

datasets_default_path = get_libero_path("datasets")
benchmark_instance = benchmark.get_benchmark_dict()[benchmark_name]()
num_tasks = benchmark_instance.get_num_tasks()


for i in range(num_tasks):

    task_name = benchmark_instance.get_task_demonstration(i)

    h5f_path = os.path.join(datasets_default_path, task_name)

    task_folder = os.path.join("video", os.path.splitext(task_name)[0])
    os.makedirs(task_folder, exist_ok=True)

    get_dataset_info(h5f_path)

    with h5py.File(h5f_path, "r") as f:
        data_group = f["data"]
        num_demos = len(data_group.keys())  # Number of demos
        images = {}
        
        for demo_key in data_group.keys():
            images[demo_key] = data_group[f"{demo_key}/obs/agentview_rgb"][()]

    for demo_key in images:
        video_path = os.path.join(task_folder, f"{demo_key}.mp4")
        video_writer = imageio.get_writer(video_path, fps=60)
        for image in images[demo_key]:
            video_writer.append_data(image[::-1])
        video_writer.close()


