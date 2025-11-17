
# 1. Rendering a Scene

## Docker Set Up

```
docker run -it \
    --name kubric_eb \
    -v "/home/cvlab21/project/eunbeen/kubric:/kubric" \
    -v "/mnt/nvme1n1/event_data:/datasets" \
    kubricdockerhub/kubruntu bash
```

```
export PYTHONPATH=/kubric:$PYTHONPATH
pip install -e .
```


## Run Rendering

Use `fixed_init_rendering.py` to control initial randomness of the scene. Initial scene can be set with `--metadata`.
WARNING!! This doesnt mean the rendered scene will be the exact replica. Due to some undeterministic simualation, the final result may differ.

```
docker exec -it kubric_eb bash
python potato/scripts/fixed_init_rendering.py
```


Use `multi_rendering.py` to generate the exact same scene, with different rendering settings (e.g. motion blur). This script simulates once then render twice to generate a duplicate scene.


```
# Uncomment the following code to render both blurred and unblurred versions. 
    render(**out_dict, blur=False)
    # render(**out_dict, blur=True)
```

```
docker exec -it kubric_eb bash
python potato/scripts/multi_rendering.py
```

Alternatively, run `./potato/quack.sh`


# 2. Generating a track and Visualize

## Use Conda Env

```
conda activate kubric_track_eb
./potato/scripts/track_n_vis.sh 
```

`FIXME`: Using PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python is slow, solve protobuf version error.


# 3. Docker clean up

The rendering doesn't clean its assigned memory.
Stop the docker and remove cache manually after rendering.

```
docker stats
docker stop kubric_eb
docker system prune
```