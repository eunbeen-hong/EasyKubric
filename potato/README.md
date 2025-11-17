
# 1. Rendering a Scene

## Docker Set Up

```
docker run -it \
    --name kubric_eb \
    -v "/home/cvlab15/project/eunbeen/4d-recon/kubric_event:/kubric" \
    -v "/mnt/data6/eunbeen/4d-recon:/datasets" \
    kubricdockerhub/kubruntu bash
```

```
export PYTHONPATH=/kubric:$PYTHONPATH
pip install -e .
```


## Run Rendering

### Base Code (ignore)

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

### 4d Recon

Use `render_4d_recon.sh` with `./potato/configs/4d_recon.json` to generate N frames with N fixed cameras (N*N scenes). 

Easiest way is to modify `./potato/quack.sh`

### RGB Tracking

Use `render_rgb_track.sh` with `rgb_track.json` to generate one (normal) scene.

Easiest way is to modify `./potato/quack.sh`


# 2. Generating a track and Visualize

## Use Conda Env

No docker is needed. Setup conda env with `kubric_track_eb.yml`.

```
conda activate kubric_track_eb
./potato/scripts/track_n_vis.sh 
```

or, run following for only tracking (no vis)

```
conda activate kubric_track_eb
./potato/scripts/trackling.sh 
```

`FIXME`: Current code generate tracks from random queries. Modify to compute all pixel-wise tracks

`FIXME`: Using PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python is slow, solve protobuf version error.


# 3. Docker clean up

The rendering doesn't clean its assigned memory.
Stop the docker and remove cache manually after rendering.

```
docker stats
docker stop kubric_eb
docker system prune
```