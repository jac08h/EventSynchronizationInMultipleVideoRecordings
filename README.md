# Event Synchronization in Multiple Video Recordings

A code for my [bachelor's thesis](https://is.muni.cz/th/cyoc1/thesis.pdf), which implements [VideoSnapping method](https://studios.disneyresearch.com/wp-content/uploads/2019/03/VideoSnapping-Interactive-Synchronization-of-Multiple-Videos-1.pdf) for video synchronization with support for various descriptors to detect image features and evaluates it on synchronization of soccer videos.

## Example output
Synchronization of two clips from [SoccerNet](https://www.soccer-net.org/):

https://github.com/user-attachments/assets/4193918a-f4b3-431c-90fc-7e4567154a12

After the ~37th frame, the videos are synchronized with only a small error:

![cost matrix visualization](https://github.com/user-attachments/assets/9d8e65b0-8224-4568-b901-94fd5cb33d67).

(While manually selected, this example is representative of the method's accuracy on the videos with similar level of visual disparity. Cost matrices for all videos in the test dataset can be found in the thesis' appendix.)

## Supported descriptors

* OpenCV: SIFT, ORB, KAZE, BRISK, BoostDesc, DAISY, FREAK, LATCH, VGG
* Kornia: MKDDescriptor, TFeat, HardNet, HardNet8, HyNet, SOSNet
* Third-party: D2Net, SuperPoint, R2D2

## Installation

* Install requirements:
    * `conda env create -f conda.yaml`
* Install git submodules `git submodule update --init`
* Apply patch to D2Net code
  * `cd third_party/d2net`
  * `git apply ../d2net_patch.diff`
  * After applying the patch, this should be the code in `third_party/d2net/lib/pyramid.py`:
```
...
86: ids = ids.cpu()  # <-- Added line.
87: fmap_pos = fmap_pos[:, ids]  
88: fmap_keypoints = fmap_keypoints[:, ids]
...
```

## Data
### Download SoccerNet videos and annotations
* To download the required SoccerNet data, follow the [instructions](https://www.soccer-net.org/data)
* Note that to download the videos, you have to fill in a linked [NDA form](https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform).
* Initialize the downloader:
```python
from SoccerNet.Downloader import SoccerNetDownloader 
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="path/to/soccernet")
mySoccerNetDownloader.password = input("Password for videos (received after filling the NDA)")
```
* And download videos, action and replay annotations and camera annotations:
  * Videos: `mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["train","valid","test","challenge"])`
  * Action and replay annotations: `mySoccerNetDownloader.downloadGames(files=["Labels-v3.json"], split=["train","valid","test"])`
  * Camera annotations: `mySoccerNetDownloader.downloadGames(files=["Labels-cameras.json"], split=["train","valid","test"])`

### Cut clips of actions and replays
* Run `data_preparation/cut_action_and_replay_videos.py [-h] soccernet_dir clips_dir`
```
soccernet_dir  Path to soccernet directory.
clips_dir      Path to directory to store action and replay videos.
```

### Create augmented videos
* Run `data_preparation/create_augmented_clips.py [-h] augmentations_annotations_file clips_dir augmented_clips_dir`
```
augmentations_annotations_file      Path to augmentation annotations.
clips_dir                           Path to directory containing action and replay clips.
augmented_clips_dir                 Path to directory to store augmented and copied clips.
```

### Annotations format

* The annotation files follow nested structure from SoccerNet: competition -> season -> match.
* Different annotations are needed based on if the videos were augmented or manually annotated.
* Augmented:
```
CLIP_ID: {
    "start_shift": Time shift on the clip start,
    "end_shift": Time shift on the clip end,
    "speed_augmentations": [
        {
            "start": Start of the part with augmented speed,
            "speed_change": How much the speed was increased,
            "end": End of the part with augmented speed,
        },
        {...}
    ]
}
```
* Annotated:
```
"action": ID of the first clip,
"replay": ID of the second clip,
"corresponding_timestamps": [
    [
        Time in the first clip,
        Corresponding time in the second clip
    ],
    [...]
]
```

## Evaluation

* To run evaluation on the dataset run `eval.py`, which uses configuration from `tests/config.yaml`. More information
  about configuration can be found in `tests/config_documentation.md`.
* Example output:

 ```
                   easy                          hard                    
                      1         3         5         1         3         5
SIFT           0.269255  0.376331  0.422668  0.118360  0.186490  0.243649
D2Net          0.359424  0.504696  0.581090  0.076212  0.189376  0.233834
HardNet8       0.413901  0.519724  0.557295  0.046189  0.117783  0.156467
```

* Results display ratio of frames which were synchronized accurately.
* Accurate synchronization is calculated with regard to the first video of the video pair. For each frame of the first
  video, was the correct frame of the second video assigned in the synchronization? The allowed error rate specifies the
  allowed difference between assigned and correct frame to be still considered as correctly assigned. Error rate 1 means
  that the prediction which is off by 1 frame is still correct.
* In the example output, 42.26% of frames of the first videos from the easy datset were synchronized correctly with an
  allowed error rate of 5 frames.

## Demo

* To run the algorithm only on two videos where no ground truth is available, use `demo.py`. The videos pair is
  specified in `test/config.yaml` in the `demo` section. Other configuration settings are
  set in the config as well.

## License
The code in the repository is licensed under GNU Lesser General Public License.
