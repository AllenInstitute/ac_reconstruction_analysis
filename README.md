# ac_reconstruction_analysis

# Alignment module
## Skeleton-based inter-sectional alignment
acanalysis.acalignment
- Generate endpoint correspondences from surface-contacting skeletons via iterative geometric RANSAC to solve for nonrigid alignment transformation and detected inlier set of connections



## Skeleton reconstruction

***Note***: These methods have been optimized to run on multiple regions of interest (ROIs) concurrently in high performance computing (HPC) environments, accounting for their standalone scripts.


**Find Skeleton Pairs to Reconnect**

***Note***: There are two detection methods: the first relies on an arbitrarily defined minimum collinearity and maximum Euclidian distance between skeletons; the second relies on a pre-trained classifier. The classifier requires model and scalar files, both of which can be found in acanalysis/test/model. This difference is defined by the "method" parameter across all methods.
```
python acanalysis/skeleton_reconstruction/reconnection/find_pairs.py \
--skels DIRECTORY_TO_INPUT_SKELETON_VOLUME \
--out_file PATH_TO_OUTPUT_PAIRS_FILES \
--cl PATH_TO_MODEL_FILE \
--sc PATH_TO_SCALAR_FILE \
--min_collin MINIMUM_PAIR_COLLINEARITY \
--method PAIR_FINDING_METHOD
```


**Build Connected Components from Pairs**
```
python acanalysis/skeleton_reconstruction/reconnection/create_components.py \
--pair_files PATH_TO_PAIRS_FILE or DIRECTORY_TO_PAIR_FILES \
--components_per_file NUMBER_CONNECTED_COMPONENTS_PER_FILE \
--method PAIR_FINDING_METHOD \
```


**Merge Skeletons**
```
python acanalysis/skeleton_reconstruction/reconnection/merge_pairs.py \
--skels DIRECTORY_TO_INPUT_SKELETON_VOLUME  \
--pair_file PATH_TO_COMPONENTS_FILE \
--prob_thresh  MINIMUM_PROBABILITY_THRESHOLD *using pre-trained model \
--method PAIR_FINDING_METHOD
```
