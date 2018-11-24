# target-tracking
The project aims to track multiple targets through a CCTV network. The end goal of the project is to plot the tracks of people in the environment.

#### Modules
- `MobileNet` for object detections. 
- `monodepth` to get a sense of depth in an image.

#### monodepth Depth Sensing
Default Image             |  Depth Map
:-------------------------:|:-------------------------:
<img src="https://github.com/shreyasnbhat/target-tracking/blob/master/images/actual.jpg" alt="drawing" width="300"/>  |  <img src="https://github.com/shreyasnbhat/target-tracking/blob/master/images/depth_map.png" alt="drawing" width="300"/>

#### Pipeline Working Example
<img src="https://github.com/shreyasnbhat/target-tracking/blob/master/test.gif" width="512" height="256" />
Lower numbers signify lesser depth with respect to the camera. Lower number directly correspond to brighter spots in the depth map.
