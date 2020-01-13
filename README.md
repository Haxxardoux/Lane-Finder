# Lane detection algorithm

This project was created in an attempt to learn computer vision techniques, as well as organizational skills. The long-term goal is to evolve into full-stack development and produce an application capable of interacting with a car to navigate on highways (highway autopilot)

# Vision and Methods

### Formatting input 
The pipeline takes video/image feeds in most formats. There is currently code to support displaying video frame-by-frame for debugging, as well as assembling video files.

The pipeline starts by opening a video with Opencv, and using a while loop to process each frame while the video feed is open. 

At the start of the pipeline function process_frame(), the frame is resized (to make computations lighter) 

After the resizing, a bird's-eye-view transformation is applied to a specified region of interest, meant to always contain the lane lines with as few other objects as possible. If there are straight paralell lines in front of the car, the transformation (henceforth warp/warped image) should show perfectly vertical paralell lane lines. An inverse matrix is created and saved for later.

### Thresholding / filtering

Several methods of thresholding were tried, but only 2 yielded sufficiently different results to merit implementation (for example, there is a large overlap between gradient and morphological). Color (HLS) and morphological. 

# Fix later
The main method of filtering is done by the top-hat approach of morphologial thresholding. 


# Contents

## Input-videos
----------
These are videos deemed important for various reasons. More will be added in time. Each contains some "obstacle" that the algorithm will need to account for, be it physical obstacle, visual obstruction, complicated shadow patterns, or other. 

**country_road.mp4** - The camera orientation is very off-center

**obstacle_challenge.mp4** - High-speed driving with dashed lane lines, will require rapid processing to function. Large shadows are also present. In a few frames, shadows cast over the dashed lane lines make them darker than the cars casting the shadows, leading to the algorithm mis-identifying the lower left rim of the car as a lane line (when color thresholding is used). 

    Note - fixed with morphological thresholding

**harder_challenge_video.mp4** - Significant curvature, and lots of shadows pose issues. Also, a change of speed is necessary for a sharp turn, which will need to be addressed in later versions of this program.

**shadow_challenge.mp4** - A narrow bridge casts a shadow that completely envelops the car- but only for a short time. This poses an issue because the camera makes a rapid adjustment to let in more light, but since the shadow is narrow, the camera lets in far too much light after passing through the shadow, leaving the camera "blind" for several frames.

    Note- fixed with line class, "keep lines constant" if not detected for a few frames
        Note on note - add failsafe - require driver takeover if lane not detected after x frames. 

## Output-videos
----------

Output of input-videos with full debugging information

## lanedetection.py
----------

    The main pipeline!

### Function documentation / explanation


**morphology_filter(img)** - 

    Input - low resolution (720) image

    Output - binary image - open morphological transformation

    This function takes an image and converts it to a combination of saturation/grayscale (30%/70%). This idea was taken from a [Cambridge professor/course github page](https://github.com/balancap). This makes the lane lines more visible for the next step.

    The next step is to do an "open" morphological transformation. This is an erosion followed by dilation, which removes lots of noise. 

    A cutoff threshold is established based on the mean and standard deviation of the filtered data, which determines the binary image output. 

**img_threshold(img)**

    Input - low resolution (720) image

    Output - binary image - based on lightness and saturation threshold

    This function with STATIC parameters is the main "color" filter used to help distinguish between things shaped like lane lines, and actual lane lines.

**apply_color_threshold(img)** -

    Input - low resolution (720) image

    Output - binary image - DYNAMIC lightness/contrast filter
    
    DYNAMIC based on how many points are caught by the color filter. More points means the filter parameters are increased to take less points in the next frame. This is useful if there are lots of bright objects in the frame that need to be ignored.

**warp(img)** -
    
    Input - low resolution (720) image

    Implicit input - region of interest coordinates

    Output - bird's eye view of the region of interest, inverse matrix used to compute perspective transforms

    This function takes 4 points, which is the region of interest in the input image. It then computes the perspective transform to be used later on. It is important that when the transformation is done, lane lines that are straight and paralell in the real world are paralell in the transformation. They should not be at equal and opposite angles. 


**slide_window(binary_img)** - 

    This function was copied from another github page

    Input - binary image (output of filters) that been transformed, bird's eye view of the region of interest.

    Output - Point indicies of the right and left lane, as well as the warped image with polynomial lines drawn on. 

**margin_search(binary_img)** - 

    This function was copied from another github page

    Input - binary image (output of filters) that been transformed, bird's eye view of the region of interest.

    Implicit input - Margin (# of pixels)

    Output - Point indicies of the right and left lane, as well as the warped image with polynomial lines drawn on. 

    This is a more efficient version of the slide_window method for finding the lanes given the filtered image. It is applied when a lane is detected with a certain level of confidence, and searches within a margin of x pixels to the right and left of the detected lane. Significant advantage to processing power.

**assemble_img(warped_img, threshold_img, warped_with_lanes_img, output_img)** - 

    Input - The warped image, the warped image after the thresholds have been applied, the warped images after thresholds have been applied and lanes drawn, and the original image with the lane lines drawn and lane lines filled in. 

    Output - 4 images in one, along with useful information for debugging.

    This is meant entirely for debugging.

**process_frame(img)**  - 

    Main "pipeline" function.

    Input - Any image, full res / uncropped. 

    Parameters - Color threshold cuttoffs- use color threshold along with morphological if color filter yields between xx,xxx - xx,xxx points. Additionally, the amount that the thresholds should increase/decrease when the number of points read is outside the bounds. 

    Output - Either debugging or original image with lane filled in 

    This is the only function used by the main pipeline for processing images, and eventually will need to include electronic signal outputs for the car to read.

### Classes
-----------

**Line()** - 

    The line class stores information about the lanes which is referenced by functions like *validate_lane_assumptions* and *process_frame* to optimize performance and tune parameters. 

    Note - Compare performance with global variables. This will definitely be simpler though.


