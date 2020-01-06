import numpy as np
import cv2
import collections


def morphology_filter(img_):
    gray = cv2.cvtColor(img_, cv2.COLOR_RGB2GRAY)
    # Saturation channel
    hls_s = cv2.cvtColor(img_, cv2.COLOR_RGB2HLS)[:, :, 2]
    # copied from github, works like a charm
    src = 0.3*hls_s + 0.7*gray
    src = np.array(src-np.min(src)/(np.max(src)-np.min(src))).astype('float32')
    blurf = np.zeros((1, 5))
    blurf.fill(1)
    src = cv2.filter2D(src, cv2.CV_32F, blurf)
    f = np.zeros((1, 30))
    f.fill(1)
    l = cv2.morphologyEx(src, cv2.MORPH_OPEN, f)
    # l is our src image with an opening morphological transformation applied to it. Anything smaller than some value in kernel will be removed. This means smaller abnormalities like lane lines are removed, and nautral gradients like shadows are not. Subtracting them yields the removed items. This is the top hat morphological transformation.
    filtered = src - l
    # 6x6 kernel

    morph_thresh_lower = 1.1*np.mean(filtered) + .7*np.std(filtered)
    morph_binary = np.zeros_like(filtered)
    morph_binary[(filtered >= morph_thresh_lower)] = 1
    # Remove noise with 6x6 kernel
   # morph_binary = cv2.morphologyEx(morph_binary, cv2.MORPH_OPEN, small_kernel)
    morph_binary=morph_binary.astype(np.float32)

    return morph_binary
def img_threshold(img_):

    # Use HLS Channels for color Thresholding 
    thresh_l = (150, 255)
    hls = img_.astype(np.float)
    # Lightness channel
    L = hls[:, :, 2]
    # Saturation channel
    S = hls[:, :, 1]
    color_binary = np.zeros_like(L)
    # Saturation threshold has lower bound than lightness threshold
    color_binary[(L > thresh_l[0]) & (L <= thresh_l[1]) & (S > thresh_l[0]-20) & (S <= thresh_l[1])] = 1
    color_binary = color_binary.astype(np.float32)
    
    # Combine thresholds
    morph_binary = morphology_filter(img_)
    img_out = cv2.bitwise_and(morph_binary, color_binary)

    return img_out.astype(np.float32)

def apply_color_threshold(image):
    thresh_s = (170, 255)
    thresh_l = (150, 255)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
    hls = image.astype(np.float)
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    channel_S = np.zeros_like(S)
    channel_S[(S > thresh_s[0]) & (S <= thresh_s[1])] = 1
    channel_L = np.zeros_like(L)
    channel_L[(L > thresh_l[0]) & (L <= thresh_l[1])] = 1
    binary = np.maximum(channel_L,channel_S)

    #binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, small_kernel)
    
    return binary

imshape=(1920//2, 1080//2)
roi_vertices = np.array([[575,350],[900,530],[50,530],[375,350]], dtype=np.int32)
pts=np.float32([[375,350],[575,350],[50,530],[900,530]])
pts2=np.float32([[250,0],[960,0],[250,540],[960,540]])

small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 22))

def warp(img):
    matrix = cv2.getPerspectiveTransform(pts,pts2)
    result = cv2.warpPerspective(img,matrix, (960,540))
    Minv = cv2.getPerspectiveTransform(pts2,pts)
    return result, Minv
def slide_window(binary_warped):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = np.int(binary_warped.shape[0]/nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 50
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2)
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if len(leftx)<500 or len(lefty)<500 or len(rightx)<500 or len(righty)<500:
        pass
    else:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        window_img = np.zeros_like(out_img)
        left_line.points = len(leftx)
        right_line.points = len(rightx)

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.intc([left_line_pts]), (0,255,0))
        cv2.fillPoly(window_img, np.intc([right_line_pts]), (0,255,0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [1, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 1]

        # Draw polyline on image
        right = np.asarray(tuple(zip(right_fitx, ploty)), np.int32)
        left = np.asarray(tuple(zip(left_fitx, ploty)), np.int32)
        cv2.polylines(out_img, [right], False, (1,1,0), thickness=5)
        cv2.polylines(out_img, [left], False, (1,1,0), thickness=5)

    return left_lane_inds, right_lane_inds, out_img
def margin_search(binary_warped):
    # Performs window search on subsequent frame, given previous frame.
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 50

    left_lane_inds = ((nonzerox > (left_line.current_fit[0]*(nonzeroy**2) + left_line.current_fit[1]*nonzeroy + left_line.current_fit[2] - margin)) & (nonzerox < (left_line.current_fit[0]*(nonzeroy**2) + left_line.current_fit[1]*nonzeroy + left_line.current_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_line.current_fit[0]*(nonzeroy**2) + right_line.current_fit[1]*nonzeroy + right_line.current_fit[2] - margin)) & (nonzerox < (right_line.current_fit[0]*(nonzeroy**2) + right_line.current_fit[1]*nonzeroy + right_line.current_fit[2] + margin)))
    left_line.points = left_lane_inds.sum()
    right_line.points = right_lane_inds.sum()

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Generate a blank image to draw on
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Create an image to draw on and an image to show the selection window
    window_img = np.zeros_like(out_img)

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.intc([left_line_pts]), (0,255,0))
    cv2.fillPoly(window_img, np.intc([right_line_pts]), (0,255,0))
    out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [1, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 1]

    # Draw polyline on image
    right = np.asarray(tuple(zip(right_fitx, ploty)), np.int32)
    left = np.asarray(tuple(zip(left_fitx, ploty)), np.int32)
    cv2.polylines(out_img, [right], False, (1,1,0), thickness=5)
    cv2.polylines(out_img, [left], False, (1,1,0), thickness=5)

    return left_lane_inds, right_lane_inds, out_img
def draw_lane(undist, img, Minv):
    # Generate x and y values for plotting
    ploty = np.linspace(0, undist.shape[0] - 1, undist.shape[0])
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img).astype(np.uint8)
    color_warp = np.stack((warp_zero, warp_zero, warp_zero), axis=-1)

    left_fit = left_line.best_fit
    right_fit = right_line.best_fit

    if left_fit is not None and right_fit is not None:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (64, 224, 208))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))

        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        return result
    return undist
def validate_lane_update(img, left_lane_inds, right_lane_inds):
    # Checks if detected lanes are good enough before updating
    img_size = (img.shape[1], img.shape[0])

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Extract left and right line pixel positions
    left_line_allx = nonzerox[left_lane_inds]
    left_line_ally = nonzeroy[left_lane_inds]
    right_line_allx = nonzerox[right_lane_inds]
    right_line_ally = nonzeroy[right_lane_inds]
    print('lane pts: ', len(left_line_allx)+len(right_line_allx))

    # Discard lane detections that have very little points,
    # as they tend to have unstable results in most cases
    if len(left_line_allx) <= 1000 or len(right_line_allx) <= 1000:
        left_line.detected = False
        right_line.detected = False
        return

    left_x_mean = np.mean(left_line_allx, axis=0)
    right_x_mean = np.mean(right_line_allx, axis=0)
    lane_width = np.subtract(right_x_mean, left_x_mean)

    # Discard the detections if lanes are not in their repective half of their screens
    if left_x_mean > 740 or right_x_mean < 740:
        left_line.detected = False
        right_line.detected = False
        return

    # Discard the detections if the lane width is too large or too small
    if  lane_width < 300 or lane_width > 800:
        left_line.detected = False
        right_line.detected = False
        return

    # If this is the first detection or
    # the detection is within the margin of the averaged n last lines
    if left_line.bestx is None or np.abs(np.subtract(left_line.bestx, np.mean(left_line_allx, axis=0))) < 100:
        left_line.update_lane(left_line_ally, left_line_allx)
        left_line.detected = True
    else:
        left_line.detected = False
    if right_line.bestx is None or np.abs(np.subtract(right_line.bestx, np.mean(right_line_allx, axis=0))) < 100:
        right_line.update_lane(right_line_ally, right_line_allx)
        right_line.detected = True
    else:
        right_line.detected = False
def assemble_img(warped, threshold_img, polynomial_img, lane_img):
    # Define output image
    # Main image
    img_out=np.zeros((540,1388,3), dtype=np.uint8)
    img_out[0:540,0:960,:] = lane_img

    # Text formatting
    fontScale=1
    thickness=1
    fontFace = cv2.FONT_HERSHEY_PLAIN

    # Perspective transform image
    img_out[0:180,961:1388,:] = cv2.resize(warped,(427,180))
    boxsize, _ = cv2.getTextSize("Transformed", fontFace, fontScale, thickness)
    cv2.putText(img_out, "Transformed", (int(1494-boxsize[0]/2),40), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)

    # Threshold image
    resized = cv2.resize(threshold_img,(427,180))
    resized=np.uint8(resized)
    gray_image = cv2.cvtColor(resized*255,cv2.COLOR_GRAY2RGB)
    img_out[181:361,961:1388,:] = cv2.resize(gray_image,(427,180))
    boxsize, _ = cv2.getTextSize("Filtered", fontFace, fontScale, thickness)
    cv2.putText(img_out, "Filtered", (int(1494-boxsize[0]/2),281), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)

    # Polynomial lines
    img_out[360:540,961:1388,:] = cv2.resize(polynomial_img*255,(427,180))
    boxsize, _ = cv2.getTextSize("Detected Lanes", fontFace, fontScale, thickness)
    cv2.putText(img_out, "Detected Lanes", (int(1494-boxsize[0]/2),521), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)

    return img_out
def process_frame(img):
    # Resizing & Copy
    img=cv2.resize(img, imshape)
    img_original = np.copy(img)

    # Bird's eye view
    warped, Minv = warp(img)

    # Threshold
    #thresh = apply_color_threshold(warped)
    thresh = morphology_filter(warped)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, small_kernel)

    # Scan for lane lines using a margin search, otherwise use a sliding window search
    if left_line.detected and right_line.detected:
        left_lane_inds, right_lane_inds, output = margin_search(thresh)
        validate_lane_update(thresh, left_lane_inds, right_lane_inds)

    else:
        left_lane_inds, right_lane_inds, output = slide_window(thresh)
        validate_lane_update(thresh, left_lane_inds, right_lane_inds)

    final = draw_lane(img_original, thresh, Minv)
    result = assemble_img(warped, thresh, output, final)

    return result

class Line():
    def __init__(self, maxSamples=8):

        self.maxSamples = maxSamples
        # x values of the last n fits of the line
        self.recent_xfitted = collections.deque(maxlen=self.maxSamples)
        # Polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # Polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # Average x values of the fitted line over the last n iterations
        self.bestx = None
        # Was the line detected in the last iteration?
        self.detected = False
        # How many points in the line in current frame?
        self.points = None
        # How many points in the line in last frame?
        self.points_last = None


    def update_lane(self, ally, allx):
        # Updates lanes on every new frame
        # Mean x value
        self.bestx = np.mean(allx, axis=0)
        # Fit 2nd order polynomial
        new_fit = np.polyfit(ally, allx, 2)
        # Update current fit
        self.current_fit = new_fit
        # Add the new fit to the queue
        self.recent_xfitted.append(self.current_fit)
        # Use the queue mean as the best fit
        self.best_fit = np.mean(self.recent_xfitted, axis=0)


# if __name__ == "__main__":
#   cap = cv2.VideoCapture('C:/Users/turbo/Documents/Lane-finder/Lane-Finder/Input_videos/obstacle_challenge.mp4')
#   left_line = Line()
#   right_line = Line()
#   writer = None
#   while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print ("Not grabbed.")
#         break
        
#     # Run detection
#     if right_line.points is not None:
#         right_line.points_last, left_line.points_last = right_line.points, left_line.points
#     result = process_frame(frame)
#     cv2.imshow('image', result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


if __name__ == "__main__":
  cap = cv2.VideoCapture('C:/Users/turbo/Documents/Lane-finder/Lane-Finder/Input_videos/shadow_challenge.mp4')
  left_line = Line()
  right_line = Line()
  writer = None
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print ("Not grabbed.")
        break
        
    # Run detection
    if right_line.points is not None:
        right_line.points_last, left_line.points_last = right_line.points, left_line.points
    result = process_frame(frame)
    results = result.astype(np.uint8)
    if writer is None:
        # Initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('C:/Users/turbo/Documents/Lane-finder/Lane-Finder/Output_videos/shadow_output_combo_thresh.mp4', 0x7634706d, 30,
        (results.shape[1], results.shape[0]))
    
    # Write the output frame to disk
    writer.write(results)
        
# Release the file pointers
print("[INFO] cleaning up...")
cap.release()
writer.release()