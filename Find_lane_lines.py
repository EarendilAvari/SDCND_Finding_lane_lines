import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


# **********************grayscale***********************
# This function converts an image in RGB color to gray
# scale. To do that it uses the function cvtColor from 
# OpenCV.
# Inputs: img: Image to be converted in RGB.
# Output: Image converted to gray scale.
def grayscale(img):
	"""Applies the Grayscale transform
	This will return an image with only one color channel
	but NOTE: to see the returned image as grayscale
	(assuming your grayscaled image is called 'gray')
	you should call plt.imshow(gray, cmap='gray')"""
	return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# Or use BGR2GRAY if you read an image with cv2.imread()
	# return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# **********************gaussian_blur***********************
# This function applies a Gaussian Noise kernel. 
# To do that it uses the function cvtColor from OpenCV.
# Inputs: img: Image to be filtered.
# 		  kernel_size: Size of the gaussian noise kernel
# Output: Gray scale image with gaussian noise kernel applied.
def gaussian_blur(img, kernel_size):
	"""Applies a Gaussian Noise kernel"""
	return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# **************************canny****************************
# This function applies the Canny transformation to detect 
# edges on a grayscale image. To do that it uses the function
# Canny from OpenCV.
# Inputs: img: Image where the edges should be detected
# 		  low_threshold: Minimum value of gradient for a pixel to be an edge.
#						 If the gradient of a pixel is lower than this value, it is
#						 for sure not an edge.
# 		  high_threshold: Maximum value of gradient for a pixel to be an edge.
#						 If the gradient of a pixel is higher than this value, it is
#						 for sure an edge.
# Output: Image where only the edges are drawn.
def canny(img, low_threshold, high_threshold):
	"""Applies the Canny transform"""
	return cv2.Canny(img, low_threshold, high_threshold)

# **********************region_of_interest***********************
# This function only keeps the region of the image defined by the polygon
# formed from `vertices`. The rest of the image is set to black.
# Inputs: img: Image after the Canny edge detection which needs to be cutted.
#		  vertices: Numpy array including the vertice points of the polygon.
# Output: Image where only the edges inside the polygon defined are drawn.
def region_of_interest(img, vertices):
	#defining a blank mask to start with
	mask = np.zeros_like(img)

	#defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	if len(img.shape) > 2:
		channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255

	#filling pixels inside the polygon defined by "vertices" with the fill color
	cv2.fillPoly(mask, vertices, ignore_mask_color)

	#returning the image only where mask pixels are nonzero
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

# *************************hough_lines****************************
# Uses the Hough transformation to determine points defining lines. To do that
# the OpenCV function houghLinesP is used. This function returns arrays of two 
# points describing a line. So it only describes the actual lines on the image.
# This correspond to the real lines in the image, so if a line is dashed, 
# it is not recognized as an entire line. To recognize the entire lines 
# the points are separated in two groups, one with positive slope and 
# other one with negative slope. and with those points the position X 
# at the bottom of the image and at the far horizon of the image are 
# calculated and with them the lines are drawn in a new image.
# This new image only contains the detected lines.
# Inputs: img_original: Original image where the lines will be detected
# 		  img_edges:	Image where the edges where detected using 
#						Canny transformation.
#		  rho:			Distance resolution of the accumulator in pixels.
#		  theta:		Angle resolution of the accumulator in radians.
#		  threshold:	Accumulator threshold parameter. Only those lines are returned that get enough votes (>threshold)
#		  min_line_len: Minimum line length. Line segments shorter than that are rejected.
# 		  max_line_gap:	Maximum allowed gap between points on the same line to link them.
#		  y_horizon:	y value where the lines should stop.
# Output: Image where only the detected lines of the lane are drawn.
def hough_lines(img_original, img_edges, rho, theta, threshold, min_line_len, max_line_gap, y_horizon):
	if not hasattr(hough_lines, "last_avg_lower_x_right"):
		hough_lines.last_avg_lower_x_right = 0
	if not hasattr(hough_lines, "last_avg_upper_x_right"):
		hough_lines.last_avg_upper_x_right = 0
	if not hasattr(hough_lines, "last_avg_lower_x_left"):
		hough_lines.last_avg_lower_x_left = 0
	if not hasattr(hough_lines, "last_avg_upper_x_left"):
		hough_lines.last_avg_upper_x_left = 0
	if not hasattr(hough_lines, "last_img_points_def_lines"):
		hough_lines.last_img_points_def_lines = 0
	img_lines = np.zeros_like(img_original)
	img_points_def_lines = cv2.HoughLinesP(img_edges, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)

	if not hasattr(hough_lines, "last_img_points_def_lines"):
		hough_lines.last_img_points_def_lines = img_points_def_lines

	# It is verified if "img_points_def_lines" is none or not, if it is, the value of the last cycle is used. For that the 
	# attribute last_img_points_def_lines is used
	if img_points_def_lines is None:
		img_points_def_lines = hough_lines.last_img_points_def_lines

	hough_lines.last_img_points_def_lines = img_points_def_lines

	# Create two lists of points, one for the left line and another of the right line
	img_points_def_lines_right = []
	img_points_def_lines_left = []

	for drawing_line in img_points_def_lines:
		x1 = drawing_line[0,0]
		y1 = drawing_line[0,1]
		x2 = drawing_line[0,2]
		y2 = drawing_line[0,3]
		if x2 != x1:
			m = (y2 - y1)/(x2 - x1)

		# Though observations it is known that the detected lines corresponding to the right line have a slope between 0.5 and 1, and the
		# slope of the lines corresponding to the left line have a slope between -0.5 and -1
		if (m > 0.5) & (m < 1) & (x2 != x1):
			img_points_def_lines_right.append(drawing_line)
		elif (m < -0.5) & (m > -1) & (x2 != x1):
			img_points_def_lines_left.append(drawing_line)

	# Create four lists, where the calculated X coordinates are saved
	lower_x_values_right = []
	upper_x_values_right = []
	lower_x_values_left = []
	upper_x_values_left = []

	y_max = img_original.shape[0]	# Corresponds to the Y coordinate value at the bottom of the image
	y_min = y_horizon # Corresponds to the Y coordinate value at the far horizon of the image, where the lines cannot be seen anymore

	for drawing_line_right in img_points_def_lines_right:
		if drawing_line_right[0,2] != drawing_line_right[0,0]:
			m = (drawing_line_right[0,3] - drawing_line_right[0,1])/(drawing_line_right[0,2] - drawing_line_right[0,0])
			x1 = drawing_line_right[0,0]
			y1 = drawing_line_right[0,1]
			lower_x_values_right.append((y_max - y1)/m + x1) # The lower X values are appended in the list
			upper_x_values_right.append((y_min - y1)/m + x1) # The upper X values are appended in the list

	for drawing_line_left in img_points_def_lines_left:
		if drawing_line_left[0,2] != drawing_line_left[0,0]:
			m = (drawing_line_left[0,3] - drawing_line_left[0,1])/(drawing_line_left[0,2] - drawing_line_left[0,0])
			x1 = drawing_line_left[0,0]
			y1 = drawing_line_left[0,1]
			lower_x_values_left.append((y_max - y1)/m + x1) # The lower X values are appended in the list
			upper_x_values_left.append((y_min - y1)/m + x1) # The upper X values are appended in the list

	if len(lower_x_values_right) != 0:
		avg_lower_x_right = int(np.mean(lower_x_values_right)) # The mean for the low X value of the right line is calculated
	else:
		avg_lower_x_right = hough_lines.last_avg_lower_x_right

	if len(upper_x_values_right) != 0:
		avg_upper_x_right = int(np.mean(upper_x_values_right)) # The mean for the high X value of the right line is calculated
	else:
		avg_upper_x_right = hough_lines.last_avg_upper_x_right

	if len(lower_x_values_left) != 0: 
		avg_lower_x_left = int(np.mean(lower_x_values_left)) # The mean for the low X value of the left line is calculated
	else:
		avg_lower_x_left = hough_lines.last_avg_lower_x_left

	if len(upper_x_values_left) != 0:
		avg_upper_x_left = int(np.mean(upper_x_values_left)) # The mean for the high X value of the left line is calculated
	else:
		avg_upper_x_left = hough_lines.last_avg_upper_x_left

	hough_lines.last_avg_lower_x_right = avg_lower_x_right
	hough_lines.last_avg_upper_x_right = avg_upper_x_right
	hough_lines.last_avg_lower_x_left = avg_lower_x_left
	hough_lines.last_avg_upper_x_left = avg_upper_x_left

	# The right line is drawn using the average values
	cv2.line(img_lines, (avg_lower_x_right, y_max), (avg_upper_x_right, y_min), (255,0,0), 10)

	# The left line is drawn using the average values
	cv2.line(img_lines, (avg_lower_x_left, y_max), (avg_upper_x_left, y_min), (255,0,0), 10)

	return img_lines


# *************************weighted_img****************************
# It overlaps one image into another. It can be used to draw the detected lines above the
# original image. It uses the OpenCV function addWeighted. 
# Inputs: img_original: Original image where the lines need to be detected.
#		  img_lines:	Image where only the detected lines of the lane are drawn.
#		  alpha:		Weight of the original image.
#		  beta:			Weight of the lines image.
# 		  gamma:		Scalar added to the sum. For this case irrelevant.
def weighted_img(img_original, img_lines, alpha=0.8, beta=1., gamma=0.):
	return cv2.addWeighted(img_original, alpha, img_lines, beta, gamma)


# *************************houghVertices***************************
# It creates the vertices array based on 6 values. Four of the X values and two of the Y values.
# Inputs: x_left_bottom:	Coordinate X of the vertice at the left and the bottom
#		  x_left_top:		Coordinate X of the vertice at the left and the top
#		  x_right_bottom:	Coordinate X of the vertice at the right and the bottom
#		  x_right_top:		Coordinate X of the vertice at the right and the top
#		  y_bottom:			Coordinate Y at the bottom
#		  y_horizon:		Coordinate Y at the top
# Output: Numpy array including the points of the four vertices of the polygon.
def houghVertices(x_left_bottom, x_left_top, x_right_bottom, x_right_top,y_bottom, y_horizon):
	vertice_left_bottom = (x_left_bottom, y_bottom)
	vertice_left_top = (x_left_top, y_horizon)
	vertice_right_bottom = (x_right_bottom, y_bottom)
	vertice_right_top = (x_right_top, y_horizon)
	return np.array([[vertice_left_bottom, vertice_left_top, vertice_right_top, vertice_right_bottom]], dtype = np.int32)


# *************************findLanes********************************
# Generalization of the entire pipeline in one function. So it is not needed
# to execute every function by separate. It also allows to occult the entire details
# of the pipeline.
# Inputs: img:					Image where the lines need to be identified.
# 		  Canny_low_threshold:	Minimum value of gradient for a pixel to be an edge.
#						 		If the gradient of a pixel is lower than this value, it is
#						 		for sure not an edge.
#		  lane_x_left_bottom:	Coordinate X of the vertice at the left and the bottom of the polygon 
#								where the lines are located.
# 		  lane_x_left_top:		Coordinate X of the vertice at the left and the top of the polygon 
#								where the lines are located.
# 		  lane_x_right_bottom:	Coordinate X of the vertice at the right and the bottom of the polygon 
#								where the lines are located.
#		  lane_x_right_top:		Coordinate X of the vertice at the right and the top of the polygon 
#								where the lines are located.
#		  lane_y_bottom:		Coordinate Y at the bottom of the polygon where the lines are located.
#		  lane_y_horizon:		Coordinate Y at the top of the polygon where the lines are located.
#		  Hough_rho:			Distance resolution of the accumulator in pixels.
#		  Hough_theta:			Angle resolution of the accumulator in radians.
#		  Hough_threshold:		Accumulator threshold parameter. Only those lines are returned that get enough votes (>threshold)
#		  Hough_min_line_len:	Minimum line length. Line segments shorter than that are rejected.
#		  Hough_max_line_gap:	Maximum allowed gap between points on the same line to link them.
# Output: Image with the lines identified and drawn in red.
def findLanes(img, Canny_low_threshold, lane_x_left_bottom, lane_x_left_top, lane_x_right_bottom, lane_x_right_top, lane_y_bottom, lane_y_horizon, 
	Hough_rho, Hough_theta, Hough_threshold, Hough_min_line_len, Hough_max_line_gap):
	
	img_gray = grayscale(img)
	img_gaussian = gaussian_blur(img_gray,5)
	img_edges = canny(img_gaussian, Canny_low_threshold, 3*Canny_low_threshold)
	img_edges_Hough_vertices = houghVertices(lane_x_left_bottom, lane_x_left_top, lane_x_right_bottom, lane_x_right_top, lane_y_bottom, lane_y_horizon)
	img_edges_masked = region_of_interest(img_edges, img_edges_Hough_vertices)
	img_lines = hough_lines(img, img_edges_masked, Hough_rho, Hough_theta, Hough_threshold, Hough_min_line_len, Hough_max_line_gap, lane_y_horizon)
	img_output = weighted_img(img, img_lines)

	return img_output

