import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2



def grayscale(img):
	"""Applies the Grayscale transform
	This will return an image with only one color channel
	but NOTE: to see the returned image as grayscale
	(assuming your grayscaled image is called 'gray')
	you should call plt.imshow(gray, cmap='gray')"""
	return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# Or use BGR2GRAY if you read an image with cv2.imread()
	# return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size):
	"""Applies a Gaussian Noise kernel"""
	return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
	"""Applies the Canny transform"""
	return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
	"""
	Applies an image mask.

	Only keeps the region of the image defined by the polygon
	formed from `vertices`. The rest of the image is set to black.
	`vertices` should be a numpy array of integer points.
	"""
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

def weighted_img(img_original, img_lines, alpha=0.8, beta=1., gamma=0.):
	"""
	`img_lines` is the output of the hough_lines(), An image with lines drawn on it.

	`img_original` should be the image before any processing.

	The result image is computed as follows:

	initial_img * alpha + img * beta + gamma
	NOTE: initial_img and img must be the same shape!
	"""
	return cv2.addWeighted(img_original, alpha, img_lines, beta, gamma)

def houghVertices(img, x_left_bottom, x_left_top, x_right_bottom, x_right_top,y_bottom, y_horizon):
	vertice_left_bottom = (x_left_bottom, y_bottom)
	vertice_left_top = (x_left_top, y_horizon)
	vertice_right_bottom = (x_right_bottom, y_bottom)
	vertice_right_top = (x_right_top, y_horizon)
	return np.array([[vertice_left_bottom, vertice_left_top, vertice_right_top, vertice_right_bottom]], dtype = np.int32)


def findLanes(img, Canny_low_threshold, lane_x_left_bottom, lane_x_left_top, lane_x_right_bottom, lane_x_right_top, lane_y_bottom, lane_y_horizon, 
	Hough_rho, Hough_theta, Hough_threshold, Hough_min_line_len, Hough_max_line_gap):
	
	img_gray = grayscale(img)
	img_gaussian = gaussian_blur(img_gray,5)
	img_edges = canny(img_gaussian, Canny_low_threshold, 3*Canny_low_threshold)
	img_edges_Hough_vertices = houghVertices(img_edges, lane_x_left_bottom, lane_x_left_top, lane_x_right_bottom, lane_x_right_top, lane_y_bottom, lane_y_horizon)
	img_edges_masked = region_of_interest(img_edges, img_edges_Hough_vertices)
	img_lines = hough_lines(img, img_edges_masked, Hough_rho, Hough_theta, Hough_threshold, Hough_min_line_len, Hough_max_line_gap, lane_y_horizon)
	img_output = weighted_img(img, img_lines)

	return img_output

# END