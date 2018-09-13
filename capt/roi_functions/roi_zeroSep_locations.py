import numpy
from matplotlib import pyplot; pyplot.ion()

def roi_zeroSep_locations(combs, roi_width, roi_length, roi_axis, roi_belowGround):
    """An array that matches the covariance map ROI shape with 1s, but 0s at 
    sub-aperture separations of zero.

    Parameters:
        combs (int): no. of GS combinations
        roi_width (int): width of covariance map ROI.
        roi_length (int): length of covariance map ROI.
        roi_axis (str): in which axis to express ROI ('x', 'y', 'x+y' or 'x and y')
        roi_belowGround (int): number of sub-aperture separations the ROI expresses 'below-ground'.

    Returns:
        ndarray: matrix of 1s and 0s - 0s at sub-aperture separations of zero."""

    ones = numpy.ones((roi_width, roi_length)).astype('int')
    middle = int((roi_width-1)/2.)
    ones[middle, roi_belowGround] = 0
    ones = numpy.tile(ones, (combs,1))
    if roi_axis=='x and y':
        ones = numpy.tile(ones, 2)
    zeroSep_locations = numpy.where(ones==0)
    return zeroSep_locations

if __name__ == '__main__':
    combs = 6
    roi_width = 5
    roi_length = 8
    roi_axis = 'x and y'
    roi_belowGround = 1

    o = roi_zeroSep_false(combs, roi_width, roi_length, roi_axis, roi_belowGround)