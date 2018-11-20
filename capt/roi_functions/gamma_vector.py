import math
import cmath
import numpy
import matplotlib
from aotools.functions import circle
from matplotlib import pyplot; pyplot.ion()
matplotlib.rcParams['image.origin'] = 'lower'


def gamma_vector(blankCovMap, angle, gs_pos, belowGround, envelope, show_plots=False):
    """Extracts covariance map pixel coordinates along the angle of GS stellar separation (the ROI).

    Parameters:
        blankCovMap (ndarray): square array of zeros with covariance map dimensions.
        angle (float): overwrites gs_pos to artificially determine angle of stellar separation. Set as 'False' to use gs_pos. 
        gs_pos (ndarray): GS asterism in telescope FoV (for 2 GSs).
        belowGround (int): number of sub-aperture separations the ROI expresses 'below-ground'.
        envelope (int): number of sub-aperture separations the ROI expresses either side of stellar separation.
        show_plots (bool): used to test function by visualising ROI within the covariance map.

    Returns:
        ndarray: x and y coordinates of pixels within ROI (has shape (1+envelope*2, (blankCovMap.shape[0]+1)/2 - belowGround, 2)).
        ndarray: imshow to visualise ROI within covariance map.
        float: angle of GS stellar separation."""

    # envelope = envelope.astype('int')
    # belowGround = belowGround.astype('int')
    x, y = (len(blankCovMap)-1)/2., (len(blankCovMap)-1)/2.
    blankCovMap = numpy.zeros(blankCovMap.shape)
    quadratureCorrection = 0
    
    if str(angle) == str(False):
        gs_pos = gs_pos.astype(float)
        theta = vector_angle(gs_pos[0], gs_pos[1])
        # print 'Vector Angle:', math.degrees(theta), '\n'
    else:
        theta = math.radians(angle)
        # print 'Vector Angle:', math.degrees(theta), '\n'

    covRadius = math.ceil(blankCovMap.shape[0]/2.)
    a = numpy.round(covRadius*numpy.sin(theta))
    o = numpy.round(covRadius*numpy.cos(theta))
    # print 'o', o
    # print 'a:', a
    # print maxRad
    vector = numpy.zeros((1+envelope*2, int(covRadius)+belowGround, 2))
    
    count=0
    #Steps taken along covariance map - starts at -belowGround
    steps = range(-belowGround, int(covRadius), 1)
    if numpy.abs(a)>=numpy.abs(o):

        #Loop over total number of steps
        for i in range(numpy.array(steps).shape[0]):
            
            #If adjacent is not positive, track negative direction
            if a!=numpy.abs(a):
                xStep = int(numpy.round(x-steps[i]))
                
                #Take step. o/a not valid if a=0 
                if a==0:
                    yStep = int(numpy.round(y-((o)*steps[i])))
                else:
                    yStep = int(numpy.round(y-((o/a)*steps[i])))
            
            #If adjacent is positive, track positive direction
            else:
                xStep = int(numpy.round(x+steps[i]))
                
                #Take step. o/a not valid if a=0
                if a==0:
                    yStep = int(numpy.round(y+((o)*steps[i])))
                else:
                    yStep = int(numpy.round(y+((o/a)*steps[i])))

            #Mark location in blankCovMap
            blankCovMap[yStep, xStep]=1

            #If current location is within map, add envelope and vector coordinates
            if count<=numpy.array(steps).shape[0]:
                
                vector[envelope,i] = numpy.array((yStep,xStep), dtype='int')
                for j in range(envelope):

                    if yStep+(j+1)<blankCovMap.shape[0] and yStep+(j+1)>=0:
                        if xStep<blankCovMap.shape[0] and xStep>=0:
                            blankCovMap[yStep+j+1, xStep]=(j*2)+3
                            vector[envelope+(j+1),i] = numpy.array((yStep+j+1, xStep))

                    if yStep-(j+1)<blankCovMap.shape[0] and yStep-(j+1)>=0:
                        if xStep<blankCovMap.shape[0] and xStep>=0:
                            blankCovMap[yStep-(j+1), xStep]=(j*2)+2
                            vector[envelope-(j+1),i] = numpy.array((yStep-(j+1), xStep))
                count+=1
                # print 'x, y:', xStep, yStep

    if numpy.abs(o)>numpy.abs(a):
        
        for i in range(numpy.array(steps).shape[0]):
            if o==numpy.abs(o):
                yStep = int(numpy.round(y+steps[i]))
                if o==0:
                    xStep = int(numpy.round(x+((a)*steps[i])))
                else:
                    xStep = int(numpy.round(x+((a/o)*steps[i])))
                    # print xStep
                    # print yStep
            else:
                yStep = int(numpy.round(y-steps[i]))
                if o==0:
                    xStep = int(numpy.round(x+((a)*steps[i])))
                else:
                    xStep = int(numpy.round(x-((a/o)*steps[i])))

            blankCovMap[yStep, xStep]=1
            
            if count<=numpy.array(steps).shape[0]:
                
                vector[envelope,i] = numpy.array((yStep,xStep), dtype='int')
                for j in range(envelope):
                    
                    if xStep+(j+1)<blankCovMap.shape[0] and xStep+(j+1)>=0:
                        if yStep<blankCovMap.shape[0] and yStep>=0:
                            blankCovMap[yStep, xStep+j+1]=(j*2)+3
                            vector[envelope+j+1,i] = numpy.array((yStep, xStep+j+1))
                        
                    if xStep-(j+1)<blankCovMap.shape[0] and xStep-(j+1)>=0:
                        if yStep<blankCovMap.shape[0] and yStep>=0:
                            blankCovMap[yStep, xStep-(j+1)]=(j*2)+2
                            vector[envelope-j-1,i] = numpy.array((yStep, xStep-(j+1)))
                count+=1
                # print 'x, y:', xStep, yStep
    
    if show_plots==True:
        # pyplot.figure('Covariance Map Vector')
        pyplot.figure()
        pyplot.imshow(blankCovMap, interpolation='nearest')
    return vector.astype(int), blankCovMap, theta



def vector_angle(a, b):
    
    """Calculates angle between 2 locations in a 2D plane.
    
    Parameters:
        a (float): y and x position of location 1.
        b (float): y and x position of location 2.
    
    Returns:
        float: angle of separation between a and b [rads]."""

    dy = a[1] - b[1]
    dx = a[0] - b[0] + 1.e-20

    count = 0
    theta = (numpy.pi/2.) - numpy.arctan(dy/dx)

    if dx<0 and dy>=0:
        theta = numpy.pi + ((numpy.pi/2.) - numpy.arctan(dy/dx))
        count+=1
        # print 'Vector Quadrant: 4'
    if dx<0 and dy<0:
        theta = 3*(numpy.pi)/2. - numpy.arctan(dy/dx)
        count+=1
        # print 'Vector Quadrant: 3'
    # if count==0:
    #     if numpy.arctan(dy/dx)<0:
    #         # print 'Vector Quadrant: 2'
    #     else:
    #         # print 'Vector Quadrant: 1'
    
    return theta



def advanced_testing(pupil_mask, quad_num):
    """Visual test to check gamma_vector is working as expected.
    
    Parameters:
        pupil_mask (ndarray): mask of SHWFS sub-apertures within the telescope's pupil.
        quad_num (float): quadrant number to test (1-4)."""

    hard = 20
    quad_num = quad_num - 1

    covMapDim = (2*pupil_mask.shape[0])-1
    blankCovMap = numpy.zeros((covMapDim,covMapDim))
    testAng = numpy.linspace(numpy.pi*quad_num/2.,  (numpy.pi/2.)*quad_num + (numpy.pi/2.), hard)

    for i in range(hard):
        gs_pos = numpy.array(([0,0], [-numpy.sin(testAng[i]), -numpy.cos(testAng[i])]))
        c, b, t = gamma_vector(blankCovMap, 'False', gs_pos, 0, 0)
        vect = b

        pyplot.figure(str(math.degrees(testAng[i])))
        pyplot.imshow(b)
        print('Input:', math.degrees(testAng[i]), '; Theta:', math.degrees(t),'\n' )
        # print gs_pos



if __name__ == '__main__':

    gs_pos = numpy.array([[0.,0.], [1.,0.]])
    pupil_mask = circle(7./2, 7)
    covMapDim = (2*pupil_mask.shape[0])-1
    blankCovMap = numpy.zeros((covMapDim,covMapDim))
    belowGround = 6
    envelope = 6
    a,b,t = gamma_vector(blankCovMap, False, gs_pos, belowGround, envelope, show_plots=False)

    pyplot.figure()
    pyplot.imshow(a[:,:,1])

    advanced_testing(pupil_mask, 4)