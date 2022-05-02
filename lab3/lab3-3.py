import numpy as np
#########################
#       Exercise 1      #
#########################

def generateTranslationMatrix(x, y, z):
    '''
    return the homogeneous transformation matrix for the given translation (x, y, z)
      parameter: 
        sx, sy, sz: scaling parameters for x-, y-, z-axis, respectively
      return:
        ndarray of size (4, 4)
    '''
    pass

def generateScalingMatrix(sx, sy, sz):
    '''
    return the homogeneous transformation matrix for the given scaling parameters (sx, sy, sz)
      parameter:
        sx, sy, sz: scaling parameters for x-, y-, z-axis, respectively
      return:
        ndarray of size (4, 4)
    '''
    pass

def generateRotationMatrix(rad, axis):
    '''
    return the homogeneous transformation matrix for the given rotation parameters (rad, axis)
      parameter:
        rad: radians for rotation
        axis: axis for rotation, can only be one of ('x', 'y', 'z', 'X', 'Y', 'Z')
      return: 
        ndarray of size (4, 4)
    '''
    pass

# Case 1
def part1Case1():
    # translation matrix
    t = 
    # scaling matrix
    s = 
    # rotation matrix
    r = 
    # data in homogeneous coordinate
    data = np.array([2, 3, 4, 1]).T
    pass

# Case 2
def part1Case2():
    # translation matrix
    t = 
    # scaling matrix
    s = 
    # rotation matrix
    r = 
    # data in homogeneous coordinate
    data = np.array([6, 5, 2, 1]).T
    pass

# Case 3
def part1Case3():
    # translation matrix
    t = 
    # scaling matrix
    s = 
    # rotation matrix
    r = 
    # data in homogeneous coordinate
    data = np.array([3, 2, 5, 1]).T
    pass

#########################
#       Exercise 2      #
#########################

# Part 1
def generateRandomSphere(r, n):
    '''
    generate a point cloud of n points in spherical coordinates (radial distance, polar angle, azimuthal angle)
      parameter:
        r: radius of the sphere
        n: total number of points
    return:
      spherical coordinates, ndarray of size (3, n), where the 3 rows are ordered as (radial distances, polar angles, azimuthal angles)
    '''
    pass

def sphericalToCatesian(coors):
    '''
    convert n points in spherical coordinates to cartesian coordinates, then add a row of 1s to them to convert
    them to homogeneous coordinates
      parameter:
        coors: ndarray of size (3, n), where the 3 rows are ordered as (radial distances, polar angles, azimuthal angles)
    return:
      catesian coordinates, ndarray of size (4, n), where the 4 rows are ordered as (x, y, z, 1)
    '''
    pass

# Part 2
def applyRandomTransformation(sphere1):
    '''
    generate two random transformations, one of each (scaling, rotation),
    apply them to the input sphere in random order, then apply a random translation,
    then return the transformed coordinates of the sphere, the composite transformation matrix,
    and the three random transformation matrices you generated
      parameter:
        sphere1: homogeneous coordinates of sphere1, ndarray of size (4, n), 
                 where the 4 rows are ordered as (x, y, z, 1)
      return:
        a tuple (p, m, t, s, r)
        p: transformed homogeneous coordinates, ndarray of size (4, n), 
                 where the 4 rows are ordered as (x, y, z, 1)
        m: composite transformation matrix, ndarray of size (4, 4)
        t: translation matrix, ndarray of size (4, 4)
        s: scaling matrix, ndarray of size (4, 4)
        r: rotation matrix, ndarray of size (4, 4)
    '''
    pass

def calculateTransformation(sphere1, sphere2):
    '''
    calculate the composite transformation matrix from sphere1 to sphere2
      parameter:
        sphere1: homogeneous coordinates of sphere1, ndarray of size (4, n), 
                 where the 4 rows are ordered as (x, y, z, 1)
        sphere2: homogeneous coordinates of sphere2, ndarray of size (4, n), 
                 where the 4 rows are ordered as (x, y, z, 1)
    return:
      composite transformation matrix, ndarray of size (4, 4)
    '''
    pass

def decomposeTransformation(m):
    '''
    decomposite the transformation and return the translation, scaling, and rotation matrices
      parameter:
        m: homogeneous transformation matrix, ndarray of size (4, 4)

    return:
      tuple of three matrices, (t, s, r)
        t: translation matrix, ndarray of size (4, 4)
        s: scaling matrix, ndarray of size (4, 4)
        r: rotation matrix, ndarray of size (4, 4)
    '''
    pass


#########################
#      Main function    #
#########################
def main():
    pass

if __name__ == "__main__":
    main()