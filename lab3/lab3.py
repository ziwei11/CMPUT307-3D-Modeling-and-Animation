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
    #pass
    # create a 4 * 4 eye matrix
    translation_matrix = np.eye(4)
    # set translation number on corresponding places
    translation_matrix[0, 3] = x
    translation_matrix[1, 3] = y
    translation_matrix[2, 3] = z
    return translation_matrix


def generateScalingMatrix(sx, sy, sz):
    '''
    return the homogeneous transformation matrix for the given scaling parameters (sx, sy, sz)
      parameter:
        sx, sy, sz: scaling parameters for x-, y-, z-axis, respectively
      return:
        ndarray of size (4, 4)
    '''
    #pass
    # create a 4 * 4 eye matrix
    scaling_matrix = np.eye(4)
    # set translation number on corresponding places
    scaling_matrix[0, 0] = sx
    scaling_matrix[1, 1] = sy
    scaling_matrix[2, 2] = sz
    return scaling_matrix


def generateRotationMatrix(rad, axis):
    '''
    return the homogeneous transformation matrix for the given rotation parameters (rad, axis)
      parameter:
        rad: radians for rotation
        axis: axis for rotation, can only be one of ('x', 'y', 'z', 'X', 'Y', 'Z')
      return: 
        ndarray of size (4, 4)
    '''
    #pass
    # create a 4 * 4 eye matrix
    rotation_matrix = np.eye(4)
    # axis x for rotation
    if axis == 'x' or axis == 'X':
      # set the rotation matrix for axis x rotation
      rotation_matrix[1, 1] = np.cos(rad)
      rotation_matrix[1, 2] = -np.sin(rad)
      rotation_matrix[2, 1] = np.sin(rad)
      rotation_matrix[2, 2] = np.cos(rad)
      return rotation_matrix
    # axis y for rotation
    if axis == 'y' or axis == 'Y':
      # set the rotation matrix for axis y rotation
      rotation_matrix[0, 0] = np.cos(rad)
      rotation_matrix[0, 2] = np.sin(rad)
      rotation_matrix[2, 0] = -np.sin(rad)
      rotation_matrix[2, 2] = np.cos(rad)
      return rotation_matrix
    # axis z for rotation
    if axis == 'z' or axis == 'Z':
      # set the rotation matrix for axis z rotation
      rotation_matrix[0, 0] = np.cos(rad)
      rotation_matrix[0, 1] = -np.sin(rad)
      rotation_matrix[1, 0] = np.sin(rad)
      rotation_matrix[1, 1] = np.cos(rad)
      return rotation_matrix      


# Case 1
def part1Case1():
    # translation matrix
    t = (2, 3, -2)
    # scaling matrix
    s = (0.5, 2, 2)
    # rotation matrix
    r = (1/4) * np.pi
    # data in homogeneous coordinate
    data = np.array([2, 3, 4, 1]).T
    #pass
    # generate translation matrix
    mt = generateTranslationMatrix(t[0], t[1], t[2])
    # generate scaling matrix
    ms = generateScalingMatrix(s[0], s[1], s[2])
    # generate rotation matrix
    mr = generateRotationMatrix(r, 'z')
    # get the temp1 matrix after multiply translation matrix
    temp1 = np.matmul(mt, data)
    # print the temp1 matrix
    print('after t: ')
    print(temp1)
    # get the temp2 matrix after multiply scaling matrix
    temp2 = np.matmul(ms, temp1)
    # print the temp2 matrix
    print('after s: ')
    print(temp2)
    # get the case1_result matrix after multiply rotation matrix
    case1_result = np.matmul(mr, temp2)
    # print the case1_result matrix
    print('after r: ')
    print(case1_result)
    print('')


# Case 2
def part1Case2():
    # translation matrix
    t = (4, -2, 3)
    # scaling matrix
    s = (3, 1, 3)
    # rotation matrix
    r = -(1/6) * np.pi
    # data in homogeneous coordinate
    data = np.array([6, 5, 2, 1]).T
    #pass
    # generate translation matrix
    mt = generateTranslationMatrix(t[0], t[1], t[2])
    # generate scaling matrix
    ms = generateScalingMatrix(s[0], s[1], s[2])
    # generate rotation matrix
    mr = generateRotationMatrix(r, 'y')
    # get the temp1 matrix after multiply scaling matrix
    temp1 = np.matmul(ms, data)
    # print the temp1 matrix
    print('after s: ')
    print(temp1)
    # get the temp2 matrix after multiply translation matrix
    temp2 = np.matmul(mt, temp1)
    # print the temp2 matrix
    print('after t: ')
    print(temp2)
    # get the case2_result matrix after multiply rotation matrix
    case2_result = np.matmul(mr, temp2)
    # print the case2_result matrix
    print('after r: ')
    print(case2_result)
    print('')


# Case 3
def part1Case3():
    # translation matrix
    t = (5, 2, -3)
    # scaling matrix
    s = (2, 2, -2)
    # rotation matrix
    r = (1/12) * np.pi
    # data in homogeneous coordinate
    data = np.array([3, 2, 5, 1]).T
    #pass
    # generate translation matrix
    mt = generateTranslationMatrix(t[0], t[1], t[2])
    # generate scaling matrix
    ms = generateScalingMatrix(s[0], s[1], s[2])
    # generate rotation matrix
    mr = generateRotationMatrix(r, 'x')
    # get the temp1 matrix after multiply rotation matrix
    temp1 = np.matmul(mr, data)
    # print the temp1 matrix
    print('after r: ')
    print(temp1)
    # get the temp2 matrix after multiply scaling matrix
    temp2 = np.matmul(ms, temp1)
    # print the temp2 matrix
    print('after s: ')
    print(temp2)
    # get the case3_result matrix after multiply translation matrix
    case3_result = np.matmul(mt, temp2)
    # print the case3_result matrix
    print('after t: ')
    print(case3_result)
    print('')


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
    #pass
    # initial a new matrix which is (3, n) and all items of the matrix are 0
    result = np.zeros((3, n))
    # the first row is random radial distances and value range is [0, r]
    result[0, :] = r * np.random.rand(n)
    # the second row is random polar angles and value range is [0, pi]
    result[1, :] = np.pi * np.random.rand(n)
    # the third row is random azimuthal angles and value range is [0, 2 * pi]
    result[2, :] = 2 * np.pi * np.random.rand(n)
    return result


def sphericalToCatesian(coors):
    '''
    convert n points in spherical coordinates to cartesian coordinates, then add a row of 1s to them to convert
    them to homogeneous coordinates
      parameter:
        coors: ndarray of size (3, n), where the 3 rows are ordered as (radial distances, polar angles, azimuthal angles)
    return:
      catesian coordinates, ndarray of size (4, n), where the 4 rows are ordered as (x, y, z, 1)
    '''
    #pass
    # get the number n of points
    n = coors.shape[1]
    # initial a new matrix which is (4, n) and all items of the matrix are 1
    new_matrix = np.ones((4, n))
    for i in range(n):
      # x = r * cos(azimuthal angles) * sin(polar angles)
      new_matrix[0, i] = coors[0, i] * np.cos(coors[2, i]) * np.sin(coors[1, i])
      # y = r * sin(azimuthal angles) * sin(polar angles)
      new_matrix[1, i] = coors[0, i] * np.sin(coors[2, i]) * np.sin(coors[1, i])
      # z = r * cos(polar angles)
      new_matrix[2, i] = coors[0, i] * np.cos(coors[1, i])
    return new_matrix


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
    #pass
    # create a 4 * 4 eye matrix as s (scaling) matrix
    s = np.eye(4)
    # initial three variables
    a = 0
    b = 0
    c = 0
    # set random scaling number on corresponding places
    # scaling number cannot be 0
    while a == 0 or b == 0 or c == 0:
      a = np.random.randint(low=-40, high=40)
      b = np.random.randint(low=-40, high=40)
      c = np.random.randint(low=-40, high=40)
    s[0, 0] = a
    s[1, 1] = b
    s[2, 2] = c
    #s[2, 2] = 10 * np.random.rand()

    # create a 4 * 4 eye matrix as t (translation) matrix
    t = np.eye(4)
    # set random translation number on corresponding places
    t[0, 3] = np.random.randint(low=-10, high=10)
    t[1, 3] = np.random.randint(low=-10, high=10)
    t[2, 3] = np.random.randint(low=-10, high=10)

    # create a 4 * 4 eye matrix as r (rotation) matrix
    r = np.eye(4)
    # init a random number to decide the axis of the rotation
    random_number = np.random.randint(3)
    # if random_number == 0, we set the rotation axis is x
    if random_number == 0:
      alpha = 2 * np.pi * np.random.rand()
      r[1, 1] = np.cos(alpha)
      r[1, 2] = -np.sin(alpha)
      r[2, 1] = np.sin(alpha)
      r[2, 2] = np.cos(alpha)
    # if random_number == 1, we set the rotation axis is y
    if random_number == 1:
      alpha = 2 * np.pi * np.random.rand()
      r[0, 0] = np.cos(alpha)
      r[0, 2] = np.sin(alpha)
      r[2, 0] = -np.sin(alpha)
      r[2, 2] = np.cos(alpha)
    # if random_number == 0, we set the rotation axis is z
    if random_number == 2:
      alpha = 2 * np.pi * np.random.rand()
      r[0, 0] = np.cos(alpha)
      r[0, 1] = -np.sin(alpha)
      r[1, 0] = np.sin(alpha)
      r[1, 1] = np.cos(alpha)
    
    # apply random scaling and random rotation to the input sphere in random order
    # then apply a random translation

    # generate a random number for the order
    ran_num = np.random.randint(2)

    # when random number == 0, suppose perform a scaling followed by a rotation, followed by a translation (first s, then r, last t)
    if ran_num == 0:
      # first, calculate s * sphere1 by np.matmul (Matrix product of two arrays)
      temp1 = np.matmul(s, sphere1)
      # second calculate r * temp1 by np.matmul 
      temp2 = np.matmul(r, temp1)
      # third calculate t * temp2 by np.matmul 
      p = np.matmul(t, temp2)
      # get the composite transformation matrix
      m = np.matmul(np.matmul(t, r), s)

    # when random number == 1, suppose perform a rotation followed by a scaling, followed by a translation (first r, then s, last t)
    if ran_num == 1:
      # first, calculate r * sphere1 by np.matmul (Matrix product of two arrays)
      temp1 = np.matmul(r, sphere1)
      # second calculate s * temp1 by np.matmul 
      temp2 = np.matmul(s, temp1)
      # third calculate t * temp2 by np.matmul 
      p = np.matmul(t, temp2)
      # get the composite transformation matrix
      m = np.matmul(np.matmul(t, s), r)

    return (p, m, t, s, r)


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
    #pass
    M1 = sphere1
    M2 = sphere2
    # M2 = H * M1
    # M2 = H * u * s * vh
    # M2 * M1_inverse = H
    # M1 = u * s * vh
    u, s, vh = np.linalg.svd(M1)
    # create a new_s matrix for s, because the return for s is a vector with the singular values
    new_s = np.zeros((u.shape[1], vh.shape[0]))
    # put s's singular values into the new_s matrix
    new_s[:len(s), :len(s)] = np.diag(s)
    v = np.transpose(vh)
    s_plus = np.linalg.pinv(new_s)
    uh = np.transpose(u)
    # M1_inverse = vh+ * s+ * u+
    # M1_inverse = vh.T * s+ * u.T
    # M1_inverse = v * s+ * uh
    M1_inverse = np.matmul(np.matmul(v, s_plus), uh)
    # M2 * M1_inverse = H
    H = np.matmul(M2, M1_inverse)
    return H

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
    #pass
     # create a 4 * 4 eye matrix as t (translation) matrix
    t = np.eye(4)
     # create a 4 * 4 eye matrix as s (scaling) matrix
    s = np.eye(4)
     # create a 4 * 4 eye matrix as r (rotation) matrix
    r = np.eye(4)

    # if a number in m is close to 0, change it to 0
    for i in range(len(m)):
      for j in range(len(m[i])):
        if np.isclose(m[i, j], 0) == True:
          m[i, j] = 0

    # the rotation axis is z
    if m[0, 0] != 0 and m[0, 1] != 0 and m[1, 0] != 0 and m[1, 1] != 0:
      # observe matrix m, we can find that t[0:3, 3] = m[0:3, 3], s[2, 2] = m[2, 2]
      t[0:3, 3] = m[0:3, 3]
      s[2, 2] = m[2, 2]
      tan = 0
      sin = 0
      alpha = 0

      # decide wheather the order is a scaling followed by a rotation, followed by a translation (first s, then r, last t)
      if np.allclose((m[1, 0] / m[0, 0]), (- m[0, 1] / m[1, 1])):
        # alpha = np.arctan(s[0, 0]*sin(alpha) / s[0, 0]*cos(alpha))
        alpha = np.arctan(m[1, 0] / m[0, 0])
        tan = m[1, 0] / m[0, 0]
        sin = m[1, 0]
        print('The order is a scaling followed by a rotation, followed by a translation.')

      # decide wheather the order is a rotation followed by a scaling, followed by a translation (first r, then s, last t)
      if np.allclose((- m[0, 1] / m[0, 0]), (m[1, 0] / m[1, 1])):
        # alpha = np.arctan(s[1, 1]*sin(alpha) / s[1, 1]*cos(alpha))
        alpha = np.arctan(m[1, 0] / m[1, 1])
        tan = m[1, 0] / m[1, 1]
        sin = m[1, 0]
        print('The order is a rotation followed by a scaling, followed by a translation.')

      # if tan > 0, then alpha belongs to [0, 1/2 * pi] union [pi, 3/2 * pi]
      if tan > 0:
        # sin(alpha) < 0, alpha belongs to [pi, 3/2 * pi], not [0, 1/2 * pi]
        if sin < 0:
          alpha = alpha + np.pi
      # if tan < 0, then alpha belongs to [1/2 * pi, pi] union [3/2 * pi, 2 * pi]
      if tan < 0:
        # sin(alpha) > 0, alpha belongs to [1/2 * pi, pi], not [3/2 * pi, 2 * pi]
        if sin > 0:
          alpha = alpha + np.pi
        else:
          alpha = alpha + 2 * np.pi
      r[0, 0] = np.cos(alpha)
      r[0, 1] = -np.sin(alpha)
      r[1, 0] = np.sin(alpha)
      r[1, 1] = np.cos(alpha)
      s[0, 0] = m[0, 0] / r[0, 0]
      s[1, 1] = m[1, 1] / r[1, 1]
      return (t, s, r)  


    # the rotation axis is x
    if m[1, 1] != 0 and m[1, 2] != 0 and m[2, 1] != 0 and m[2, 2] != 0:
      # observe matrix m, we can find that t[0:3, 3] = m[0:3, 3], s[0, 0] = m[0, 0]
      t[0:3, 3] = m[0:3, 3]
      s[0, 0] = m[0, 0]
      tan = 0
      sin = 0
      alpha = 0

      # decide wheather the order is a scaling followed by a rotation, followed by a translation (first s, then r, last t)
      if np.allclose((m[2, 1] / m[1, 1]), (- m[1, 2] / m[2, 2])):      
        # alpha = np.arctan(s[1, 1]*sin(alpha) / s[1, 1]*cos(alpha))
        alpha = np.arctan(m[2, 1] / m[1, 1])
        tan = m[2, 1] / m[1, 1]
        sin = m[2, 1]
        print('The order is a scaling followed by a rotation, followed by a translation.')

      # decide wheather the order is a rotation followed by a scaling, followed by a translation (first r, then s, last t)
      if np.allclose((- m[1, 2] / m[1, 1]), (m[2, 1] / m[2, 2])): 
        # alpha = np.arctan(s[2, 2]*sin(alpha) / s[2, 2]*cos(alpha))
        alpha = np.arctan(m[2, 1] / m[2, 2])
        tan = m[2, 1] / m[2, 2]
        sin = m[2, 1]
        print('The order is a rotation followed by a scaling, followed by a translation.')

      # if tan > 0, then alpha belongs to [0, 1/2 * pi] union [pi, 3/2 * pi]
      if tan > 0:
        # sin(alpha) < 0, alpha belongs to [pi, 3/2 * pi], not [0, 1/2 * pi]
        if sin < 0:
          alpha = alpha + np.pi
      # if tan < 0, then alpha belongs to [1/2 * pi, pi] union [3/2 * pi, 2 * pi]
      if tan < 0:
        # sin(alpha) > 0, alpha belongs to [1/2 * pi, pi], not [3/2 * pi, 2 * pi]
        if sin > 0:
          alpha = alpha + np.pi
        else:
          alpha = alpha + 2 * np.pi
      r[1, 1] = np.cos(alpha)
      r[1, 2] = -np.sin(alpha)
      r[2, 1] = np.sin(alpha)
      r[2, 2] = np.cos(alpha)
      s[2, 2] = m[2, 2] / r[2, 2]
      s[1, 1] = m[1, 1] / r[1, 1]
      return (t, s, r)


    # the rotation axis is y
    if m[0, 0] != 0 and m[0, 2] != 0 and m[2, 0] != 0 and m[2, 2] != 0:
      # observe matrix m, we can find that t[0:3, 3] = m[0:3, 3], s[1, 1] = m[1, 1]
      t[0:3, 3] = m[0:3, 3]
      s[1, 1] = m[1, 1]
      tan = 0
      sin = 0
      alpha = 0

      # decide wheather the order is a scaling followed by a rotation, followed by a translation (first s, then r, last t)
      if np.allclose((m[0, 2] / m[2, 2]), (- m[2, 0] / m[0, 0])):      
        # alpha = np.arctan(s[2, 2]*sin(alpha) / s[2, 2]*cos(alpha))
        alpha = np.arctan(m[0, 2] / m[2, 2])
        tan = m[0, 2] / m[2, 2]
        sin = m[0, 2]
        print('The order is a scaling followed by a rotation, followed by a translation.')

      # decide wheather the order is a rotation followed by a scaling, followed by a translation (first r, then s, last t)
      if np.allclose((- m[2, 0] / m[2, 2]), (m[0, 2] / m[0, 0])): 
        # alpha = np.arctan(s[0, 0]*sin(alpha) / s[0, 0]*cos(alpha))
        alpha = np.arctan(m[0, 2] / m[0, 0])
        tan = m[0, 2] / m[0, 0]
        sin = m[0, 2]
        print('The order is a rotation followed by a scaling, followed by a translation.')

      # if tan > 0, then alpha belongs to [0, 1/2 * pi] union [pi, 3/2 * pi]
      if (m[0, 2] / m[0, 0]) > 0:
        # sin(alpha) < 0, alpha belongs to [pi, 3/2 * pi], not [0, 1/2 * pi]
        if m[0, 2] < 0:
          alpha = alpha + np.pi
      # if tan < 0, then alpha belongs to [1/2 * pi, pi] union [3/2 * pi, 2 * pi]
      if (m[0, 2] / m[0, 0]) < 0:
        # sin(alpha) > 0, alpha belongs to [1/2 * pi, pi], not [3/2 * pi, 2 * pi]
        if m[0, 2] > 0:
          alpha = alpha + np.pi
        else:
          alpha = alpha + 2 * np.pi
      r[0, 0] = np.cos(alpha)
      r[0, 2] = np.sin(alpha)
      r[2, 0] = -np.sin(alpha)
      r[2, 2] = np.cos(alpha)
      s[2, 2] = m[2, 2] / r[2, 2]
      s[0, 0] = m[0, 0] / r[0, 0]
      return (t, s, r)



#########################
#      Main function    #
#########################
def main():
    #pass
    print('Exercise 1: ')
    print('Case 1: ')
    part1Case1()
    print('Case 2: ')
    part1Case2()
    print('Case 3: ')
    part1Case3()  


    print('Exercise 2: ')
    radius = np.random.randint(low=1, high=10)
    number = np.random.randint(low=5, high=50)

    # create spherical_coordinates
    spherical_coordinates = generateRandomSphere(radius, number)
    # transform spherical to catesian
    sphere1 = sphericalToCatesian(spherical_coordinates)
    # get transformed homogeneous coordinates, composite transformation matrix,
    # translation matrix, scaling matrix, and rotation matrix
    (sphere2, m1, t1, s1, r1) = applyRandomTransformation(sphere1)
    # calculate the transformation matrix m2 from sphere1 and sphere2
    m2 = calculateTransformation(sphere1, sphere2)

    print('The transformation matrix returned from applyRandomTransformation: ')
    print(m1)
    print('The transformation matrix calculated by svd: ')
    print(m2)
    print('')

    # decide if m1 and m2 are same
    if np.allclose(m1, m2):
      print('The calculated composite transformation matrix is the same from the original composite transformation matrix.')
    else:
      print('The calculated composite transformation matrix is not the same from the original composite transformation matrix.')

    print('')
    # calculate t2, s2, r2 from composite transformation matrix m1
    (t2, s2, r2) = decomposeTransformation(m2)
    
    print('')
    print('t1: ')
    print(t1)
    print('t2: ')
    print(t2)

    print('')
    print('r1: ')
    print(r1)
    print('r2: ')
    print(r2)

    print('')
    print('s1: ')
    print(s1)
    print('s2: ')
    print(s2)
    print('')
    
    # decide if t1 and t2, s1 and s2, r1 and r2 are same
    if np.allclose(t1, t2) and np.allclose(s1, s2) and np.allclose(r1, r2):
      print('The three components from decomposition is the same from the ones from applyRandomTransformation.')
    else:
      new_s1 = np.copy(s1)
      new_s2 = np.copy(s2)
      new_r1 = np.copy(r1)
      new_r2 = np.copy(r2)
      # get the absolute matrices of the matrices
      for i in range(len(s1)):
        for j in range(len(s1)):
          new_s1[i, j] = np.abs(s1[i, j])
      for i in range(len(s2)):
        for j in range(len(s2)):
          new_s2[i, j] = np.abs(s2[i, j])
      for i in range(len(r1)):
        for j in range(len(r1)):
          new_r1[i, j] = np.abs(r1[i, j])
      for i in range(len(r2)):
        for j in range(len(r2)):
          new_r2[i, j] = np.abs(r2[i, j])
      # if the absolute matrices are same with the original absolute matrices
      if np.allclose(t1, t2) and np.allclose(new_s1, new_s2) and np.allclose(new_r1, new_r2):
        # they have the same effect, actually they are the same
        print('Flipping the sign of scale has the same effect as an additional rotation of pi.')
        print('Thus, the effect of the three components from decompositions is the same from the ones from applyRandomTransformation.')
      else:
        print('The three components from decomposition is not the same from the ones from applyRandomTransformation.')


if __name__ == "__main__":
    main()