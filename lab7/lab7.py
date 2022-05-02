import numpy as np

def distance(data, centers):
    '''
    Calculate the distances from points to cluster centers.
      parameter:
        data: nparray of shape (n, 2)
        centers: nparray of shape (m, 2)
      return:
        distance: nparray of shape (n, m)
    '''
    #pass
    # get length of data (n)
    (n,_) = np.shape(data)
    # get length of centers (m)
    (m,_) = np.shape(centers)
    # initial distance nparray (n x m)
    distance = np.zeros((n, m))
    for i in range(len(data)):
      for j in range(len(centers)):
        # calculate the distance between data[i, 0] and centers[j, 0]
        distance[i, j] = np.sqrt((data[i, 0] - centers[j, 0]) ** 2 + (data[i, 1] - centers[j, 1]) ** 2)
    return distance


def kmeans(data, n_centers):
    """
    Divide data into n_centers clusters, return the cluster centers and assignments.
    Assignments are the indices of the closest cluster center for the corresponding data point.
      parameter:
        data: nparray of shape (n, 2)
        n_centers: int
      return:
        centers: nparray of shape (n_centers, 2)
        assignments: nparray of shape (n,)
    """
    #pass
    # get length of data (n)
    (n,_) = np.shape(data)
    # initial assignments nparray (n,)
    assignments = np.zeros((n,))
    #total_number_of_data = len(data)
    #random_value = np.random.randint(0, total_number_of_data - n_centers + 1)
    centers = np.zeros((n_centers, 2))
    #new_centers = data[random_value:(random_value+n_centers),:]
    new_centers = data[0:(0+n_centers),:]

    # while old centers and new centers are not same
    while not np.allclose(centers, new_centers):
      # assign new centers to old centers first
      centers = new_centers
      # get distance nparray dist
      dist = distance(data, centers)
      # get row number and column number of dist
      (row, col) = np.shape(dist)
      for i in range(row):
        # set the initial value of 'smallest' for every row to the first number of this row
        smallest = dist[i, 0]
        # set corresponding indice = 0
        indice = 0
        # update smallest and the corresponding indice
        for j in range(col):
          if dist[i, j] < smallest:
            smallest = dist[i, j]
            indice = j
        # assign indice to assignments[i]
        assignments[i] = indice
    
      for j in range(col):
        # set the initial value of total_x, total_y, and number
        total_x = 0
        total_y = 0
        number = 0
        for i in range(row):
          # if data[i] is closest to the jth cluster center
          if assignments[i] == j:
            # add data[i, 0] to total_x
            total_x = total_x + data[i, 0]
            # add data[i, 1] to total_y
            total_y = total_y + data[i, 1]
            # number increase by 1
            number = number + 1
        # if number is not 0
        if number != 0:
          mean_total_x = total_x / number
          mean_total_y = total_y / number
          # update jth center
          new_centers[j, 0] = mean_total_x
          new_centers[j, 1] = mean_total_y
        # if number is 0
        else:
          # do not update jth center
          new_centers[j, 0] = centers[j, 0]
          new_centers[j, 1] = centers[j, 1]

    return centers, assignments


def distortion(data, centers, assignments):
    """
    Calculate the distortion of the clustering.
      parameter:
        data: nparray of shape (n, 2)
        centers: nparray of shape (m, 2)
        assignments: nparray of shape (n,)
      return:
        distortion: float
    """
    #pass
    # initial distortion
    distortion = 0
    total_data_number = len(assignments)
    for i in range(total_data_number):
      # get data[i]
      x_value = data[i, 0]
      y_value = data[i, 1]
      # get corresponding assignment for data[i]
      indice = int(assignments[i])
      # mu is cluster center
      mu_x = centers[indice, 0]
      mu_y = centers[indice, 1]
      # calculate the sum of squares of each cluster from its cluster center
      distortion = distortion + np.sqrt((x_value - mu_x) ** 2 + (y_value - mu_y) ** 2)
    
    return distortion



if __name__ == "__main__":
  # test your code here 
  # load 'lab7.npy' data
  data = np.load('./lab7.npy')
  # set the value of n_centers
  n_centers = 20
  # get centers and assignments by kmeans(data, n_centers)
  centers, assignments = kmeans(data, n_centers)
  # print final cluster centers
  print('Final cluster centers are: ')
  print(centers)
  print('')
  # print distortion
  distor = distortion(data, centers, assignments)
  print('Distortion: ', distor)


