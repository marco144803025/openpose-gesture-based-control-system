import numpy as np

def flatten(array):
    x_coordinates = list(array[0,:])
    y_coordinates = list(array[1,:])

    # Reshape to match the dimensions
    x_coordinates_reshaped = np.array(x_coordinates).reshape(21, 1)
    y_coordinates_reshaped = np.array(y_coordinates).reshape(21, 1)

    # Concatenate along columns (axis 1)
    flattenedArray = np.hstack((x_coordinates_reshaped, y_coordinates_reshaped))
    flattenedArray = np.ravel(flattenedArray)
    return flattenedArray

array=np.array([[1, 2 ,3],[4 ,5 ,6]])
newArray=flatten(array)
print(newArray)