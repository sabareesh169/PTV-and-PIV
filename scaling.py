def rescale(array):
    rescaled = (array - np.mean(array))/ np.std(array)
    return rescaled, np.mean(array), np.std(array)
    
def rescale_test(array, mean, sigma):
    return (array - mean)/ sigma
    
