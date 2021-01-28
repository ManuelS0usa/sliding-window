from numpy import load, savez, array, arange, tile, hstack, vstack, random, delete


class Cria_Lags:
	
	def varlag(self, data, lags):

		'''
		dl=varlag(data,lags)

		Return a matrix with lagged vectors from data (dl), corresponding to the lags provided.
		Negative value lags are interpreted as future values, 0 is the "present".
		Lagged vectors are given sorted in ascending order of the columns of dl.
		'''
	
		lags.sort()
		maxlag = lags[-1]
		minlag = lags[0]

		if (minlag >= 0):
			N = data.size-maxlag 
		else:
			N = data.size-maxlag+minlag
		
		return data[tile(arange(maxlag,maxlag+N,1).reshape(N,1),(1,lags.size))-tile(lags,(N,1))]


	def finalmatrix_dividerand(self, matrix):
		""" Build matrices spliting the dataset randomly shuffled. This removes the dataset temporal order from training purposes. """
		
		trainpoints = round(matrix.shape[0] * 0.60)
		testepoints = round((matrix.shape[0] - trainpoints) * 0.50)
		validationspoints = matrix.shape[0] - trainpoints - testepoints

		lista = []
		i=0
		while len(lista) < matrix.shape[0]:
			r=random.randint(0, matrix.shape[0])
			if r not in lista: 
				lista.append(r)
			i+=1

		ind_matrix_train = lista[:trainpoints]
		ind_matrix_teste = lista[trainpoints:trainpoints+testepoints]
		ind_matrix_validation = lista[trainpoints+testepoints:]

		for x in range(trainpoints):
			if x == 0:
				finalmatrixtrain = matrix[ind_matrix_train[x],:].copy()
			else:
				finalmatrixtrain = vstack((finalmatrixtrain,matrix[ind_matrix_train[x],:].copy()))

		for x in range(testepoints):
			if x == 0:
				finalmatrixtest = matrix[ind_matrix_teste[x],:].copy()
			else:
				finalmatrixtest = vstack((finalmatrixtest,matrix[ind_matrix_teste[x],:].copy()))

		for x in range(validationspoints):
			if x == 0:
				finalmatrixvalidation = matrix[ind_matrix_validation[x],:].copy()
			else:
				finalmatrixvalidation = vstack((finalmatrixvalidation,matrix[ind_matrix_validation[x],:].copy()))

		return {'training': finalmatrixtrain, 'testing': finalmatrixtest, 'validation': finalmatrixvalidation}


	def finalmatrix_divideblock(self, matrix):
		""" Build matrices spliting the dataset by blocks. This keeps the dataset temporal order. """
		
		trainpoints = round(matrix.shape[0] * 0.60)
		testepoints = round((matrix.shape[0] - trainpoints) * 0.50)
		validationspoints = matrix.shape[0] - trainpoints - testepoints
		
	
		for x in range(int(trainpoints)):
			if x == 0:
				finalmatrixtrain = matrix[x,:].copy()
			else:
				finalmatrixtrain = vstack((finalmatrixtrain,matrix[x,:].copy()))


		for x in range(int(trainpoints), int(testepoints + trainpoints)):
			if x == trainpoints:
				finalmatrixtest = matrix[x,:].copy()
			else:
				finalmatrixtest = vstack((finalmatrixtest,matrix[x,:].copy()))


		for x in range(int(testepoints + trainpoints), int(testepoints + trainpoints + validationspoints)):
			if x == trainpoints+testepoints:
				finalmatrixvalidation = matrix[x,:].copy()
			else:
				finalmatrixvalidation = vstack((finalmatrixvalidation,matrix[x,:].copy()))

		return {'training': finalmatrixtrain, 'testing': finalmatrixtest, 'validation': finalmatrixvalidation}
	
		
	''' Dataset normalization (compress data in a numeric range)

		x - dataset value
		m - minimum value in the dataset
		M - maximum value in the dataset
	'''

	def scale(self,x,m,M):
	    return (x-m)/(M-m)


	def unscale(self,x,m,M):
	    return x*(M-m)+m
