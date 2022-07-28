import numpy as np

for i in range(1000,1001):
	n1=np.loadtxt("./new/min_config.500000."+str(i)+".data",dtype='float',skiprows=1018)
	np.savetxt("./RemovedUnwantedLines/min_config."+str(i)+".data",n1)

