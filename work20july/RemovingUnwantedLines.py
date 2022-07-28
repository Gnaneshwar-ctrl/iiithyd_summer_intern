import numpy as np

for i in range(100,10000):
	n1=np.loadtxt("./new/min_config.10000000."+str(i)+".data",dtype='float',skiprows=1018)
	np.savetxt("./RemovedUnwantedLines/min_config.10000000"+str(i)+".data",n1)