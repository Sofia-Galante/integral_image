import matplotlib.pyplot as plt
import pandas as pd

t1_cuda = pd.read_csv('test1_cuda.csv', names = ['x', 'seq', 'cuda1', 'cuda2'], sep=';')
t1_omp = pd.read_csv('test1_omp.csv', names = ['x', 'omp'], sep=';')

plt.plot(t1_cuda["x"], t1_cuda["seq"], color = 'blue', label = 'sequenziale')
plt.plot(t1_cuda["x"], t1_cuda["cuda1"], color = 'red', label = 'CUDA - global memory')
plt.plot(t1_cuda["x"], t1_cuda["cuda2"], color = 'green', label = 'CUDA - shared memory')
plt.plot(t1_omp["x"], t1_omp["omp"], color = 'purple', label = 'OpenMP')
plt.yscale('log')
plt.xlabel("dimensioni immagine (numero di pixel)")
plt.ylabel("tempo in ms")
plt.legend(loc="upper left")
plt.savefig("test1.png")
plt.show()
