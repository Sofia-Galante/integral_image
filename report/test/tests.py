import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

n = 10

t1_cuda = list()
t1_omp = list()
t2_10 = list()
t2_100 = list()
t2_1000 = list()
t2_mul32 = list()
t3_level = list()
t3_velocity = list()

os.makedirs("dimTest", exist_ok=True)
os.makedirs("dimTest/cuda", exist_ok=True)
os.makedirs("dimTest/omp", exist_ok=True)
os.makedirs("gridTest", exist_ok=True)
os.makedirs("gridTest/mul32", exist_ok=True)
os.makedirs("gridTest/size10", exist_ok=True)
os.makedirs("gridTest/size100", exist_ok=True)
os.makedirs("gridTest/size1000", exist_ok=True)
os.makedirs("threadsTest", exist_ok=True)
os.makedirs("threadsTest/levelsTest", exist_ok=True)
os.makedirs("threadsTest/velocityTest", exist_ok=True)

for i in range(n):
    filename = f"test_{i+1}.csv"
    os.system(f"integral_image_cuda {filename}")
    os.system(f"integral_image_omp {filename}")
    t1_cuda.append(pd.read_csv(f"dimTest/cuda/{filename}", names = ['x', 'seq', 'cuda'], sep=';'))
    t1_omp.append(pd.read_csv(f"dimTest/omp/{filename}", names = ['x', 'omp'], sep=';'))
    t2_10.append(pd.read_csv(f"gridTest/size10/{filename}", names = ['x', 'y', 'cuda1', 'cuda2'], sep=';'))
    t2_100.append(pd.read_csv(f"gridTest/size100/{filename}", names = ['x', 'y', 'cuda1', 'cuda2'], sep=';'))
    t2_1000.append(pd.read_csv(f"gridTest/size1000/{filename}", names = ['x', 'y', 'cuda1', 'cuda2'], sep=';'))
    t2_mul32.append(pd.read_csv(f"gridTest/mul32/{filename}", names = ['x', 'cuda1_1', 'cuda2_1', 'cuda1_2', 'cuda2_2', 'cuda1_3', 'cuda2_3'], sep=';'))
    t3_level.append(pd.read_csv(f"threadsTest/levelsTest/{filename}", names = ['x', 'omp1', 'omp2', 'omp3', 'omp4', 'omp5'], sep=';'))
    t3_velocity.append(pd.read_csv(f"threadsTest/velocityTest/{filename}", names = ['x', 'omp1', 'omp2', 'omp3'], sep=';'))

def compute_mean(df_list, filename):
    df = pd.concat(df_list).reset_index(drop=True)
    for col in df.columns:
        if col != "x":
            df.loc[df.groupby("x")[col].idxmin(), col] = np.nan
            df.loc[df.groupby("x")[col].idxmax(), col] = np.nan
    df = df.groupby("x").mean().reset_index()
    for col in df.columns:
        if col != "x":
            df[col] = df[col].round(decimals = 3)
    df.to_csv(filename, header = None, index = None, sep = ';')
    return df

def compute_mean_sizeTest(df_list, filename):
    df = pd.concat(df_list).reset_index(drop=True)
    for col in df.columns:
        if col != "x" and col != "y":
            df.loc[df.groupby("x")[col].idxmin(), col] = np.nan
            df.loc[df.groupby("x")[col].idxmax(), col] = np.nan
    df = df.groupby(["x", "y"]).mean().reset_index()
    for col in df.columns:
        if col != "x" and col != "y" :
            df[col] = df[col].round(decimals = 3)
    df = df.drop(['x'], axis=1)
    df[['cuda1', 'cuda2']] = df[['cuda1', 'cuda2']].astype(str) + " ms"
    df.to_csv(filename, header = [' ', 'GPU - global', 'GPU - shared'], index = None, sep = ';')
    return df


t1_cuda = compute_mean(t1_cuda, "dimTest/cuda/result.csv")
t1_omp = compute_mean(t1_omp, "dimTest/omp/result.csv")
t2_10 = compute_mean_sizeTest(t2_10, "gridTest/size10/result.csv")
t2_100 = compute_mean_sizeTest(t2_100, "gridTest/size100/result.csv")
t2_1000 = compute_mean_sizeTest(t2_1000, "gridTest/size1000/result.csv")
t2_mul32 = compute_mean(t2_mul32, "gridTest/mul32/result.csv")
t3_level = compute_mean(t3_level, "threadsTest/levelsTest/result.csv")
t3_velocity = compute_mean(t3_velocity, "threadsTest/velocityTest/result.csv")



plt.plot(t1_cuda["x"], t1_cuda["seq"], color = 'blue', label = 'sequenziale')
plt.plot(t1_cuda["x"], t1_cuda["cuda"], color = 'green', label = 'CUDA - shared memory')
plt.plot(t1_omp["x"], t1_omp["omp"], color = 'red', label = 'OpenMP - primo livello')
plt.xlabel("dimensioni immagine (numero di pixel)")
plt.ylabel("tempo in ms")
plt.yscale('log')
plt.legend(loc="upper left")
plt.savefig("dimTest/result.png")
plt.clf()

plt.plot(t2_mul32["x"], t2_mul32["cuda1_1"], color = 'blue', label = 'threads = 8 x 8')
plt.plot(t2_mul32["x"], t2_mul32["cuda1_2"], color = 'red', label = 'threads = 16 x 16')
plt.plot(t2_mul32["x"], t2_mul32["cuda1_3"], color = 'green', label = 'threads = 32 x 32')
plt.xlabel("dimensioni immagine (numero di pixel)")
plt.ylabel("tempo in ms")
plt.legend(loc="upper left")
plt.savefig("gridTest/mul32/global_result.png")
plt.clf()

plt.plot(t2_mul32["x"], t2_mul32["cuda2_1"], color = 'blue', label = 'threads = 8 x 8')
plt.plot(t2_mul32["x"], t2_mul32["cuda2_2"], color = 'red', label = 'threads = 16 x 16')
plt.plot(t2_mul32["x"], t2_mul32["cuda2_3"], color = 'green', label = 'threads = 32 x 32')
plt.xlabel("dimensioni immagine (numero di pixel)")
plt.ylabel("tempo in ms")
plt.legend(loc="upper left")
plt.savefig("gridTest/mul32/shared_result.png")
plt.clf()

plt.plot(t2_mul32["x"], t2_mul32["cuda2_1"], color = 'blue', label = 'shared - 8 x 8')
plt.plot(t2_mul32["x"], t2_mul32["cuda1_1"], color = 'purple', label = 'global - 8 x 8')
plt.plot(t2_mul32["x"], t2_mul32["cuda2_2"], color = 'red', label = 'shared - 16 x 16')
plt.plot(t2_mul32["x"], t2_mul32["cuda1_2"], color = 'orange', label = 'global - 16 x 16')
plt.plot(t2_mul32["x"], t2_mul32["cuda2_3"], color = 'green', label = 'shared - 32 x 32')
plt.plot(t2_mul32["x"], t2_mul32["cuda1_3"], color = 'black', label = 'global - 32 x 32')
plt.xlabel("dimensioni immagine (numero di pixel)")
plt.ylabel("tempo in ms")
plt.legend(loc="upper left")
plt.savefig("gridTest/mul32/result.png")
plt.clf()

plt.plot(t3_level["x"], t3_level["omp1"], color = 'blue', label = 'entrambi i livelli: 1 - 15')
plt.plot(t3_level["x"], t3_level["omp2"], color = 'red', label = 'entrambi i livelli: 15 - 1')
plt.plot(t3_level["x"], t3_level["omp3"], color = 'green', label = 'primo livello: 15')
plt.plot(t3_level["x"], t3_level["omp4"], color = 'purple', label = 'secondo livello: 15')
plt.plot(t3_level["x"], t3_level["omp5"], color = 'orange', label = 'entrambi i livelli: 15 - 15')
plt.yscale('log')
plt.xlabel("dimensioni immagine (numero di pixel)")
plt.ylabel("tempo in ms")
plt.legend(loc="upper left")
plt.savefig("threadsTest/levelsTest/result.png")
plt.clf()

plt.plot(t3_velocity["x"], t3_velocity["omp1"], color = 'blue', label = 'threads = 10')
plt.plot(t3_velocity["x"], t3_velocity["omp2"], color = 'red', label = 'threads = 100')
plt.plot(t3_velocity["x"], t3_velocity["omp3"], color = 'green', label = 'threads = 1000')
plt.xlabel("dimensioni immagine (numero di pixel)")
plt.ylabel("tempo in ms")
plt.legend(loc="upper left")
plt.savefig("threadsTest/velocityTest/result.png")
plt.clf()
