"""
Author: Minglang Yin, minglang_yin@brown.edu
Generate tfile with distributed boundary condition.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

def Fourier(n, x):
    a, b = [], []
    y = np.zeros((x.shape[0]))
    down, up = 0.003, 0.006
    for i in range(0, n):
        if i == 0:
            aa, bb = np.random.uniform(down, up), 0
        else:
            # aa, bb = np.random.uniform(down, up), np.random.uniform(down, up)
            aa, bb = np.random.uniform(0.0005, 0.001), np.random.uniform(0.0005, 0.001)
        a.append(aa)
        b.append(bb)
        y = y + aa*np.cos(i*x) + bb*np.sin(i*x)
    return a, b, y 
        
def main():
    ## define parameters
    x = np.linspace(0, 10, 100)
    for i in range(0, 900):
        print(f'printing {i}th file')

        totalFourier = np.random.randint(1, 5)
        print(totalFourier)

        a, b, y = Fourier(totalFourier, x)

        # ## write new tfile
        f = open("tfile_center/"+str(i), "a")
        f.write("DeltaRatio 4.0 \n")
        f.write("TotalTime 10.0 \n")
        f.write("TotalStep 1000 \n")
        f.write("DomainSizeX 10. \n")
        f.write("DomainSizeY 4. \n")
        f.write("MeshSize 0.025 \n")
        f.write("YoungsModulus 72. \n")
        f.write("PoissonRatio 0.23 \n")
        f.write("FractureEnergy 3.8e-7 \n")
        f.write("MaterilDensity 2.44 \n")
        f.write("OuputFreq 10 \n")
        f.write("CrackLocationY 0.0 \n")
        f.write("CrackLength 3 \n")
        f.write("totalFourier %d \n" % totalFourier)

        for line in range(0, totalFourier):
            f.write("%f %f \n" % (a[line], b[line]))

        f.close()

        # ## figure
        fig = plt.figure(constrained_layout=False, figsize=(5, 5))
        gs = fig.add_gridspec(1, 1)

        ax = fig.add_subplot(gs[0])
        h = ax.plot(x, y)
        ax.set_title('force')
        fig.savefig('figs/force' + str(i) + '.png')
        plt.close()

        ## save y
        tmp = np.concatenate((x[:, None], y[:, None]), axis=1)
        np.savetxt('force_center/' + str(i) + '.txt', tmp)

    


if __name__ == "__main__":
    main()
