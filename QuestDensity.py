import numpy
import h5py


def get_densities(fname, tol=1e-12):
    with h5py.File(fname) as inp:
        rho = numpy.array(inp['rho'])
        idxs = rho[:,0]+rho[:,1] > tol  # Filter out really small densities
        grd = numpy.array(inp['grd'])
        vabs = lambda a,b: numpy.sum(numpy.multiply(a,b),axis=0)
        ga = numpy.sum(grd[:,0:3]*grd[:,0:3],axis=1)
        gb = numpy.sum(grd[:,4:7]*grd[:,4:7],axis=1)
        # gaa = numpy.sum(numpy.multiply(ga,ga),axis=0)
        # gbb = numpy.sum(numpy.multiply(gb,gb),axis=0)
        # gab = numpy.sum(numpy.multiply(ga,gb),axis=0)
        # gtt = gaa + gbb + 2*gab
        tau = numpy.array(inp['tau'])
        lap = numpy.array(inp['lap'])
        xyz = numpy.array(inp['xyz'])

        out = numpy.zeros((rho[idxs].shape[0], 8))
        out[:,0] = rho[idxs,0]
        out[:,1] = rho[idxs,1]
        out[:,2] = numpy.sqrt(ga[idxs])
        out[:,3] = numpy.sqrt(gb[idxs])
        out[:,4] = tau[idxs,0]
        out[:,5] = tau[idxs,1]
        out[:,6] = lap[idxs,0]
        out[:,7] = lap[idxs,1]
        # out[:,8] = numpy.sqrt(gtt[idxs])
    return out

if __name__ == "__main__":
    atom = get_densities("HCOOH.out.plot")
    print(atom.shape)