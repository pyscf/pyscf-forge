import numpy

def uks_sf_gga_wv1(rho1, fxc_sf,weight):
    # fxc_sf with a shape (4,4,ngrid), 4 means I, \nabla_x,y,z.
    rho1_ab,rho1_ba = rho1
    ngrid = weight.shape[-1]
    wv_ab, wv_ba = numpy.empty((2,4,ngrid))
    wv_ab = numpy.einsum('yp,xyp->xp',  rho1_ab,fxc_sf)
    wv_ba = numpy.einsum('yp,xyp->xp',  rho1_ba,fxc_sf)
    # wv_ab[0] = wv_ab[0] *2 *.5 # *2 bacause of kernel, *0.5 for the (x + x.T)*0.5
    # wv_ba[0] = wv_ba[0] *2 *.5

    # Don't forget (sigma_x sigma_x + sigma_y sigma_y) needs *2 for kernel term.
    wv_ab[1:] *=2.0
    wv_ba[1:] *=2.0
    return wv_ab*weight, wv_ba*weight

def uks_sf_gga_wv2_p(rho1, kxc_sf,weight):
    # kxc_sf with a shape (4,4,2,4,ngrid), 4 means I,\nabla_x,y,z, 
    # 0: n, \nabla_x,y,z n;  1: s, \nabla_x,y,z s.
    rho1_ab,rho1_ba = rho1
    ngrid = weight.shape[-1]
    gv_ab, gv_ba = numpy.empty((2,2,4,ngrid))
    # Note *2 and *0.5 like in function uks_sf_gga_wv1
    gv_ab = numpy.einsum('xp,yp,xyvzp->vzp', rho1_ab, rho1_ab+rho1_ba, kxc_sf, optimize=True)
    gv_ba = numpy.einsum('xp,yp,xyvzp->vzp', rho1_ba, rho1_ba+rho1_ab, kxc_sf, optimize=True)

    gv_ab[0,1:] *=2.0
    gv_ab[1,1:] *=2.0
    gv_ba[0,1:] *=2.0
    gv_ba[1,1:] *=2.0
    return gv_ab*weight, gv_ba*weight

def uks_sf_gga_wv2_m(rho1, kxc_sf,weight):
    rho1_ab,rho1_ba = rho1
    ngrid = weight.shape[-1]
    gv_ab, gv_ba = numpy.empty((2,2,5,ngrid))
    # Note *2 and *0.5 like in function uks_sf_mgga_wv1
    gv_ab = numpy.einsum('xp,yp,xyvzp->vzp', rho1_ab, rho1_ab-rho1_ba, kxc_sf , optimize=True)
    gv_ba = numpy.einsum('xp,yp,xyvzp->vzp', rho1_ba, rho1_ba-rho1_ab, kxc_sf , optimize=True)

    gv_ab[:,1:] *=2.0
    gv_ba[:,1:] *=2.0
    return gv_ab*weight, gv_ba*weight

def uks_sf_mgga_wv1(rho1, fxc_sf,weight):
    rho1_ab,rho1_ba = rho1
    # fxc_sf with a shape (5,5,ngrid), 5 means I, \nabla_x,y,z s, u
    # s_s, s_Ns, Ns_s, Ns_Ns, s_u, u_s, u_Ns, Ns_u, u_u
    ngrid = weight.shape[-1]
    wv_ab, wv_ba = numpy.empty((2,5,ngrid))
    wv_ab = numpy.einsum('yp,xyp->xp',  rho1_ab,fxc_sf)
    wv_ba = numpy.einsum('yp,xyp->xp',  rho1_ba,fxc_sf)
    # wv_ab[0] = wv_ab[0] *2 *.5 # *2 bacause of kernel, *0.5 for the (x + x.T)*0.5
    # wv_ba[0] = wv_ba[0] *2 *.5

    # Don't forget (sigma_x sigma_x + sigma_y sigma_y) needs *2 for kernel term.
    wv_ab[1:4] *=2.0
    wv_ba[1:4] *=2.0
    # *0.5 below is for tau->ao
    wv_ab[4] *= 0.5
    wv_ba[4] *= 0.5
    return wv_ab*weight, wv_ba*weight

def uks_sf_mgga_wv2_p(rho1, kxc_sf,weight):
    rho1_ab,rho1_ba = rho1
    # kxc_sf with a shape (5,5,2,5,ngrid), 5 means s \nabla_x,y,z s, u
    # s_s    ->  0: n, \nabla_x,y,z n, tau ;  1: s, \nabla_x,y,z s, u
    # s_Ns   ->
    # Ns_s   ->
    # Ns_Ns  ->
    # s_u    ->
    # u_s    ->
    # u_Ns   ->
    # Ns_u   ->
    # u_u    ->
    ngrid = weight.shape[-1]
    gv_ab, gv_ba = numpy.empty((2,2,5,ngrid))
    # Note *2 and *0.5 like in function uks_sf_mgga_wv1
    gv_ab = numpy.einsum('xp,yp,xyvzp->vzp', rho1_ab, rho1_ab+rho1_ba, kxc_sf, optimize=True)
    gv_ba = numpy.einsum('xp,yp,xyvzp->vzp', rho1_ba, rho1_ba+rho1_ab, kxc_sf, optimize=True)

    gv_ab[:,1:4] *=2.0
    gv_ba[:,1:4] *=2.0
    gv_ab[:,4] *= 0.5
    gv_ba[:,4] *= 0.5
    return gv_ab*weight, gv_ba*weight

def uks_sf_mgga_wv2_m(rho1, kxc_sf,weight):
    rho1_ab,rho1_ba = rho1
    ngrid = weight.shape[-1]
    gv_ab, gv_ba = numpy.empty((2,2,5,ngrid))
    # Note *2 and *0.5 like in function uks_sf_mgga_wv1
    gv_ab = numpy.einsum('xp,yp,xyvzp->vzp', rho1_ab, rho1_ab-rho1_ba, kxc_sf , optimize=True)
    gv_ba = numpy.einsum('xp,yp,xyvzp->vzp', rho1_ba, rho1_ba-rho1_ab, kxc_sf , optimize=True)

    gv_ab[:,1:4] *=2.0
    gv_ba[:,1:4] *=2.0
    gv_ab[:,4] *= 0.5
    gv_ba[:,4] *= 0.5
    return gv_ab*weight, gv_ba*weight
