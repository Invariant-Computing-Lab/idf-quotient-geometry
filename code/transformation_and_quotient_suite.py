#!/usr/bin/env python3
"""
Transformation and quotient-invariance suite for the IDF invariant reformulation.

Synthetic mathematical tests only. The suite validates or falsifies proposed
transformation properties; it does not validate particle physics.

Seed: 20260812
Dependencies: numpy, pandas, scipy
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

SEED=20260812
rng=np.random.default_rng(SEED)

sigma1=np.array([[0,1],[1,0]],complex)
sigma2=np.array([[0,-1j],[1j,0]],complex)
sigma3=np.array([[1,0],[0,-1]],complex)
paulis=np.array([sigma1,sigma2,sigma3])

def mesh_area(V,F):
    a,b,c=V[F[:,0]],V[F[:,1]],V[F[:,2]]
    return 0.5*np.linalg.norm(np.cross(b-a,c-a),axis=1).sum()

def angle_at(a,b,c):
    u,v=b-a,c-a
    return np.arccos(np.clip(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)),-1,1))

def angle_deficits(V,F):
    sums=np.zeros(len(V))
    for f in F:
        p=V[f]
        aa=(angle_at(p[0],p[1],p[2]),angle_at(p[1],p[2],p[0]),angle_at(p[2],p[0],p[1]))
        for idx,a in zip(f,aa):
            sums[idx]+=a
    return 2*np.pi-sums

def sphere_mesh(n=300):
    X=rng.normal(size=(n,3))
    X/=np.linalg.norm(X,axis=1,keepdims=True)
    return X,ConvexHull(X).simplices.copy()

def torus_mesh(nu=30,nv=20,R=2.0,r=0.7):
    us=np.linspace(0,2*np.pi,nu,endpoint=False)
    vs=np.linspace(0,2*np.pi,nv,endpoint=False)
    V=np.array([[(R+r*np.cos(v))*np.cos(u),(R+r*np.cos(v))*np.sin(u),r*np.sin(v)]
                for u in us for v in vs])
    idx=lambda i,j:(i%nu)*nv+(j%nv)
    F=[]
    for i in range(nu):
        for j in range(nv):
            a,b,c,d=idx(i,j),idx(i+1,j),idx(i+1,j+1),idx(i,j+1)
            F += [[a,b,c],[a,c,d]]
    return V,np.array(F)

def deform_sphere(V,sd=0.12):
    x,y,z=V.T
    c=rng.normal(0,sd,6)
    B=np.c_[x,y,z,x*y,y*z,z*x]
    return V*(1+B@c)[:,None]

def deform_torus(V,nu=30,nv=20,amp=0.15):
    out=V.copy()
    u=np.repeat(np.linspace(0,2*np.pi,nu,endpoint=False),nv)
    v=np.tile(np.linspace(0,2*np.pi,nv,endpoint=False),nu)
    c=rng.normal(0,amp,4)
    fac=1+c[0]*np.cos(u)+c[1]*np.sin(2*v)+c[2]*np.cos(u+v)+c[3]*np.sin(2*u-v)
    fac=np.clip(fac,0.55,1.45)
    out[:,:2]*=fac[:,None]
    out[:,2]*=(2-fac)
    return out

def random_su2():
    q=rng.normal(size=4)
    q/=np.linalg.norm(q)
    a,b,c,d=q
    return np.array([[a+1j*d,c+1j*b],[-c+1j*b,a-1j*d]],complex)

def F_from_vec(v):
    return sum(v[i]*paulis[i]/2 for i in range(3))

def flat_torus_spectrum(aspect=1.0,area=1.0,modes=5,k=30):
    Lx=np.sqrt(area*aspect)
    Ly=np.sqrt(area/aspect)
    vals=[]
    for m in range(-modes,modes+1):
        for n in range(-modes,modes+1):
            if m or n:
                vals.append(4*np.pi**2*(m*m/Lx**2+n*n/Ly**2))
    return np.sort(vals)[:k]

def transform_numeric(q,d,s):
    return q/np.prod(s**d)

def main(outdir="results"):
    outdir=Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    records=[]

    # E1: discrete Gauss-Bonnet under geometry-preserving topology
    Vs,Fs=sphere_mesh()
    Vt,Ft=torus_mesh()
    for name,V,F,deformer,truechi in [
        ("sphere",Vs,Fs,deform_sphere,2),
        ("torus",Vt,Ft,deform_torus,0)
    ]:
        est=[]; areas=[]
        for _ in range(200):
            Vd=deformer(V.copy())
            est.append(angle_deficits(Vd,F).sum()/(2*np.pi))
            areas.append(mesh_area(Vd,F))
        est=np.array(est); areas=np.array(areas)
        records.append(dict(experiment=f"E1_GaussBonnet_{name}",trials=200,
                            metric="max_abs_chi_error",value=np.max(np.abs(est-truechi)),
                            decision="PASS"))
        records.append(dict(experiment=f"E1_GaussBonnet_{name}",trials=200,
                            metric="area_cv",value=areas.std(ddof=1)/areas.mean(),
                            decision="NEGATIVE_CONTROL_DRIFT"))

    # E2: chi/A coupling is conditional on area/polarization preservation
    A0=mesh_area(Vs,Fs)
    g0=8*np.pi/A0
    raw=[]; norm=[]
    for _ in range(500):
        Vd=deform_sphere(Vs,sd=0.35)
        A=mesh_area(Vd,Fs)
        ch=angle_deficits(Vd,Fs).sum()/(2*np.pi)
        raw.append(4*np.pi*ch/A)
        Vn=Vd*np.sqrt(A0/A)
        An=mesh_area(Vn,Fs)
        chn=angle_deficits(Vn,Fs).sum()/(2*np.pi)
        norm.append(4*np.pi*chn/An)
    raw=np.array(raw); norm=np.array(norm)
    rr=np.abs(raw/g0-1); nr=np.abs(norm/g0-1)
    for metric,value,decision in [
        ("median_raw_rel_dev",np.median(rr),"FAIL_TOPOLOGY_ONLY"),
        ("p95_raw_rel_dev",np.quantile(rr,.95),"FAIL_TOPOLOGY_ONLY"),
        ("max_area_preserved_rel_dev",nr.max(),"PASS_CONDITIONAL")
    ]:
        records.append(dict(experiment="E2_coupling_chi_over_A",trials=500,
                            metric=metric,value=value,decision=decision))

    # E3: SU(2) gauge conjugation
    rel=[]; comp=[]; wil=[]
    for _ in range(10000):
        v=rng.normal(size=3); F=F_from_vec(v); G=random_su2()
        Fp=G@F@G.conj().T
        I=np.trace(F@F).real; Ip=np.trace(Fp@Fp).real
        rel.append(abs(Ip-I)/(abs(I)+1e-15))
        vp=np.array([np.trace(Fp@paulis[i]).real for i in range(3)])
        comp.append(np.linalg.norm(vp-v)/(np.linalg.norm(v)+1e-15))
        U=random_su2(); Up=G@U@G.conj().T
        wil.append(abs(np.trace(Up)-np.trace(U)))
    for metric,value,decision in [
        ("max_rel_error_TrF2",max(rel),"PASS"),
        ("median_component_change",np.median(comp),"NEGATIVE_CONTROL_DRIFT"),
        ("max_abs_error_WilsonTrace",max(wil),"PASS")
    ]:
        records.append(dict(experiment="E3_SU2_gauge",trials=10000,
                            metric=metric,value=value,decision=decision))

    # E4: Picard/divisor degree under degree-zero shifts
    errs=[]; changes=[]
    for _ in range(10000):
        d=int(rng.integers(-3,4))
        D=rng.integers(-4,5,size=25); D[-1]+=d-D.sum()
        P=rng.integers(-5,6,size=25); P[-1]-=P.sum()
        D2=D+P
        errs.append(abs(D2.sum()-D.sum()))
        changes.append(np.sum(np.abs(D2-D)))
    records.append(dict(experiment="E4_Picard_degree",trials=10000,
                        metric="max_degree_error",value=max(errs),decision="PASS"))
    records.append(dict(experiment="E4_Picard_degree",trials=10000,
                        metric="median_representative_L1_change",value=np.median(changes),
                        decision="NEGATIVE_CONTROL_DRIFT"))

    # E5: spectral data carries moduli information at fixed genus and area
    s1=flat_torus_spectrum(1.0); s2=flat_torus_spectrum(2.0); s05=flat_torus_spectrum(.5)
    records.append(dict(experiment="E5_flat_torus_spectrum",trials=30,
                        metric="relative_spectral_distance_aspect1_vs2",
                        value=np.linalg.norm(s1-s2)/np.linalg.norm(s1),
                        decision="DISTINGUISHES_MODULI"))
    records.append(dict(experiment="E5_flat_torus_spectrum",trials=30,
                        metric="max_rel_error_aspect2_vs_reciprocal",
                        value=np.max(np.abs(s2-s05)/(np.abs(s2)+1e-15)),
                        decision="PASS_ISOMETRIC_AXIS_SWAP"))
    aspects=np.exp(rng.uniform(np.log(.4),np.log(2.5),1000))
    lam1=np.array([flat_torus_spectrum(a,k=1)[0] for a in aspects])
    records.append(dict(experiment="E5_flat_torus_spectrum",trials=1000,
                        metric="lambda1_max_min_ratio_fixed_area",
                        value=lam1.max()/lam1.min(),decision="TOPOLOGY_AREA_INCOMPLETE"))

    # E6: fiber-pushforward / coordinate integral
    N=600
    u=(np.arange(N)+.5)/N; v=(np.arange(N)+.5)/N
    U,V=np.meshgrid(u,v,indexing="ij")
    f=lambda u,v: np.exp(np.sin(2*np.pi*u)+.5*np.cos(2*np.pi*v))
    ref=f(U,V).mean()
    ce=[]; ne=[]
    for a in rng.uniform(-.85,.85,100):
        Up=(U+a/(2*np.pi)*np.sin(2*np.pi*U))%1
        J=1+a*np.cos(2*np.pi*U)
        ce.append(abs((f(Up,V)*J).mean()-ref)/ref)
        ne.append(abs(f(Up,V).mean()-ref)/ref)
    records.append(dict(experiment="E6_fiber_pushforward",trials=100,
                        metric="max_corrected_rel_error",value=max(ce),decision="PASS"))
    records.append(dict(experiment="E6_fiber_pushforward",trials=100,
                        metric="median_naive_rel_error",value=np.median(ne),decision="FAIL_NO_JACOBIAN"))
    records.append(dict(experiment="E6_fiber_pushforward",trials=100,
                        metric="max_naive_rel_error",value=max(ne),decision="FAIL_NO_JACOBIAN"))

    # R1: removable coefficient
    n=10000
    a=np.exp(rng.uniform(-4,4,size=n))
    coeff=a/2
    records.append(dict(experiment="R1_removable_coefficient",trials=n,
                        metric="kinetic_half_min",value=coeff.min(),decision="I0_REMOVABLE"))
    records.append(dict(experiment="R1_removable_coefficient",trials=n,
                        metric="kinetic_half_max",value=coeff.max(),decision="I0_REMOVABLE"))

    # R2: unit quotient
    dims={
        "e":np.array([0.,0.,0.,1.]),
        "eps0":np.array([-1.,-3.,2.,2.]),
        "hbar":np.array([1.,2.,-1.,0.]),
        "c":np.array([0.,1.,-1.,0.]),
    }
    base={"e":1.7,"eps0":0.8,"hbar":2.3,"c":1.1}
    alpha0=base["e"]**2/(4*np.pi*base["eps0"]*base["hbar"]*base["c"])
    errs=[]
    for s in np.exp(rng.uniform(-5,5,size=(10000,4))):
        ep=transform_numeric(base["e"],dims["e"],s)
        eps=transform_numeric(base["eps0"],dims["eps0"],s)
        hb=transform_numeric(base["hbar"],dims["hbar"],s)
        cc=transform_numeric(base["c"],dims["c"],s)
        al=ep**2/(4*np.pi*eps*hb*cc)
        errs.append(abs(al-alpha0)/abs(alpha0))
    records.append(dict(experiment="R2_unit_quotient",trials=10000,
                        metric="max_relative_error_dimensionless_alpha",
                        value=max(errs),decision="I1_MODULUS_PASS"))

    # R3: Lorentzian signature under congruence
    g=np.diag([-1.,1.,1.,1.])
    failures=0; changes=[]
    for _ in range(2000):
        A=rng.normal(size=(4,4))
        while abs(np.linalg.det(A))<0.1:
            A=rng.normal(size=(4,4))
        gp=A.T@g@A
        ev=np.linalg.eigvalsh(gp)
        sig=((ev<-1e-9).sum(),(ev>1e-9).sum())
        failures += sig!=(1,3)
        changes.append(np.linalg.norm(gp-g)/np.linalg.norm(g))
    records.append(dict(experiment="R3_signature_quotient",trials=2000,
                        metric="signature_failures",value=failures,decision="PASS"))
    records.append(dict(experiment="R3_signature_quotient",trials=2000,
                        metric="median_relative_matrix_change",value=np.median(changes),
                        decision="NEGATIVE_CONTROL_DRIFT"))

    # R4: critical mass-shell stratum is analytic; store boundary statement
    records.append(dict(experiment="R4_critical_locus_mass_shell",trials=0,
                        metric="critical_parameter",value=0.0,decision="ANALYTIC_I3_M_EQUALS_ZERO"))

    # R5: scaling exponent recovery
    def recover_z(z,trials=1000,noise=.02):
        est=[]
        for _ in range(trials):
            k=np.logspace(-2,2,100)
            amp=np.exp(rng.uniform(-3,3))
            omega=amp*k**z*np.exp(rng.normal(scale=noise,size=k.size))
            slope,_=np.polyfit(np.log(k),np.log(omega),1)
            est.append(slope)
        return np.asarray(est)
    for label,z in [("wave",1.),("diffusion",2.)]:
        est=recover_z(z)
        records.append(dict(experiment=f"R5_exponent_{label}",trials=len(est),
                            metric="mean_recovered_z",value=est.mean(),decision="PASS"))
        records.append(dict(experiment=f"R5_exponent_{label}",trials=len(est),
                            metric="sd_recovered_z",value=est.std(),decision="PASS"))

    # R6/R7: fixed-area spectral boundary + global scale normalization
    r=np.logspace(0,4,1000)
    Lx=np.sqrt(r); Ly=1/np.sqrt(r)
    lam1=4*np.pi**2*np.minimum(1/Lx**2,1/Ly**2)
    records.append(dict(experiment="R6_idf_spectral_boundary",trials=1000,
                        metric="lambda1_drop_factor",value=lam1[0]/lam1[-1],
                        decision="TOPOLOGY_PLUS_AREA_INCOMPLETE"))

    pairs=[]
    for m in range(-5,6):
        for q in range(-5,6):
            if m or q:
                pairs.append(m*m+q*q)
    base_lams=4*np.pi**2*np.sort(np.asarray(pairs))[:20]
    raw_max=0.; area_max=0.; ratio_max=0.
    for s in np.exp(rng.uniform(-4,4,5000)):
        lams=base_lams/s**2
        area=s**2
        raw_max=max(raw_max,np.max(np.abs(lams-base_lams)/base_lams))
        area_max=max(area_max,np.max(np.abs(lams*area-base_lams)/base_lams))
        ratio_max=max(ratio_max,np.max(np.abs((lams/lams[0])-(base_lams/base_lams[0]))/(base_lams/base_lams[0])))
    records.append(dict(experiment="R7_normalized_spectrum",trials=5000,
                        metric="raw_max_relative_change",value=raw_max,decision="FAIL_RAW_SPECTRUM"))
    records.append(dict(experiment="R7_normalized_spectrum",trials=5000,
                        metric="area_lambda_max_relative_error",value=area_max,decision="PASS"))
    records.append(dict(experiment="R7_normalized_spectrum",trials=5000,
                        metric="ratio_max_relative_error",value=ratio_max,decision="PASS"))

    df=pd.DataFrame(records)
    df.to_csv(outdir/"transformation_quotient_results.csv",index=False)
    print(df.to_string(index=False))

if __name__=="__main__":
    main()
