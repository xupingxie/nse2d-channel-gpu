// Part of nse2d-channel-gpu: GPU DNS of 2D channel turbulence (vorticity-streamfunction).
// Command-line parsing and initial conditions.
#include "init.cuh"
#include "io.cuh"

/* -------------------- CLI & init -------------------- */
void parse_args(int argc, char** argv, Grid& G, Params& P,
                       int& statsEvery, int& timeEvery, bool& profile,
                       double& snap_save_dt, std::string& snap_fmt_str,
                       double& diag_save_dt,
                       int& spectraEvery,
                       int& profilesWriteEvery,
                       std::string& outdir)
{
    statsEvery=50; timeEvery=50; profile=false; outdir="out";
    snap_save_dt = 0.1; snap_fmt_str = "h5";
    diag_save_dt = 0.2;
    spectraEvery = 0;
    profilesWriteEvery = 0;  // NEW: 0 => only at end
    for (int i=1;i<argc;i++){
        auto eq=[&](const char* a,const char* b){ return std::strcmp(a,b)==0; };
        if (eq(argv[i],"--Nx")&&i+1<argc) G.Nx=std::atoi(argv[++i]);
        else if (eq(argv[i],"--Ny")&&i+1<argc) G.Ny=std::atoi(argv[++i]);
        else if (eq(argv[i],"--Lx")&&i+1<argc) G.Lx=std::atof(argv[++i]);
        else if (eq(argv[i],"--h") &&i+1<argc) G.h =std::atof(argv[++i]);
        else if (eq(argv[i],"--Ub")&&i+1<argc) P.Ub=std::atof(argv[++i]);
        else if (eq(argv[i],"--Reb")&&i+1<argc) P.Reb=std::atof(argv[++i]);
        else if (eq(argv[i],"--dt")&&i+1<argc) P.dt_init=std::atof(argv[++i]);
        else if (eq(argv[i],"--dtmax")&&i+1<argc) P.dt_max=std::atof(argv[++i]);
        else if (eq(argv[i],"--cfl")&&i+1<argc) P.cfl=std::atof(argv[++i]);
        else if (eq(argv[i],"--cvisc")&&i+1<argc) P.cvisc=std::atof(argv[++i]);
        else if (eq(argv[i],"--no-adapt")) P.adapt=false;
        else if (eq(argv[i],"--tend")&&i+1<argc) P.t_end=std::atof(argv[++i]);
        else if (eq(argv[i],"--F0")&&i+1<argc) P.F0=std::atof(argv[++i]);
        else if (eq(argv[i],"--nforce")&&i+1<argc) P.nforce=std::atoi(argv[++i]);
        else if (eq(argv[i],"--drag")&&i+1<argc) P.lin_drag=std::atof(argv[++i]);
        else if (eq(argv[i],"--nu")&&i+1<argc)   P.nu_override=(Real)std::atof(argv[++i]);
        else if (eq(argv[i],"--stretch")&&i+1<argc) P.stretch = argv[++i];            // y metric
        else if (eq(argv[i],"--beta")   &&i+1<argc) P.beta    = std::atof(argv[++i]);  
        else if (eq(argv[i],"--ytable") &&i+1<argc) P.ytable  = argv[++i];
        else if (eq(argv[i],"--integrator")&&i+1<argc) P.integrator = argv[++i];  // ssprk3, imex, or imex3
        //IC options
        else if (eq(argv[i],"--init")&&i+1<argc) P.init=argv[++i];
        else if (eq(argv[i],"--amp")&&i+1<argc) P.amp=std::atof(argv[++i]);
        else if (eq(argv[i],"--alpha")&&i+1<argc) P.alpha=std::atof(argv[++i]);
        else if (eq(argv[i],"--my")&&i+1<argc) P.my=std::atoi(argv[++i]);
        else if (eq(argv[i],"--phase")&&i+1<argc) P.phase=std::atof(argv[++i]);
        else if (eq(argv[i],"--rand_amp")&&i+1<argc) P.rand_amp=std::atof(argv[++i]);
        else if (eq(argv[i],"--seed")&&i+1<argc) { P.seed = (unsigned long long)std::stoull(argv[++i]); }
        else if (eq(argv[i],"--load_path")&&i+1<argc) P.load_path=argv[++i];
        else if (eq(argv[i],"--load_use_psi")) P.load_use_psi=true;
        // 'rand' (turbulent) IC knobs
        else if (eq(argv[i],"--rand_nx")&&i+1<argc) P.rand_nx=std::atoi(argv[++i]);
        else if (eq(argv[i],"--rand_mymax")&&i+1<argc) P.rand_mymax=std::atoi(argv[++i]);
        else if (eq(argv[i],"--rand_k0")&&i+1<argc) P.rand_k0=std::atof(argv[++i]);
        else if (eq(argv[i],"--rand_sigma")&&i+1<argc) P.rand_sigma=std::atof(argv[++i]);
        else if (eq(argv[i],"--rand_amp_abs")&&i+1<argc) P.rand_amp_abs=std::atof(argv[++i]);
        // MMS (manufactured solution) knobs
	else if (eq(argv[i],"--mms")) {
		// allow both forms: "--mms" (enables) and "--mms 0/1"
		P.mms = true;
		if (i+1 < argc && argv[i+1][0] != '-') {
			P.mms = (std::atoi(argv[++i]) != 0);
		}
	}
	else if (eq(argv[i],"--mmsA0")  && i+1<argc) P.mms_A0  = (Real)std::atof(argv[++i]);
	else if (eq(argv[i],"--mmsEps") && i+1<argc) P.mms_eps = (Real)std::atof(argv[++i]);
	else if (eq(argv[i],"--mmsOm")  && i+1<argc) P.mms_Om  = (Real)std::atof(argv[++i]);
	else if (eq(argv[i],"--mmsKx")  && i+1<argc) P.mms_kx  = std::atoi(argv[++i]);

        else if (eq(argv[i],"--statsEvery")&&i+1<argc) statsEvery=std::atoi(argv[++i]);
        else if (eq(argv[i],"--timeEvery")&&i+1<argc) timeEvery=std::atoi(argv[++i]);
        else if (eq(argv[i],"--profile")) profile=true;
        
        else if (eq(argv[i],"--resume")) P.resume = true;     // NEW: resume switches requires --load_path
        // NEW: write rolling time-averaged profiles every N steps
        else if (eq(argv[i],"--profilesWriteEvery")&&i+1<argc) profilesWriteEvery=std::atoi(argv[++i]);

        else if (eq(argv[i],"--snap_save")&&i+1<argc) snap_save_dt=std::atof(argv[++i]);
        else if (eq(argv[i],"--snap_fmt")&&i+1<argc) snap_fmt_str=argv[++i];
	else if (eq(argv[i],"--snap_fields")&&i+1<argc) P.snap_fields=argv[++i];   // NEW
        else if (eq(argv[i],"--diag_save")&&i+1<argc) diag_save_dt=std::atof(argv[++i]);
        else if (eq(argv[i],"--diagsave")&&i+1<argc) diag_save_dt=std::atof(argv[++i]); // legacy alias

        else if (eq(argv[i],"--spectraEvery")&&i+1<argc) spectraEvery=std::atoi(argv[++i]);

        else if (eq(argv[i],"--outdir")&&i+1<argc) outdir=argv[++i];
    }
    // Back-compat: if user passed only --rand_amp, interpret as init=rand
    if (P.init=="laminar" && P.rand_amp>0.0 && P.amp<=0.0){
        P.init="rand"; P.amp=P.rand_amp;
    }
    // Convenience: if user selects init=mms, implicitly enable MMS forcing
    if (P.init=="mms") { P.mms = true; }
}

/* -------------------- Initial conditions (laminar | rand | ts | mix) -------------------- */
void init_omega_with_IC(const Grid& G, const Params& P, std::vector<Real>& h_w){
    const int Nx=G.Nx, Ny=G.Ny;
    const Real Lx=G.Lx, h=G.h, dy=G.dy();
    const Real Ub=P.Ub;
    h_w.assign((size_t)Nx*Ny, 0.0);

    // local logger: tee to stdout + progress.log if path is known
    auto log_init = [&](const std::string& line){
        if (!P.progress_path.empty()){
            tee_progress(P.progress_path, "%s\n", line.c_str());
        } else {
            std::cout << line << std::endl;
        }
    };

    // y-grid (use non-uniform table if present)
    std::vector<Real> y(Ny);
    if (!G.y.empty()) { y = G.y; }
    else { for (int j=0;j<Ny;++j) y[j] = -h + j*dy; }
    if (P.init == "mms") {
	    const Real k  = (Real)(2.0*M_PI) * (Real)P.mms_kx / Lx;
	    const Real c  = (Real)M_PI / ((Real)2.0*h);
	    const Real A0 = P.mms_A0;   // at t=0, sin(Om*0)=0 => A=A0

	    for (int j=0; j<Ny; ++j){
		    Real yj = y[j];
		    Real theta = c*(yj + h);
		    Real s = std::sin(theta); s *= s;
		    Real cos2 = std::cos((Real)2.0*theta);
		    Real g = (k*k)*s - (Real)2.0*(c*c)*cos2;

		    for (int i=0; i<Nx; ++i){
			    Real x = (Real)i * (Lx/(Real)Nx);
			    h_w[(size_t)j*Nx + i] = A0 * std::sin(k*x) * g;
		    }
	    }
	    log_init("Init mode: mms manufactured solution");
	    return; // IMPORTANT: do NOT add Poiseuille base on top
    }
    // =====================================================
    // Analytic diffusion eigenmode tests
    //
    //  --init diffusion   : no-slip compatible shear eigenmode (kx = 0)
    //      u(y,0) = Au sin(mu (y+h)),  v = 0
    //      omega(y,0) = -du/dy = -Au*mu cos(mu (y+h))
    //      J(psi,omega)=0 exactly (x-independent), and omega_t = nu omega_yy
    //      => omega(t) = omega(0) * exp(-nu*mu^2*t)
    //
    //  --init diffusion2d : legacy 2D Helmholtz mode with omega = (alpha^2+beta^2) psi,
    //      so J=0, but it enforces only psi=0 at walls (no-penetration) and DOES NOT
    //      satisfy u=psi_y=0 (no-slip). Useful for interior diffusion checks only.
    // =====================================================
    if (P.init == "diffusion") {
	    const int my0 = (P.my > 0 ? P.my : 1);
	    const int n   = 2*my0;                       // even -> zero-mean perturbation (compatible with Ub=0)
	    const Real mu = (Real)n * M_PI / (2.0 * h);  // = my0*pi/h
	    const Real Au = (P.amp != 0.0) ? P.amp : (Real)1.0; // interpret 'amp' as velocity amplitude

	    for (int j = 0; j < Ny; ++j) {
		    const Real omega_j = -Au * mu * std::cos(mu * (y[j] + h)); // omega = -u_y
		    for (int i = 0; i < Nx; ++i) {
			    h_w[(size_t)j*Nx + i] = omega_j;
		    }
	    }

	    const Real nu = P.nu(h);
	    std::ostringstream oss;
	    oss << "Init mode: diffusion (no-slip eigenmode, kx=0), "
		    << "mu=" << mu << " (n=" << n << "), Au=" << Au
		    << ", decay sigma=nu*mu^2=" << (nu*mu*mu);
	    log_init(oss.str());
	    return;
    }

    if (P.init == "diffusion2d") {
	    // streamwise & wall-normal mode indices
	    const int mx = 1;
	    const int my = (P.my > 0 ? P.my : 1);

	    // wavenumbers
	    const Real alpha  = (P.alpha > 0.0)
		    ? P.alpha
		    : (2.0 * M_PI * (Real)mx / Lx);
	    const Real beta   = (Real)my * M_PI / (2.0 * h);
	    const Real lambda = alpha*alpha + beta*beta;

	    // amplitude of ψ; if amp==0, use 1.0
	    const Real Apsi = (P.amp != 0.0) ? P.amp : (Real)1.0;

	    for (int j = 0; j < Ny; ++j) {
		    const Real sj = std::sin(beta * (y[j] + h));
		    for (int i = 0; i < Nx; ++i) {
			    const Real x = (Real)i * (Lx / (Real)Nx);
			    const Real psi0   = Apsi * std::sin(alpha * x) * sj;
			    const Real omega0 = lambda * psi0;       // ω = λ ψ  => J=0
			    h_w[(size_t)j*Nx + i] = omega0;
		    }
	    }

	    std::ostringstream oss;
	    oss << "Init mode: diffusion2d (legacy Helmholtz eigenmode free-slip), "
		    << "alpha=" << alpha << ", beta=" << beta
		    << ", lambda=" << lambda << ", Apsi=" << Apsi;
	    log_init(oss.str());
	    return;
    }

    // simple LCG for fast portable noise (same structure as your snippet)
    unsigned long long seed = P.seed;
    auto rnd01 = [&](){
        seed = 6364136223846793005ULL*seed + 1ULL;
        // Map upper 53 bits to (0,1]; same constant as your snippet
        return ((seed>>11) * 1.1102230246251565e-16);
    };
    // ---New: Load from a snapshot: prefer 'omega'; or, with --load_use_psi, read 'psi' and rebuild omega. For Freedecay test
#ifndef NO_HDF5
    if (P.init=="load"){
        if (P.load_path.empty()) throw std::runtime_error("--init load needs --load_path /path/to/snap_tXXXXXX.h5");
        hid_t file = H5Fopen(P.load_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file<0) throw std::runtime_error("Cannot open HDF5 file: "+P.load_path);
        // Metadata checks
        int Nx_f = h5_read_attr_int(file, "Nx", -1);
        int Ny_f = h5_read_attr_int(file, "Ny", -1);
        double Lx_f = h5_read_attr_double(file, "Lx", -1.0);
        double h_f  = h5_read_attr_double(file, "h",  -1.0);
        if (Nx_f!=Nx || Ny_f!=Ny) {
            H5Fclose(file);
            throw std::runtime_error("Snapshot grid does not match (Nx,Ny).");
        }
        if (std::abs(Lx_f - Lx) > 1e-12 || std::abs(h_f - h) > 1e-12){
            H5Fclose(file);
            throw std::runtime_error("Snapshot geometry (Lx,h) does not match current run.");
        }
        bool has_w = h5_has_dataset(file,"omega");
        bool has_p = h5_has_dataset(file,"psi");
        if (!has_w && !has_p){
            H5Fclose(file);
            throw std::runtime_error("Snapshot has neither 'omega' nor 'psi'.");
        }
        if (has_w && !P.load_use_psi){
            // Preferred path: read omega directly
            h5_read_2d(file, "omega", h_w.data(), Nx, Ny);
        } else {
            // Read psi, then build omega = -(δxx ψ + δyy ψ), walls via Thom with current Ub
            std::vector<Real> h_psi((size_t)Nx*Ny);
            h5_read_2d(file, "psi", h_psi.data(), Nx, Ny);
            const Real dx = G.dx();
            const Real inv_dx2 = 1.0/(dx*dx), inv_dy2 = 1.0/(dy*dy);
            auto IX = [Nx](int i){ int ii=i; if (ii<0) ii+=Nx; if (ii>=Nx) ii-=Nx; return ii; };
            auto id  = [Nx](int i,int j){ return (size_t)j*Nx + i; };
            // interior
            for (int j=1;j<Ny-1;++j){
                for (int i=0;i<Nx;++i){
                    Real psi_c = h_psi[id(i,j)];
                    Real psi_xx = (h_psi[id(IX(i+1),j)] - 2*psi_c + h_psi[id(IX(i-1),j)])*inv_dx2;
                    Real psi_yy = (h_psi[id(i,j+1)]    - 2*psi_c + h_psi[id(i,j-1)])*inv_dy2;
                    h_w[id(i,j)] = -(psi_xx + psi_yy);
                }
            }
            // walls (Thom closure, consistent with current Ub and ψ-Dirichlet)
            const Real psi_bot = 0.0;
            const Real psi_top = 2.0*h*Ub;
            for (int i=0;i<Nx;++i){
                h_w[id(i,0)]     = - 2.0* (h_psi[id(i,1)]     - psi_bot) * inv_dy2;
                h_w[id(i,Ny-1)]  = - 2.0* (h_psi[id(i,Ny-2)]  - psi_top) * inv_dy2;
            }
        }
        // log what we did (pull some metadata for info)
        //double Ub_f = h5_read_attr_double(file, "Ub", NAN);
        //double Reb_f= h5_read_attr_double(file, "Reb", NAN);
        double nu_f = h5_read_attr_double(file, "nu", NAN);
        H5Fclose(file);
        {
            std::ostringstream oss;
            oss << "Loaded IC from " << P.load_path
                << (has_w && !P.load_use_psi ? " (omega)" : " (psi->omega)") << ", nu=" << nu_f;
                //<< "; file Ub=" << Ub_f << ", Reb=" << Reb_f << ", nu=" << nu_f;
            log_init(oss.str());
        }
        return;
    }
#else
    if (P.init=="load") throw std::runtime_error("built with NO_HDF5: --init load is unavailable");
#endif

    // laminar Poiseuille base: U(y)=1.5*Ub*(1-(y/h)^2), omega=-dU/dy = 3 Ub * y / h^2
    for (int j=0;j<Ny;++j){
        Real wj = 3.0*Ub * y[j] / (h*h);
        for (int i=0;i<Nx;++i) h_w[(size_t)j*Nx + i] = wj;
    }
    // Random ω-noise
    if (P.init=="mix"){
        Real scale = std::max((Real)1e-12, P.amp) * (Ub/h);
        size_t N = (size_t)Nx*Ny;
        for (size_t idx=0; idx<N; ++idx){
            h_w[idx] += scale * (rnd01() - 0.5);
        }
    }

    // TS mode: ψ'(x,y) = Aψ sin(μ(y+h)) cos(α x + φ), ω'=(α^2+μ^2)ψ'
    if (P.init=="ts" || P.init=="mix"){
        Real alpha = (P.alpha>0 ? P.alpha : (2.0*M_PI/Lx));
        Real mu    = M_PI * (Real)P.my / (2.0*h);
        Real Apsi  = (P.amp*Ub) / std::max((Real)1e-12, mu); // target |u'|~Apsi*mu ≈ amp*Ub
        for (int j=0;j<Ny;++j){
            Real sy = std::sin(mu*(y[j]+h));
            for (int i=0;i<Nx;++i){
                Real x = (Real)i * (Lx/(Real)Nx);
                Real psi_p = Apsi * sy * std::cos(alpha*x + P.phase);
                Real w_p   = (alpha*alpha + mu*mu) * psi_p; // since ω = -∇²ψ
                h_w[(size_t)j*Nx + i] += w_p;
            }
        }
    }
    // --- New: 'rand' = turbulent vorticity IC (band-limited, independent of Ub) ---
    if (P.init=="rand"){
        const Real dx = G.dx();
        const Real alpha_min = 2.0*M_PI/Lx;
        const Real alpha_nyq = M_PI/dx;
        const Real alpha_max = 0.35*alpha_nyq;      // margin from Nyquist
        const int  nxm  = std::max(1, P.rand_nx);
        const int  mymx = std::max(1, P.rand_mymax);
        const Real k0   = (P.rand_k0>0 ? P.rand_k0 : (6.0 * 2.0*M_PI/Lx));
        const Real sig  = std::max((Real)0.05, P.rand_sigma);

        std::mt19937_64 rng(P.seed);
        std::uniform_real_distribution<Real> U01(0.0,1.0);
        std::normal_distribution<Real> N0(0.0,1.0);

        auto sample_alpha = [&]()->Real{
            Real a = std::exp(std::log(k0) + sig * N0(rng));
            if (a < alpha_min) a = alpha_min;
            if (a > alpha_max) a = alpha_max;
            return a;
        };

        // Treat 'amp' (or rand_amp_abs) as an ABSOLUTE vorticity scale. Works for Ub=0.
        const Real A0 = std::max((Real)1e-12, (P.rand_amp_abs>0? P.rand_amp_abs : P.amp))
                        / std::sqrt((Real)nxm);

        // Precompute y-basis sin(mu(y+h)) for 1..mymx (Dirichlet-ψ compliant)
        std::vector<std::vector<Real>> sY(mymx+1, std::vector<Real>(Ny,0.0));
        for (int my=1; my<=mymx; ++my){
            Real mu = M_PI*(Real)my/(2.0*h);
            for (int j=0;j<Ny;++j) sY[my][j] = std::sin(mu*(y[j]+h));
        }

        // Add modes
        for (int m=0; m<nxm; ++m){
            Real alpha = sample_alpha();
            int  my    = 1 + (int)std::floor(U01(rng)*mymx);
            Real phi   = 2.0*M_PI*U01(rng);
            Real mu    = M_PI*(Real)my/(2.0*h);
            Real k     = std::sqrt(alpha*alpha + mu*mu);
            Real wamp  = A0 * std::exp(-0.5*std::pow(std::log(k/k0)/sig,2));
            for (int j=0;j<Ny;++j){
                Real sj = sY[my][j];
                for (int i=0;i<Nx;++i){
                    Real x = ( (Real)i * (Lx/(Real)Nx) );
                    h_w[(size_t)j*Nx + i] += wamp * std::cos(alpha*x + phi) * sj;
                }
            }
        }
    }
    {
        std::ostringstream oss;
        oss << "Init mode: " << P.init
            << " (amp=" << P.amp
            << ", alpha=" << (P.alpha>0?P.alpha:2.0*M_PI/G.Lx)
            << ", my=" << P.my
            << ", phase=" << P.phase
            << ", seed=" << P.seed;
        if (P.init=="load"){
            oss << ", path=" << P.load_path
                << (P.load_use_psi? ", using psi" : ", using omega");
        }
        oss << ")";
        log_init(oss.str());
    }
}

