// Part of nse2d-channel-gpu: GPU DNS of 2D channel turbulence (vorticity-streamfunction).
// Time loop: SSPRK(3,3) with Poisson solve and wall closure per stage; outputs.
#include "common.cuh"
#include "io.cuh"
#include "grid.cuh"
#include "ops.cuh"
#include "diag.cuh"
#include "poisson.cuh"
#include "init.cuh"

/* -------------------- Main -------------------- */
#ifndef NSE2D_BUILD_AS_LIB
int main(int argc, char** argv){
    Grid G; Params P;
    int statsEvery, timeEvery; bool profile; std::string outdir;
    double snap_save_dt, diag_save_dt; int spectraEvery;
    std::string snap_fmt_str;
    int profilesWriteEvery; // NEW

    parse_args(argc, argv, G, P,
               statsEvery, timeEvery, profile,
               snap_save_dt, snap_fmt_str,
               diag_save_dt,
               spectraEvery, profilesWriteEvery,
               outdir);
    const Real nu = P.nu(G.h);

    // at startup
    if (!ensure_dir_p(outdir)) {
        fprintf(stderr,"ERROR: cannot create outdir %s\n", outdir.c_str());
        return 1;
    }
    //ensure_dir(outdir);
    std::string diag_path = outdir + "/diagnostics.csv";
    std::string prof_path = outdir + "/profiles_timeavg.csv";
    std::string time_path = outdir + "/timing.csv";
    std::string prog_path = outdir + "/progress.log";
    P.progress_path = prog_path; // tell the IC routine where to tee its logs

    // Convenience macro after prog_path is in scope
    #define LOGP(...) tee_progress(prog_path, __VA_ARGS__)
 
    // Build mapping on host and adopt into G
    Grid Gmap;
    build_y_metrics(G, P, Gmap);
    G = Gmap;
    
    const bool mapped = !G.y.empty();
    const Real dx = G.dx(), dy = G.dy();
    const Real inv_dx = 1.0/dx, inv_dy = 1.0/dy, inv_dx2 = inv_dx*inv_dx, inv_dy2 = inv_dy*inv_dy;
    const Real deta = (G.Ny>1) ? 2.0/(Real)(G.Ny-1) : 1.0;
    const Real inv_deta = 1.0/deta;
    const Real inv_deta2= inv_deta*inv_deta;
    size_t N = (size_t)G.Nx*G.Ny;
    // compute dt-related
    Real dy_min = dy;
    if (!G.dy_edge.empty()) {
	    dy_min = std::fabs(G.dy_edge[0]);
	    for (int j = 1; j < G.Ny-1; ++j)
		    dy_min = std::min(dy_min, std::fabs(G.dy_edge[j]));
    }
    const Real inv_dymin2 = 1.0 / (dy_min * dy_min);

    // Effective "heights" from the quadrature weights
    Real Ly_full = 0.0;              // uses j = 0..Ny-1
    Real Ly_int  = 0.0;              // interior, j = 1..Ny-2
    for (int j=0; j<G.Ny;   ++j) Ly_full += G.w_node[j];
    for (int j=1; j<G.Ny-1; ++j) Ly_int  += G.w_node[j];
    // In practice Ly_full ≈ 2*h for both uniform and tanh grids
    const Real invA_full = 1.0 / ((Real)G.Nx * Ly_full);
    const Real invA_int  = 1.0 / ((Real)G.Nx * Ly_int);
    // ------------------------------------------------------------------
    // Precompute <F_x>_A (area-mean streamwise body force) for consistent dpdx.
    // Forcing in solver: F_x(y) = F0 * sin(n*pi*(y+h)/(2h))
    // Since F_x is x-independent: <F_x>_A = (1/Ly) * sum_j w_node[j] * F_x(y_j)
    // where Ly = sum_j w_node[j] (≈ 2h).
    // ------------------------------------------------------------------
    Real meanFx_A = 0.0;
    if (P.F0 != 0.0) {
        for (int j = 0; j < G.Ny; ++j) {
            const Real y   = G.y[j];
            const Real arg = (Real)P.nforce * (Real)M_PI * ((y + G.h) / (2.0 * G.h));
            const Real Fx  = P.F0 * std::sin(arg);
            meanFx_A += Fx * G.w_node[j];
        }
        meanFx_A /= Ly_full;
    }
    // Allocate device copies of metrics if mapped
    Real *d_a_node=nullptr, *d_a_edge=nullptr, *d_w_node=nullptr, *d_y=nullptr, *d_y_node=nullptr, *d_dy_edge=nullptr;
    Real *d_asub=nullptr, *d_csup=nullptr, *d_b0=nullptr;  // tri-diagonal coeffs (Ny-2)

    if (mapped) {
	    CUDA_CHECK(cudaMalloc(&d_a_node, G.Ny    *sizeof(Real)));
	    CUDA_CHECK(cudaMalloc(&d_a_edge, (G.Ny-1)*sizeof(Real)));
	    CUDA_CHECK(cudaMalloc(&d_w_node, G.Ny    *sizeof(Real)));
	    CUDA_CHECK(cudaMalloc(&d_y_node,  G.Ny    *sizeof(Real)));
	    CUDA_CHECK(cudaMalloc(&d_dy_edge,(G.Ny-1) *sizeof(Real)));
	    CUDA_CHECK(cudaMemcpy(d_a_node,  G.a_node.data(),  G.Ny   *sizeof(Real), cudaMemcpyHostToDevice));
	    CUDA_CHECK(cudaMemcpy(d_a_edge,  G.a_edge.data(), (G.Ny-1)*sizeof(Real), cudaMemcpyHostToDevice));
	    CUDA_CHECK(cudaMemcpy(d_w_node,  G.w_node.data(),  G.Ny   *sizeof(Real), cudaMemcpyHostToDevice));
	    CUDA_CHECK(cudaMemcpy(d_y_node,  G.y.data(),       G.Ny   *sizeof(Real), cudaMemcpyHostToDevice));
	    CUDA_CHECK(cudaMemcpy(d_dy_edge, G.dy_edge.data(),(G.Ny-1)*sizeof(Real), cudaMemcpyHostToDevice));

	    // variable-coefficient Poisson coefficients (Ny-2)
	    std::vector<Real> asub(G.Ny-2), csup(G.Ny-2), b0(G.Ny-2);
	    for (int j=1; j<=G.Ny-2; ++j) {
		    const Real aj = G.a_node[j];
		    const Real aL = G.a_edge[j-1];
		    const Real aR = G.a_edge[j];
		    asub[j-1] =  aj * aL / (deta*deta);
		    csup[j-1] =  aj * aR / (deta*deta);
		    b0  [j-1] = -(aj * (aL + aR) / (deta*deta));
	    }
	    CUDA_CHECK(cudaMalloc(&d_asub, (G.Ny-2)*sizeof(Real)));
	    CUDA_CHECK(cudaMalloc(&d_csup, (G.Ny-2)*sizeof(Real)));
	    CUDA_CHECK(cudaMalloc(&d_b0,   (G.Ny-2)*sizeof(Real)));
	    CUDA_CHECK(cudaMemcpy(d_asub,asub.data(),(G.Ny-2)*sizeof(Real),cudaMemcpyHostToDevice));
	    CUDA_CHECK(cudaMemcpy(d_csup,csup.data(),(G.Ny-2)*sizeof(Real),cudaMemcpyHostToDevice));
	    CUDA_CHECK(cudaMemcpy(d_b0,  b0.data(),  (G.Ny-2)*sizeof(Real),cudaMemcpyHostToDevice));

	    LOGP("[grid-y] stretch=%s beta=%.3f (mapped)\n", P.stretch.c_str(), (double)P.beta);
    } else {
	    LOGP("[grid-y] stretch=none (uniform) ERROR! \n");
    }

    SnapFmt snap_fmt = SNAP_H5;
    if      (snap_fmt_str=="h5")   snap_fmt = SNAP_H5;

    // --- determine start time if resuming (read attribute 't' from snapshot) ---
    double t_start = 0.0;
#ifndef NO_HDF5
    if (P.resume){
        if (P.load_path.empty()){
            fprintf(stderr,"ERROR: --resume requires --load_path /path/to/snap.h5\n");
            return 1;
        }
        hid_t file = H5Fopen(P.load_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0){
            fprintf(stderr,"ERROR: cannot open snapshot %s\n", P.load_path.c_str());
            return 1;
        }
        t_start = h5_read_attr_double(file, "t", 0.0);
        H5Fclose(file);
    }
#else
    if (P.resume){ fprintf(stderr,"ERROR: built with NO_HDF5; cannot --resume\n"); return 1; }
#endif

    LOGP("Z2Z channel DNS | Nx=%d Ny=%d Lx=%.6f h=%.6f\n",G.Nx,G.Ny,G.Lx,G.h);
    LOGP("Reb=%.6g, Ub=%.6g, nu=%.6g%s\n", (double)P.Reb, (double)P.Ub, (double)P.nu(G.h),
        (P.nu_override > 0 ? " (override)" : " (from 2h*Ub/Reb)"));

    size_t total_snaps = scheduled_count(P.t_end, snap_save_dt);
    size_t total_diags = scheduled_count(P.t_end, diag_save_dt);
    EventProgress prog_snap("snap", total_snaps);
    EventProgress prog_diag("diag", total_diags);
    StepProgress  prog_spec("spec");

    // write headers (append-safe on resume)
    if (!file_nonempty(diag_path)){
        std::ofstream f(diag_path, std::ios::out);
        f << "step,t,dt,umax,vmax,K,Omega,dOmega_rhs,eta,BOmega_wall,Pin,epsilon,POmega,tauw_bot,tauw_top,dpdx,utau_avg,Cf,Re_tau\n";
    }
    if (profile && !file_nonempty(time_path)){
        std::ofstream g(time_path, std::ios::out);
        g << "step,t,dt,t_step_ms,t_fft_ms,t_tridiag_ms\n";
    }
    // create the profiles file header now; we will refresh/overwrite it later
    { std::ofstream f(prof_path, std::ios::out);
      f << "j,y,yplus,U,V,uu,vv,uv,Uplus,tau_total\n"; }

    if (P.resume)
        LOGP("[mode] Resume simulation from %s, t0=%.6f\n", P.load_path.c_str(), t_start);
    else
        LOGP("[mode] New simulation (fresh IC), t0=0.000000\n");
    LOGP("[params] Nx=%d Ny=%d Lx=%.6f h=%.6f Ub=%.6f Reb=%.1f nu=%.6e "
           "dt_init=%.6g cfl=%.3g adapt=%s t_end=%.6g\n",
           G.Nx, G.Ny, G.Lx, G.h, P.Ub, P.Reb, P.nu(G.h),
           (double)P.dt_init, (double)P.cfl, (P.adapt? "yes":"no"), (double)P.t_end);
    if (snap_save_dt > 0)       LOGP("[io] snapshots every Δt=%.6g\n", snap_save_dt);
    if (diag_save_dt  > 0)      LOGP("[io] diagnostics every Δt=%.6g\n", diag_save_dt);
    if (profilesWriteEvery > 0) LOGP("[io] rolling profiles every %d steps -> profiles_timeavg.csv\n", profilesWriteEvery);
    if (profile)                LOGP("[io] timing (performance profile) every %d steps -> timing.csv\n", timeEvery); 

    Real *d_w,*d_w1,*d_w2,*d_psi,*d_rhs,*d_J,*d_lap;
    CUDA_CHECK(cudaMalloc(&d_w,   N*sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_w1,  N*sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_w2,  N*sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_psi, N*sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_rhs, N*sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_J,   N*sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_lap, N*sizeof(Real)));

    //===========Initial conditions=========
    std::vector<Real> h_w;
    init_omega_with_IC(G, P, h_w);
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), N*sizeof(Real), cudaMemcpyHostToDevice));

    cudaStream_t stream = 0;
    Poisson2D_Z2Z poisson(G, stream);
    poisson.set_profile(profile);
    poisson.enable_mapped(d_a_node, d_a_edge, inv_deta, stream);

    // ============= IMEX-specific setup =============
    LOGP("[integrator] SSPRK3 (fully explicit)\n");
    // ============= END IMEX setup =============

    // initial solve, (from ω; ψ is recovered, so resume-from-ω works)
    Real psi_bot=0.0, psi_top=2.0*G.h*P.Ub;
    poisson.solve(d_w, d_psi, dy, psi_bot, psi_top, stream);
    if (mapped) launch_thom_wall_vorticity_mapped(d_w, d_psi, G.Nx, G.Ny, d_dy_edge, psi_bot, psi_top, stream);

    CUDA_CHECK(cudaDeviceSynchronize());

    // buffers for moments and scalars
    Real *d_sum_u,*d_sum_v,*d_sum_u2,*d_sum_v2,*d_sum_uv;
    CUDA_CHECK(cudaMalloc(&d_sum_u,  G.Ny*sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_sum_v,  G.Ny*sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_sum_u2, G.Ny*sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_sum_v2, G.Ny*sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_sum_uv, G.Ny*sizeof(Real)));
    Real *d_sum_w2,*d_pin,*d_sumK,*d_mean_ome_bot,*d_mean_ome_top;
    Real *d_sum_gradw2; // sum over full domain of |∇ω|^2, palinstrophy
    Real *d_mean_wwy_bot, *d_mean_wwy_top;  // wall-flux boundary integrand
    CUDA_CHECK(cudaMalloc(&d_sum_w2, sizeof(Real)));
    //CUDA_CHECK(cudaMalloc(&d_sum_gw2, sizeof(Real))); // this one is Delta\omega
    CUDA_CHECK(cudaMalloc(&d_sum_gradw2, sizeof(Real)));
    // wall flux integrand means ⟨ω ω_y⟩_x at y=±h
    CUDA_CHECK(cudaMalloc(&d_mean_wwy_bot, sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_mean_wwy_top, sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_pin, sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_sumK, sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_mean_ome_bot, sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_mean_ome_top, sizeof(Real)));
    Real *d_sum_eps, *d_p_omega;   // Add to the declaration line, epsilon = real energy dissipation rate
    CUDA_CHECK(cudaMalloc(&d_sum_eps, sizeof(Real)));  // Add allocation
    CUDA_CHECK(cudaMalloc(&d_p_omega, sizeof(Real)));   //Powerinput for Omega(Enstrophy)

    //for new d_sum_omega_rhs
    Real *d_sum_omega_rhs;
    CUDA_CHECK(cudaMalloc(&d_sum_omega_rhs, sizeof(Real)));

    // velocity buffers
    Real *d_u=nullptr, *d_v=nullptr;
    CUDA_CHECK(cudaMalloc(&d_u, N*sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_v, N*sizeof(Real)));

    // spectra machinery (1D in x)   do we need
    Spectra1D spec1d;
    bool do_spec = (spectraEvery > 0);
    if (do_spec) spec1d.init(G);
    // time-weighted utau accumulation (from diag cadence)
    double tauw_abs_dt_sum = 0.0, T_diag = 0.0, last_diag_t = t_start; // NEW
    Real *d_umax,*d_vmax, *d_vmaxm; 
    CUDA_CHECK(cudaMalloc(&d_umax,sizeof(Real))); 
    CUDA_CHECK(cudaMalloc(&d_vmax,sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_vmaxm,sizeof(Real)));
    
    cudaEvent_t e_step0, e_step1; if (profile){ cudaEventCreate(&e_step0); cudaEventCreate(&e_step1); }
    // schedules
    //const double epsT = 1e-14;
    // Relative time tolerance: ~64 ULPs at magnitude t
    auto time_tol = [](double t){
        using std::abs; using std::max;
        return 64.0 * std::numeric_limits<double>::epsilon() * max(1.0, abs(t));
    };
    bool use_snap = (snap_save_dt > 0);
    bool use_diag = (diag_save_dt > 0);
    double next_snap_t = use_snap ? (P.resume ? next_after(t_start, snap_save_dt) : 0.0) : 1e300;
    double next_diag_t = use_diag ? (P.resume ? next_after(t_start, diag_save_dt)  : 0.0) : 1e300;

    // prepare u,v and set up scheduled events at initial, use mapped
    launch_max_uv_from_psi_mapped(d_psi,G.Nx,G.Ny,inv_dx,inv_deta,d_a_node,d_umax,d_vmax,d_vmaxm,stream);
    CUDA_CHECK(cudaDeviceSynchronize());

    // t=0 outputs only for NEW runs, resume does not need initial write
    if (!P.resume && use_snap){
        save_snapshot_any(outdir, G.Nx, G.Ny, 0.0, d_psi,d_w,d_u,d_v,snap_fmt,G,P);
        next_snap_t = snap_save_dt;
    }
    if (!P.resume && use_diag){
        // t=0 diagnostics
        Real umax0,vmax0; 
        launch_max_uv_from_psi_mapped(d_psi, G.Nx,G.Ny,inv_dx,inv_deta,d_a_node,d_umax,d_vmax,d_vmaxm,stream);
        CUDA_CHECK(cudaMemcpy(&umax0,d_umax,sizeof(Real),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&vmax0,d_vmax,sizeof(Real),cudaMemcpyDeviceToHost));
	launch_kinetic_energy_mapped(d_psi,G.Nx,G.Ny,inv_dx,inv_deta,d_a_node,d_w_node,d_sumK,stream);
	// full-domain Ω and η (include wall rows)
	launch_enstrophy_diss_mapped_full(d_w,G.Nx,G.Ny,inv_dx,inv_deta,d_a_node,d_w_node,d_sum_w2,d_sum_gradw2,stream);
	// wall enstrophy flux integrand ⟨ω ω_y⟩_x at y=±h
	launch_mean_wall_omega_omegay_mapped(d_w,G.Nx,G.Ny,inv_deta,d_a_node,d_mean_wwy_bot,d_mean_wwy_top,stream);

	//launch_enstrophy_diss_mapped(d_w,G.Nx,G.Ny,inv_dx,inv_deta,d_a_node,d_w_node,d_sum_w2,d_sum_gradw2,stream); // regular \nabla w^2
	// New: wall‑balanced η = ν⟨ω Δω⟩ (works for no‑slip & free‑slip)
        //launch_enstrophy_diss_balance_mapped(d_w,G.Nx,G.Ny,inv_dx2,inv_deta2,d_a_node,d_a_edge,d_w_node,d_sum_w2,d_sum_gw2,stream); 
        launch_power_input_mapped(d_psi,G.Nx,G.Ny,inv_deta,G.h,P.F0,P.nforce,d_a_node,d_y_node,d_w_node,d_pin,stream);
        launch_energy_dissipation_mapped(d_psi,d_w,G.Nx,G.Ny,inv_dx,inv_deta,d_a_node,d_a_edge,d_w_node,d_sum_eps,stream);
    	launch_enstrophy_input_mapped_full(d_w,G.Nx,G.Ny,G.h,P.F0,P.nforce,d_y_node,d_w_node,d_p_omega,stream);
 
	//===========new t=0 to test the dOmega_rhs
	launch_arakawa_J_mapped(d_psi, d_w, d_J,
			G.Nx, G.Ny,
			inv_dx, inv_deta,
			d_a_node, stream);

	launch_laplacian_fd_mapped(d_w, d_lap,
			G.Nx, G.Ny,
			inv_dx2, inv_deta2,
			d_a_node, d_a_edge, stream);

	launch_rhs_mapped(d_J, d_lap, d_w, d_rhs,
			G.Nx, G.Ny,
			nu, P.F0, P.nforce, G.h, P.lin_drag,
			d_y_node, stream);
	if (P.mms) {
		// at t=t_start
		launch_mms_add_forcing(d_rhs, G.Nx, G.Ny, G.Lx, G.h, t_start, nu, P.lin_drag,
				P.mms_A0, P.mms_eps, P.mms_Om, P.mms_kx,d_y_node, stream);}
	// now accumulate <ω * rhs> with same interior weights as Ω, η
	launch_omega_rhs_balance_mapped(
			d_w, d_rhs,
			G.Nx, G.Ny,
			d_w_node,
			d_sum_omega_rhs,
			stream);
	//================end of rhs
        Real sumK0,sum_w20,sum_gradw20;
        Real Pin0, sum_eps0; 
        Real p_omega_sum0;
        Real sum_omega_rhs0=0;
	Real mean_wwy_bot0, mean_wwy_top0; // Enstrophy wall flux integrand <w w_y>_x at y=-+h
        CUDA_CHECK(cudaMemcpy(&sumK0, d_sumK,   sizeof(Real),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&sum_w20,d_sum_w2,sizeof(Real),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&sum_gradw20,d_sum_gradw2,sizeof(Real),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&Pin0,d_pin,sizeof(Real),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&sum_eps0, d_sum_eps, sizeof(Real), cudaMemcpyDeviceToHost));
    	CUDA_CHECK(cudaMemcpy(&p_omega_sum0,d_p_omega,sizeof(Real),cudaMemcpyDeviceToHost));
    	CUDA_CHECK(cudaMemcpy(&sum_omega_rhs0,d_sum_omega_rhs,sizeof(Real),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&mean_wwy_bot0,d_mean_wwy_bot,sizeof(Real),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&mean_wwy_top0,d_mean_wwy_top,sizeof(Real),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaStreamSynchronize(stream)); // all diagnostics are ready
        // Global scalars
        Real sumK0_area = sumK0 * invA_full;  // K = ½⟨u²+v²⟩_A     (full domain)
        Real Omega0 = 0.5 * sum_w20 * invA_full;  // Ω = ½⟨ω²⟩_A     (full)
        // bulk vs balanced enstorphy dissipation
        //Real eta0_bal  = nu * sum_gw20 *invA_int;   // η = ν⟨ω Δω⟩_A,int
        Real eta0      = nu * sum_gradw20 * invA_full; // ν⟨|∇ω|²⟩_A
	Real BOmega0_wall = (nu>0.0) ? (nu * (mean_wwy_top0 - mean_wwy_bot0) / (2.0 * G.h)) : (Real)0; // ν/(2h) [⟨ω ω_y⟩_top - ⟨ω ω_y⟩_bot]
	//Real eta0_wall = eta0_bal + eta0;  // -ν/A ∮ ω ∂nω ds   -------------> this is old interior-only balance
        Real dOmega0 = sum_omega_rhs0 * invA_full;
        Pin0 *= invA_full;    // Pin = ⟨u F_x⟩_A      (full domain)
        Real epsilon0 = nu * sum_eps0 * invA_full;  // ε = ν⟨|∇u|²⟩_A   (full-domain energy dissipation)
    	Real P_Omega0 = p_omega_sum0 * invA_full; // P_Ω = ⟨ω f_ω⟩_A  (enstrophy input)

        //we do not need mean_wall_omega_mapped
	Real mOb,mOt; 
	launch_mean_wall_omega(d_w,G.Nx,G.Ny,d_mean_ome_bot,d_mean_ome_top,stream);
        CUDA_CHECK(cudaMemcpy(&mOb,d_mean_ome_bot,sizeof(Real),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&mOt,d_mean_ome_top,sizeof(Real),cudaMemcpyDeviceToHost));
        Real tauwb=-nu*mOb, tauwt=-nu*mOt;
        Real utau0=sqrt(0.5*(fabs(tauwb)+fabs(tauwt)));
        Real Cf0 = (P.Ub!=0.0) ? 2.0*(0.5*(fabs(tauwb)+fabs(tauwt)))/(P.Ub*P.Ub) : std::numeric_limits<Real>::quiet_NaN();
	// Re_tau undefined when nu = 0
	Real Re_tau0 = (nu > 0.0) ? utau0 * G.h / nu : 0.0;  // or NaN, depending on preference
        // Mean pressure gradient <p_x>_A from mean momentum balance (constant-flux):
        //   <p_x>_A = (tau_top - tau_bot)/(2h) + <F_x>_A - r*Ub
        const Real dpdx0 = (tauwt - tauwb) / (2.0 * G.h) + meanFx_A - P.lin_drag * P.Ub;

        std::ofstream f(diag_path,std::ios::app);
        f.setf(std::ios::scientific); f.precision(8);
        f<<0<<","<<0.0<<","<<P.dt_init<<","<<umax0<<","<<vmax0<<","<<sumK0_area<<","<<Omega0<<","<<dOmega0<<","<<eta0<<","<<BOmega0_wall<<","<<Pin0<<","<<epsilon0<<","<<P_Omega0<<","
		<<tauwb<<","<<tauwt<<","<<dpdx0<<","<<utau0<<","<<Cf0<<","<<Re_tau0<<"\n";
        log_event(prog_diag, prog_path, 0.0, (double)P.dt_init);
        next_diag_t = diag_save_dt;
    }
    // helper preview of the first future write times
    if (use_snap) LOGP("[schedule] next snapshot    at t=%.6f (Δt=%.6g)\n", next_snap_t, snap_save_dt);
    if (use_diag) LOGP("[schedule] next diagnostics at t=%.6f (Δt=%.6g)\n", next_diag_t, diag_save_dt);
    LOGP("[run] t0=%.6f -> t_end=%.6f (ΔT_remain=%.6f)\n", t_start, (double)P.t_end, (double)(P.t_end - t_start));

    // time-weighted profile accumulators
    std::vector<Real> H_int_u(G.Ny,0.0), H_int_v(G.Ny,0.0), H_int_u2(G.Ny,0.0), H_int_v2(G.Ny,0.0), H_int_uv(G.Ny,0.0);
    double T_accum = 0.0, last_stats_t = t_start; // start clocks at t0, start accum clocks at current time

    struct { Real t, dt; } S { (Real)t_start, P.dt_init };   // NEW: start at t0 if resuming
    int step=0;

    while (S.t < P.t_end - 0.5*S.dt){
        if (profile) poisson.profile_reset();
        if (profile) cudaEventRecord(e_step0, stream);

        // adaptive dt block
        if (P.adapt){
            if (mapped) launch_max_uv_from_psi_mapped(d_psi,G.Nx,G.Ny,inv_dx,inv_deta,d_a_node,d_umax,d_vmax,d_vmaxm,stream);
	    Real umax,vmax,vmaxm; 
	    CUDA_CHECK(cudaMemcpy(&umax,d_umax,sizeof(Real),cudaMemcpyDeviceToHost));
	    CUDA_CHECK(cudaMemcpy(&vmax,d_vmax,sizeof(Real),cudaMemcpyDeviceToHost));
	    CUDA_CHECK(cudaMemcpy(&vmaxm,d_vmaxm,sizeof(Real),cudaMemcpyDeviceToHost));
	   // // Advective CFL: use smallest physical dy in y
	    //Real dt_adv = P.cfl * fmin( (umax > 0 ? dx    / umax : 1e30),
	    //		    (vmax > 0 ? dy_min / vmax : 1e30) );
            //advective CFL on mapped grid:
	    //   dt_x ~ CFL * dx   / max|u|
	    //   dt_y ~ CFL * Δη   / max(|v| * a(j))   since dy(j)=Δη/a(j)
	    const Real dt_x = (umax > 0 ? P.cfl *(dx / umax) : 1e30);
	    const Real dt_y = (vmaxm >0 ? P.cfl *(deta / vmaxm) : 1e30);
	    Real dt_adv = fmin(dt_x, dt_y);
	    // Diffusive CFL: skip for IMEX schemes (diffusion is implicit)
	    Real dt_vis;
	    dt_vis = (nu > 0) ? P.cvisc * 0.5 / ( nu * (inv_dx2 + inv_dymin2) ) : 1e30;

	    S.dt = fmin(P.dt_max, fmin(dt_adv, dt_vis));
	    if (step % 5000 == 0) {
		    LOGP("[dt] step=%d t=%.6g umax=%.3e vmax=%.3e vmaxm=%.3e dt_x=%.e dt_y=%.3e dt_adv=%.3e dt_vis=%.3e dt=%.3e \n",
				    step, (double)S.t, (double)umax, (double)vmax, (double)vmaxm,(double)dt_x, (double)dt_y,
				    (double)dt_adv, (double)dt_vis, (double)S.dt);
    	    }
        }

        // clip dt to land on next scheduled outputs
        if (P.adapt && (use_snap || use_diag)){
            double t_next = 1e300;
            if (use_snap) t_next = std::min(t_next, next_snap_t);
            if (use_diag) t_next = std::min(t_next, next_diag_t);
            if (t_next < 1e299){
                double rem = t_next - (double)S.t;
                //if (rem > 1e-14 && rem < (double)S.dt) S.dt = (Real)rem;
                double tol_abs = time_tol(t_next);
		double tol_rel = 1e-6 * (double)S.dt;
		double tol = std::max(tol_abs, tol_rel);
                // Only trim if the remainder is meaningfully large; avoid sub‑ULP micro‑steps
                if (rem > tol && rem < 0.5*(double)S.dt) S.dt = (Real)rem;
            }
        }

	/* SSPRK(3,3) - fully explicit */
	// Stage 1
	launch_arakawa_J_mapped(d_psi,d_w,d_J,G.Nx,G.Ny,inv_dx,inv_deta,d_a_node,stream);
	launch_laplacian_fd_mapped(d_w, d_lap, G.Nx, G.Ny, inv_dx2, inv_deta2, d_a_node, d_a_edge, stream);
	launch_rhs_mapped(d_J,d_lap,d_w,d_rhs,G.Nx,G.Ny,nu,P.F0,P.nforce,G.h,P.lin_drag,d_y_node,stream); 
	if (P.mms) {// Stage 1 uses time t
		launch_mms_add_forcing(d_rhs, G.Nx, G.Ny, G.Lx, G.h,
				S.t, nu, P.lin_drag,
				P.mms_A0, P.mms_eps, P.mms_Om, P.mms_kx,
				d_y_node, stream);}

	launch_axpy(d_w1,d_w,d_rhs,S.dt,N,stream);
	poisson.solve(d_w1,d_psi,dy,psi_bot,psi_top,stream);
	launch_thom_wall_vorticity_mapped(d_w1,d_psi,G.Nx,G.Ny,d_dy_edge,psi_bot,psi_top,stream);

	// Stage 2
	launch_arakawa_J_mapped(d_psi,d_w1,d_J,G.Nx,G.Ny,inv_dx,inv_deta,d_a_node,stream);
	launch_laplacian_fd_mapped(d_w1,d_lap,G.Nx,G.Ny,inv_dx2,inv_deta2,d_a_node,d_a_edge,stream);
	launch_rhs_mapped(d_J,d_lap,d_w1,d_rhs,G.Nx,G.Ny,nu,P.F0,P.nforce,G.h,P.lin_drag,d_y_node,stream);  
	if (P.mms) {// Stage 2 uses time t+S.dt
		launch_mms_add_forcing(d_rhs, G.Nx, G.Ny, G.Lx, G.h,
				S.t+S.dt, nu, P.lin_drag,	P.mms_A0, P.mms_eps, P.mms_Om, P.mms_kx,
				d_y_node, stream);}

	launch_ssprk2(d_w2,d_w,d_w1,d_rhs,S.dt,N,stream);
	poisson.solve(d_w2,d_psi,dy,psi_bot,psi_top,stream);
	launch_thom_wall_vorticity_mapped(d_w2,d_psi,G.Nx,G.Ny,d_dy_edge,psi_bot,psi_top,stream);

	// Stage 3
	launch_arakawa_J_mapped(d_psi,d_w2,d_J,G.Nx,G.Ny,inv_dx,inv_deta,d_a_node,stream);
	launch_laplacian_fd_mapped(d_w2,d_lap,G.Nx,G.Ny,inv_dx2,inv_deta2,d_a_node,d_a_edge,stream);
	launch_rhs_mapped(d_J,d_lap,d_w2,d_rhs,G.Nx,G.Ny,nu,P.F0,P.nforce,G.h,P.lin_drag,d_y_node,stream);
	if (P.mms) {// Stage 3 uses time t
		launch_mms_add_forcing(d_rhs, G.Nx, G.Ny, G.Lx, G.h, S.t+0.5*S.dt, nu, P.lin_drag,
				P.mms_A0, P.mms_eps, P.mms_Om, P.mms_kx,d_y_node, stream);}

	launch_ssprk3(d_w,d_w,d_w2,d_rhs,S.dt,N,stream);
	poisson.solve(d_w,d_psi,dy,psi_bot,psi_top,stream);
	launch_thom_wall_vorticity_mapped(d_w,d_psi,G.Nx,G.Ny,d_dy_edge,psi_bot,psi_top,stream);

        if (profile){ cudaEventRecord(e_step1, stream); cudaEventSynchronize(e_step1); }
        S.t += S.dt; step++;

        // Snap time to scheduled targets within tolerance (prevents drift/micro-steps)
        if (use_snap){
            double tol_s = sched_tol((double)S.t, (double)S.dt);
            if (std::fabs((double)S.t - next_snap_t) <= tol_s) S.t = (Real)next_snap_t;
        }
        if (use_diag){
            double tol_d = sched_tol((double)S.t, (double)S.dt);
            if (std::fabs((double)S.t - next_diag_t) <= tol_d) S.t = (Real)next_diag_t;
        }
        // snapshots
        if (use_snap){
            while ((double)S.t + sched_tol((double)S.t, (double)S.dt) >= next_snap_t &&
                   next_snap_t <= (double)P.t_end + sched_tol((double)P.t_end,(double)S.dt))
            {
                if (mapped) launch_uv_from_psi_mapped(d_psi,d_u,d_v,G.Nx,G.Ny,inv_dx,inv_deta,d_a_node,stream);
                CUDA_CHECK(cudaDeviceSynchronize());
                save_snapshot_any(outdir,G.Nx,G.Ny,(Real)next_snap_t,d_psi,d_w,d_u,d_v,snap_fmt,G,P);
                log_event(prog_snap, prog_path, (double)next_snap_t, (double)S.dt);
                next_snap_t += snap_save_dt;
            }
        }

        // diagnostics (time-based)
        if (use_diag && ( (double)S.t + time_tol((double)S.t) >= next_diag_t )){
            double dtw = next_diag_t - last_diag_t; if (dtw < 0) dtw = 0;
            last_diag_t = next_diag_t;

	    Real umax,vmax;
	    launch_max_uv_from_psi_mapped(d_psi,G.Nx,G.Ny,inv_dx,inv_deta,d_a_node,d_umax,d_vmax,d_vmaxm,stream);
            CUDA_CHECK(cudaMemcpy(&umax,d_umax,sizeof(Real),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&vmax,d_vmax,sizeof(Real),cudaMemcpyDeviceToHost));

            Real sumK, sum_w2, sum_gradw2, Pin, sum_eps, p_omega_sum;
	    Real sum_omega_rhs = 0; //compute omega*rhs for diffusion-only test
	    Real mean_wwy_bot, mean_wwy_top; // wall flux integrand means ⟨ω ω_y⟩_x at y=±h

	    launch_kinetic_energy_mapped(d_psi,G.Nx,G.Ny,inv_dx,inv_deta,d_a_node,d_w_node,d_sumK,stream);
	    launch_enstrophy_diss_mapped_full(d_w,G.Nx,G.Ny,inv_dx,inv_deta,d_a_node,d_w_node,d_sum_w2,d_sum_gradw2,stream); // full-domain include walls
	    launch_mean_wall_omega_omegay_mapped(d_w,G.Nx,G.Ny,inv_deta,d_a_node,d_mean_wwy_bot,d_mean_wwy_top,stream); // wall enstrophy flux integrand ⟨ω ω_y⟩_x at y=±h 
	    //launch_enstrophy_diss_balance_mapped(d_w,G.Nx,G.Ny,inv_dx2,inv_deta2,d_a_node,d_a_edge,d_w_node,d_sum_w2,d_sum_gw2,stream); 
            //compute <w*rhs> for Enstrophy budget check purpose
            launch_omega_rhs_balance_mapped(d_w,d_rhs,G.Nx, G.Ny, d_w_node,d_sum_omega_rhs,stream);
            launch_power_input_mapped(d_psi,G.Nx,G.Ny,inv_deta,G.h,P.F0,P.nforce,d_a_node,d_y_node,d_w_node,d_pin,stream);
    	    launch_energy_dissipation_mapped(d_psi,d_w,G.Nx,G.Ny,inv_dx,inv_deta,d_a_node,d_a_edge,d_w_node,d_sum_eps,stream);
            launch_enstrophy_input_mapped_full(d_w,G.Nx,G.Ny,G.h,P.F0,P.nforce,d_y_node,d_w_node,d_p_omega,stream);

            CUDA_CHECK(cudaMemcpy(&sumK,d_sumK,sizeof(Real),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&sum_w2,d_sum_w2,sizeof(Real),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&sum_gradw2,d_sum_gradw2,sizeof(Real),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&Pin,d_pin,sizeof(Real),cudaMemcpyDeviceToHost));
    	    CUDA_CHECK(cudaMemcpy(&sum_eps,d_sum_eps,sizeof(Real),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&p_omega_sum,d_p_omega,sizeof(Real),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&sum_omega_rhs,d_sum_omega_rhs,sizeof(Real),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&mean_wwy_bot,d_mean_wwy_bot,sizeof(Real),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&mean_wwy_top,d_mean_wwy_top,sizeof(Real),cudaMemcpyDeviceToHost));

            Real sumK_area = sumK * invA_full; // Domain-averaged kinetic energy K(t) 
            Real Omega = 0.5 * sum_w2 * invA_full; // same Ω = ½⟨ω²⟩_A full-domain
	    Real eta     = nu * sum_gradw2 * invA_full; // ν⟨|∇ω|²⟩_A full-domain
            Real BOmega_wall = (nu>0.0) ? (nu * (mean_wwy_top - mean_wwy_bot) / (2.0 * G.h)) : (Real)0; // boundary integral ν/(2h)[⟨ω ω_y⟩_top-⟨ω ω_y⟩_bot]

            Real dOmega_rhs = sum_omega_rhs * invA_full; // new: dO/dt = <omega * rhs>_A,int
            Real Pin1  = Pin * invA_full; //Power input (full domain)
            Real epsilon = nu * sum_eps * invA_full; // Energy dissipation ε = ν⟨|∇u|²⟩_A
            Real P_Omega = p_omega_sum * invA_full; // Enstrophy input P_Ω = ⟨ω f_ω⟩_A,int
            Real mean_ome_bot, mean_ome_top;
            launch_mean_wall_omega(d_w,G.Nx,G.Ny,d_mean_ome_bot,d_mean_ome_top,stream);
            CUDA_CHECK(cudaMemcpy(&mean_ome_bot,d_mean_ome_bot,sizeof(Real),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&mean_ome_top,d_mean_ome_top,sizeof(Real),cudaMemcpyDeviceToHost));
            Real tauw_bot = - nu * mean_ome_bot;
            Real tauw_top = - nu * mean_ome_top;
            Real utau_inst = sqrt( 0.5*(fabs(tauw_bot)+fabs(tauw_top)) );
            Real Cf = (P.Ub!=0.0) ? 2.0 * (0.5*(fabs(tauw_bot)+fabs(tauw_top))) / (P.Ub*P.Ub) : std::numeric_limits<Real>::quiet_NaN();
	    Real Re_tau = (nu > 0.0) ? utau_inst * G.h / nu : 0.0;  // or NaN
            // Mean pressure gradient <p_x>_A from mean momentum balance (constant-flux):
            //   <p_x>_A = (tau_top - tau_bot)/(2h) + <F_x>_A - r*Ub
            const Real dpdx_mean = (tauw_top - tauw_bot) / (2.0 * G.h) + meanFx_A - P.lin_drag * P.Ub;

            if (dtw > 0){ tauw_abs_dt_sum += 0.5*(fabs(tauw_bot)+fabs(tauw_top))*dtw; T_diag += dtw; }

            std::ofstream f(diag_path, std::ios::app);
            f.setf(std::ios::scientific); f.precision(8);
            f << step << "," << S.t << "," << S.dt <<"," << umax << "," << vmax << ","
              << sumK_area << "," << Omega << "," << dOmega_rhs<<","<< eta <<"," <<BOmega_wall<<","<< Pin1 << "," << epsilon<<","<<P_Omega<<","
              << tauw_bot << "," << tauw_top << "," <<dpdx_mean<<","<< utau_inst << "," << Cf << "," << Re_tau << "\n";

            log_event(prog_diag, prog_path, (double)next_diag_t, (double)S.dt);
            next_diag_t += diag_save_dt;
        }

        // time-weighted profile accumulation
        if (step % statsEvery == 0){
            double dtw = (double)S.t - last_stats_t;
            if (dtw > 0){
                if (mapped) launch_row_moments_from_psi_mapped(d_psi,G.Nx,G.Ny,inv_dx,inv_deta,d_a_node,d_sum_u,d_sum_v,d_sum_u2,d_sum_v2,d_sum_uv,stream);
                std::vector<Real> U(G.Ny),V(G.Ny),U2(G.Ny),V2(G.Ny),UV(G.Ny);
                CUDA_CHECK(cudaMemcpy(U.data(),  d_sum_u,  G.Ny*sizeof(Real), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(V.data(),  d_sum_v,  G.Ny*sizeof(Real), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(U2.data(), d_sum_u2, G.Ny*sizeof(Real), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(V2.data(), d_sum_v2, G.Ny*sizeof(Real), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(UV.data(), d_sum_uv, G.Ny*sizeof(Real), cudaMemcpyDeviceToHost));
                for (int j=0;j<G.Ny;j++){
                    Real Ubar=U[j]/(Real)G.Nx, Vbar=V[j]/(Real)G.Nx;
                    Real U2b=U2[j]/(Real)G.Nx, V2b=V2[j]/(Real)G.Nx, UVb=UV[j]/(Real)G.Nx;
                    H_int_u[j]+=Ubar*dtw; H_int_v[j]+=Vbar*dtw;
                    H_int_u2[j]+=U2b*dtw; H_int_v2[j]+=V2b*dtw; H_int_uv[j]+=UVb*dtw;
                }
                T_accum += dtw; last_stats_t = (double)S.t;

                // NEW: optionally refresh the rolling profiles CSV
                if (profilesWriteEvery > 0 && (step % profilesWriteEvery) == 0){
                    // pick a u_tau estimate: time-averaged if we have diag samples, else instantaneous
                    Real utau_live = 0.0;
                    if (T_diag > 0){
                        utau_live = sqrt( tauw_abs_dt_sum / T_diag );
                    } else {
                        Real mean_ome_bot_now, mean_ome_top_now;
                        launch_mean_wall_omega(d_w, G.Nx, G.Ny, d_mean_ome_bot, d_mean_ome_top, 0);
                        CUDA_CHECK(cudaMemcpy(&mean_ome_bot_now, d_mean_ome_bot, sizeof(Real), cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaMemcpy(&mean_ome_top_now, d_mean_ome_top, sizeof(Real), cudaMemcpyDeviceToHost));
                        Real tauw_bot_now = - P.nu(G.h) * mean_ome_bot_now;
                        Real tauw_top_now = - P.nu(G.h) * mean_ome_top_now;
                        utau_live = sqrt( (Real)0.5 * (fabs(tauw_bot_now) + fabs(tauw_top_now)) );
                    }
                    write_profiles_csv(prof_path, G, P, (double)utau_live, T_accum,
                                       H_int_u, H_int_v, H_int_u2, H_int_v2, H_int_uv,
                                       G.y, G.a_node);
                }
            }
        }

        // 1D spectra accumulation (step-based)
        if (do_spec && (step % spectraEvery == 0)){
            // velocities for FFT
            if (mapped) launch_uv_from_psi_mapped(d_psi,d_u,d_v,G.Nx,G.Ny,inv_dx,inv_deta,d_a_node,stream);
            CUDA_CHECK(cudaDeviceSynchronize());
            spec1d.accumulate(d_u, d_v);
	    if ((prog_spec.samples + 1) %10000 == 0){
		    log_step(prog_spec, prog_path, (double)S.t, (double)S.dt, spectraEvery);
	    }
        }

        // timing
        if (profile && (step % timeEvery == 0)){
            float ms_step=0.0f, fft_ms=0.0f, tri_ms=0.0f;
            cudaEventElapsedTime(&ms_step, e_step0, e_step1);
            poisson.profile_get(fft_ms, tri_ms);
            std::ofstream g(time_path, std::ios::app);
            g.setf(std::ios::scientific); g.precision(6);
            g << step << "," << S.t << "," << S.dt << "," << ms_step << "," << fft_ms << "," << tri_ms << "\n";
        }
    }

    // time-weighted u_tau mean (fallback if no diag samples)
    Real utau_mean = 0.0;
    if (T_diag > 0){
        utau_mean = sqrt( tauw_abs_dt_sum / T_diag );
    } else {
        Real mean_ome_bot_now, mean_ome_top_now;
        launch_mean_wall_omega(d_w, G.Nx, G.Ny, d_mean_ome_bot, d_mean_ome_top, 0);
        CUDA_CHECK(cudaMemcpy(&mean_ome_bot_now, d_mean_ome_bot, sizeof(Real), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&mean_ome_top_now, d_mean_ome_top, sizeof(Real), cudaMemcpyDeviceToHost));
        Real tauw_bot_now = - P.nu(G.h) * mean_ome_bot_now;
        Real tauw_top_now = - P.nu(G.h) * mean_ome_top_now;
        utau_mean = sqrt( 0.5 * (fabs(tauw_bot_now) + fabs(tauw_top_now)) );
    }
 
    // NEW: final time-avged profiles write via helper (same format as rolling)
    write_profiles_csv(prof_path, G, P, utau_mean, T_accum,
                       H_int_u, H_int_v, H_int_u2, H_int_v2, H_int_uv,
                       G.y, G.a_node);

    // write spectra (time-averaged)
    if (do_spec){
        std::string spath = outdir + "/spectra_kx.csv";
        spec1d.write_csv(spath);
    }

    if (profile){ cudaEventDestroy(e_step0); cudaEventDestroy(e_step1); }
    if (mapped) {
	    CUDA_CHECK(cudaFree(d_a_node)); CUDA_CHECK(cudaFree(d_a_edge));
	    CUDA_CHECK(cudaFree(d_w_node)); CUDA_CHECK(cudaFree(d_y));
	    CUDA_CHECK(cudaFree(d_asub));   CUDA_CHECK(cudaFree(d_csup)); CUDA_CHECK(cudaFree(d_b0));
        CUDA_CHECK(cudaFree(d_y_node)); CUDA_CHECK(cudaFree(d_dy_edge));
    }

    // cleanup
    CUDA_CHECK(cudaFree(d_umax)); 
    CUDA_CHECK(cudaFree(d_vmax)); 
    CUDA_CHECK(cudaFree(d_vmaxm)); 
    CUDA_CHECK(cudaFree(d_p_omega)); 
    CUDA_CHECK(cudaFree(d_sum_eps));  // Add this line
    CUDA_CHECK(cudaFree(d_v)); CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_sumK)); CUDA_CHECK(cudaFree(d_pin));
    CUDA_CHECK(cudaFree(d_sum_w2));
    CUDA_CHECK(cudaFree(d_sum_gradw2));
    CUDA_CHECK(cudaFree(d_mean_ome_top)); CUDA_CHECK(cudaFree(d_mean_ome_bot));
    CUDA_CHECK(cudaFree(d_mean_wwy_bot));
    CUDA_CHECK(cudaFree(d_mean_wwy_top));
    CUDA_CHECK(cudaFree(d_sum_omega_rhs));
    CUDA_CHECK(cudaFree(d_sum_uv)); CUDA_CHECK(cudaFree(d_sum_v2)); CUDA_CHECK(cudaFree(d_sum_u2));
    CUDA_CHECK(cudaFree(d_sum_v));  CUDA_CHECK(cudaFree(d_sum_u));
    CUDA_CHECK(cudaFree(d_lap)); CUDA_CHECK(cudaFree(d_J));
    CUDA_CHECK(cudaFree(d_rhs)); CUDA_CHECK(cudaFree(d_psi));
    CUDA_CHECK(cudaFree(d_w2));  CUDA_CHECK(cudaFree(d_w1)); CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaDeviceSynchronize());

    //std::cout << "Done.\n";
    LOGP("Done. \n");
    return 0;
}
#endif
