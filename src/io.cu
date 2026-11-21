// Part of nse2d-channel-gpu: GPU DNS of 2D channel turbulence (vorticity-streamfunction).
// Run log, scheduling helpers, HDF5 snapshots and CSV writers.
#include "io.cuh"

#ifndef NO_HDF5
// ------- Minimal HDF5 read helpers (row-major [Ny, Nx] stored as write helper does) -------
bool h5_has_dataset(hid_t file, const char* name){
    return H5Lexists(file, name, H5P_DEFAULT) > 0;
}
double h5_read_attr_double(hid_t file, const char* name, double defv){
    if (H5Aexists(file, name)<=0) return defv;
    hid_t a = H5Aopen_name(file, name);
    double v=defv; H5Aread(a, H5T_NATIVE_DOUBLE, &v); H5Aclose(a); return v;
}
int h5_read_attr_int(hid_t file, const char* name, int defv){
    if (H5Aexists(file, name)<=0) return defv;
    hid_t a = H5Aopen_name(file, name);
    int v=defv; H5Aread(a, H5T_NATIVE_INT, &v); H5Aclose(a); return v;
}
void h5_read_2d(hid_t file, const char* name, Real* out, int Nx, int Ny){
    hid_t ds = H5Dopen(file, name, H5P_DEFAULT);
    if (ds < 0) throw std::runtime_error(std::string("HDF5: dataset not found: ")+name);

    hid_t sp = H5Dget_space(ds);
    int rank = H5Sget_simple_extent_ndims(sp);
    if (rank != 2){
        H5Sclose(sp); H5Dclose(ds);
        throw std::runtime_error(std::string("HDF5: rank != 2 for ")+name);
    }

    hsize_t dims[2]; H5Sget_simple_extent_dims(sp, dims, nullptr);
    const int D0 = (int)dims[0], D1 = (int)dims[1];

    const bool file_is_NyNx = (D0 == Ny && D1 == Nx);   // most common
    const bool file_is_NxNy = (D0 == Nx && D1 == Ny);   // swapped order

    if (!file_is_NyNx && !file_is_NxNy){
        H5Sclose(sp); H5Dclose(ds);
        throw std::runtime_error(
            "HDF5: dataset dims mismatch for " + std::string(name) +
            " file=(" + std::to_string(D0) + "," + std::to_string(D1) + ")" +
            " expected (" + std::to_string(Ny) + "," + std::to_string(Nx) + ") or transpose");
    }

    std::vector<Real> tmp((size_t)D0*D1);
    hid_t dt = (sizeof(Real)==8)? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;
    H5Dread(ds, dt, H5S_ALL, H5S_ALL, H5P_DEFAULT, tmp.data());

    if (file_is_NyNx){
        // file layout already matches out[j*Nx + i]
        std::memcpy(out, tmp.data(), (size_t)Nx*Ny*sizeof(Real));
    } else {
        // transpose: file is [Nx, Ny] -> we want [Ny, Nx]
        for (int j=0; j<Ny; ++j)
            for (int i=0; i<Nx; ++i)
                out[(size_t)j*Nx + i] = tmp[(size_t)i*Ny + j];
    }

    H5Sclose(sp); H5Dclose(ds);
}
#endif

bool dir_exists(const std::string& p){
    struct stat st; return ::stat(p.c_str(), &st)==0 && S_ISDIR(st.st_mode);
}
bool make_one(const std::string& p){
#if defined(_WIN32)
    return _mkdir(p.c_str())==0 || errno==EEXIST;
#else
    return ::mkdir(p.c_str(), 0755)==0 || errno==EEXIST;
#endif
}
bool ensure_dir_p(const std::string& p){
    if (p.empty() || dir_exists(p)) return true;
    size_t pos = p.find_last_of("/\\");
    if (pos != std::string::npos){
        if (!ensure_dir_p(p.substr(0,pos))) return false;
    }
    return make_one(p);
}

std::string fmt_hms(double sec){
    if (sec < 0) sec = 0;
    long s = (long)(sec + 0.5);
    int hh = (int)(s / 3600); s %= 3600;
    int mm = (int)(s / 60);  int ss = (int)(s % 60);
    char buf[32]; std::snprintf(buf,sizeof(buf), "%02d:%02d:%02d", hh, mm, ss);
    return std::string(buf);
}
void progress_append(const std::string& path, const std::string& line){
    std::ofstream f(path, std::ios::app); f << line << "\n";
}
size_t scheduled_count(double tend, double dt_save){
    if (dt_save <= 0) return 0;
    double n = std::floor(tend / dt_save + 1e-12) + 1.0; // include t=0
    if (n < 1.0) n = 1.0;
    return (size_t)n;
}
void log_event(EventProgress& P, const std::string& path, double t, double dt){
    using namespace std::chrono;
    auto now  = steady_clock::now();
    double since_last = duration<double>(now - P.last).count();
    double elapsed    = duration<double>(now - P.t0).count();
    size_t i = P.done + 1;
    double est_total = (P.total>0 && i>0) ? elapsed * (double)P.total / (double)i : 0.0;
    double eta = (est_total>0.0 ? est_total - elapsed : 0.0);

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << P.tag << " " << i << "/" << P.total << " complete, "
        << "t=" << std::setprecision(6) << t << ", "
        << "dt=" << std::scientific << dt << std::fixed << ", "
        << "wall_since_last=" << std::setprecision(3) << since_last << "s, "
        << "elapsed=" << elapsed << "s, "
        << "est_total=" << fmt_hms(est_total) << ", "
        << "ETA=" << fmt_hms(eta);

    progress_append(path, oss.str());
    //std::printf("%s\n", oss.str().c_str());
    P.last = now; P.done = i;
}
void log_step(StepProgress& P, const std::string& path, double t, double dt, int every){
    using namespace std::chrono;
    auto now  = steady_clock::now();
    double since_last = duration<double>(now - P.last).count();
    double elapsed    = duration<double>(now - P.t0).count();
    size_t i = P.samples + 1;

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << P.tag << " sample " << i << " (every " << every << " steps) complete, "
        << "t=" << std::setprecision(6) << t << ", "
        << "dt=" << std::scientific << dt << std::fixed << ", "
        << "wall_since_last=" << std::setprecision(3) << since_last << "s, "
        << "elapsed=" << elapsed << "s";

    progress_append(path, oss.str());
    //std::printf("%s\n", oss.str().c_str());
    P.last = now; P.samples = i;
}
// --- console + progress.log tee ---
void tee_progress(const std::string& path, const char* fmt, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n < 0) return;
    // write to console
    std::fwrite(buf, 1, (size_t)n, stdout);
    std::fflush(stdout);
    // append to progress.log
    std::ofstream p(path, std::ios::app);
    if (p) { p.write(buf, n); p.flush(); }
}

// ---------- tiny helpers (file + time scheduling) ----------
bool file_nonempty(const std::string& path){
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    return f.good() && f.tellg() > 0;
}
double time_tol(double t){
    using std::abs; using std::max;
    return 64.0 * std::numeric_limits<double>::epsilon() * max(1.0, abs(t));
}
double sched_tol(double t, double dt){
    using std::abs; using std::max;
    const double tol_abs = time_tol(t);
    const double tol_dt  = 1e-6 * max(1e-300, abs(dt));   // dt-relative tolerance
    return max(tol_abs, tol_dt);
}

double next_after(double t0, double dt){
    if (dt <= 0.0) return 1e300; // "never"
    double k   = std::floor((t0 + time_tol(t0)) / dt);
    double t1  = (k + 1.0) * dt;
    return (t1 <= t0 + time_tol(t0)) ? t1 + dt : t1;
}

/* -------------------- HDF5 helpers -------------------- */

#ifndef NO_HDF5
void h5_write_attr_double(hid_t where, const char* name, double v){
    hid_t sid = H5Screate(H5S_SCALAR);
    hid_t aid = H5Acreate2(where, name, H5T_NATIVE_DOUBLE, sid, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(aid, H5T_NATIVE_DOUBLE, &v); H5Aclose(aid); H5Sclose(sid);
}
void h5_write_attr_int(hid_t where, const char* name, int v){
    hid_t sid = H5Screate(H5S_SCALAR);
    hid_t aid = H5Acreate2(where, name, H5T_NATIVE_INT, sid, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(aid, H5T_NATIVE_INT, &v); H5Aclose(aid); H5Sclose(sid);
}
void h5_write_2d(hid_t file, const char* dname, const Real* data, int Nx, int Ny){
    hsize_t dims[2] = { (hsize_t)Ny, (hsize_t)Nx };
    hid_t space = H5Screate_simple(2, dims, NULL);
    hid_t dset  = H5Dcreate2(file, dname, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(dset); H5Sclose(space);
}
#endif

/* -------------------- Snapshots -------------------- */
void save_snapshot_any(const std::string& outdir, int Nx, int Ny, Real t,
                              const Real* d_psi, const Real* d_w,
                              const Real* d_u,   const Real* d_v,
                              SnapFmt fmt, const Grid& G, const Params& P)
{
    std::string sdir = outdir + "/snaps";
    ensure_dir_p(sdir);
    size_t N = (size_t)Nx*Ny;
    // decide which fields to copy + write
    const bool want_omega = true;                              // omega always written
    const bool want_psi   = (P.snap_fields == "all" || P.snap_fields == "omega_psi");
    const bool want_uv    = (P.snap_fields == "all");

    std::vector<Real> Hpsi(N), Homega(N), Hu(N), Hv(N);
    if (want_psi) Hpsi.resize(N);
    if (want_uv) {Hu.resize(N); Hv.resize(N); }

    CUDA_CHECK(cudaMemcpy(Homega.data(), d_w,   N*sizeof(Real), cudaMemcpyDeviceToHost));
    if (want_psi)
        CUDA_CHECK(cudaMemcpy(Hpsi.data(),   d_psi, N*sizeof(Real), cudaMemcpyDeviceToHost));
    if (want_uv){
        CUDA_CHECK(cudaMemcpy(Hu.data(),     d_u,   N*sizeof(Real), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(Hv.data(),     d_v,   N*sizeof(Real), cudaMemcpyDeviceToHost));
    }
#ifndef NO_HDF5
    if (fmt==SNAP_H5 || fmt==SNAP_BOTH){
        std::ostringstream fn; fn<<sdir<<"/snap_t"<<std::fixed<<std::setprecision(6)<<t<<".h5";
        hid_t file = H5Fcreate(fn.str().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        h5_write_2d(file, "omega", Homega.data(), Nx, Ny);
        if (want_psi) h5_write_2d(file, "psi", Hpsi.data(), Nx, Ny);
        if (want_uv){
            h5_write_2d(file, "u", Hu.data(), Nx, Ny);
            h5_write_2d(file, "v", Hv.data(), Nx, Ny);
        }
        // write y grid if stretched
        if (P.stretch!="none" && P.beta>0.0){
            // NEW (just write what you already have in G)
            if (!G.y.empty()) {
              h5_write_2d(file, "y", G.y.data(), G.Ny, 1);            // Ny×1 dataset
              h5_write_attr_double(file, "beta", P.beta);
              h5_write_attr_double(file, "stretch_code", P.stretch=="tanh" ? 1.0 : 0.0);
            }
        }

        h5_write_attr_double(file, "t", t);
        h5_write_attr_int(file, "Nx", Nx);
        h5_write_attr_int(file, "Ny", Ny);
        h5_write_attr_double(file, "Lx", G.Lx);
        h5_write_attr_double(file, "Ly", G.Ly());
        h5_write_attr_double(file, "h",  G.h);
        h5_write_attr_double(file, "dx", G.dx());
        h5_write_attr_double(file, "dy", G.dy());
        h5_write_attr_double(file, "nu", P.nu(G.h));
        H5Fclose(file);
    }
#endif
}

// ------NEW: helper to (re)write the current time-averaged mean/stress profiles CSV
void write_profiles_csv(const std::string& path,
                               const Grid& G, const Params& P,
                               double utau_mean, double T_accum,
                               const std::vector<Real>& H_int_u,
                               const std::vector<Real>& H_int_v,
                               const std::vector<Real>& H_int_u2,
                               const std::vector<Real>& H_int_v2,
                               const std::vector<Real>& H_int_uv,
                               const std::vector<Real>& y_node,
                               const std::vector<Real>& a_node)
{
    std::ofstream f(path, std::ios::out); // overwrite
    f << "j,y,yplus,U,V,uu,vv,uv,Uplus,tau_total\n";
    for (int j=0; j<G.Ny; ++j){
        Real y    = (P.stretch!="none" && P.beta>0.0 ? y_node[j] : (-G.h + (Real)j * G.dy()));
        Real ybot = y + G.h;
        Real Ubar = (T_accum>0? H_int_u[j] / T_accum : 0.0);
        Real Vbar = (T_accum>0? H_int_v[j] / T_accum : 0.0);
        Real U2b  = (T_accum>0? H_int_u2[j]/ T_accum : 0.0);
        Real V2b  = (T_accum>0? H_int_v2[j]/ T_accum : 0.0);
        Real UVb  = (T_accum>0? H_int_uv[j]/ T_accum : 0.0);

        Real uu = fmax((Real)0.0, U2b - Ubar*Ubar);
        Real vv = fmax((Real)0.0, V2b - Vbar*Vbar);
        Real uv = UVb - Ubar*Vbar;
        // dU/dy via centered (one-sided at walls), using running mean U
        Real Ujp = (T_accum>0? H_int_u[ std::min(j+1,G.Ny-1) ]/T_accum : 0.0);
        Real Ujm = (T_accum>0? H_int_u[ std::max(j-1,0)     ]/T_accum : 0.0);
        Real dUdy;
        if (P.stretch!="none" && P.beta>0.0){
            Real inv_deta = (Real)( (G.Ny>1) ? (0.5*(G.Ny-1)) : 0.0 ); // 1/Δη with Δη=2/(Ny-1)
            Real a = a_node[j];
            if      (j==0)       dUdy = a * (Ujp - Ubar) * inv_deta;
            else if (j==G.Ny-1)  dUdy = a * (Ubar - Ujm) * inv_deta;
            else                 dUdy = a * (Ujp - Ujm) * 0.5 * inv_deta;
        } else {
            if      (j==0)       dUdy = (Ujp - Ubar) / G.dy();
            else if (j==G.Ny-1)  dUdy = (Ubar - Ujm) / G.dy();
            else                 dUdy = (Ujp - Ujm) * 0.5 / G.dy();
        }

        Real tau_total = P.nu(G.h) * dUdy - uv;
        Real yplus = (utau_mean>0? utau_mean * ybot / P.nu(G.h) : 0.0);
        Real Uplus = (utau_mean>0? Ubar / utau_mean : 0.0);

        f.setf(std::ios::scientific); f.precision(8);
        f << j << "," << y << "," << yplus << ","
          << Ubar << "," << Vbar << "," << uu << "," << vv << "," << uv << ","
          << Uplus << "," << tau_total << "\n";
    }
}

