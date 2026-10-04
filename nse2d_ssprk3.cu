/*
* 
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <ctime>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#if defined(_WIN32)
#include <direct.h>
#endif

#include <stdexcept>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math_constants.h>
#include <limits>

#ifndef NO_HDF5
#include <hdf5.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

using Real = double;
#ifndef NO_HDF5
// ------- Minimal HDF5 read helpers (row-major [Ny, Nx] stored as write helper does) -------
static bool h5_has_dataset(hid_t file, const char* name){
    return H5Lexists(file, name, H5P_DEFAULT) > 0;
}
static double h5_read_attr_double(hid_t file, const char* name, double defv){
    if (H5Aexists(file, name)<=0) return defv;
    hid_t a = H5Aopen_name(file, name);
    double v=defv; H5Aread(a, H5T_NATIVE_DOUBLE, &v); H5Aclose(a); return v;
}
static int h5_read_attr_int(hid_t file, const char* name, int defv){
    if (H5Aexists(file, name)<=0) return defv;
    hid_t a = H5Aopen_name(file, name);
    int v=defv; H5Aread(a, H5T_NATIVE_INT, &v); H5Aclose(a); return v;
}
static void h5_read_2d(hid_t file, const char* name, Real* out, int Nx, int Ny){
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

static bool dir_exists(const std::string& p){
    struct stat st; return ::stat(p.c_str(), &st)==0 && S_ISDIR(st.st_mode);
}
static bool make_one(const std::string& p){
#if defined(_WIN32)
    return _mkdir(p.c_str())==0 || errno==EEXIST;
#else
    return ::mkdir(p.c_str(), 0755)==0 || errno==EEXIST;
#endif
}
static bool ensure_dir_p(const std::string& p){
    if (p.empty() || dir_exists(p)) return true;
    size_t pos = p.find_last_of("/\\");
    if (pos != std::string::npos){
        if (!ensure_dir_p(p.substr(0,pos))) return false;
    }
    return make_one(p);
}

struct Grid {
    int Nx{1024};
    int Ny{1025};
    Real Lx{12.566370612};
    Real h {1.0};
    // --- Moved from YMetrics to here ---
    std::vector<Real> y;       // Ny
    std::vector<Real> a_node;  // Ny (a_j = 1/y_eta at nodes)
    std::vector<Real> a_edge;  // Ny-1 (a at j+1/2 edges)
    std::vector<Real> dy_edge; // Ny-1 (y_{j+1}-y_j)
    std::vector<Real> w_node;  // Ny (trapezoid weights)
    
    __host__ __device__ inline Real dx() const { return Lx / (Real)Nx; }
    __host__ __device__ inline Real dy() const { return (2.0*h) / (Real)(Ny-1); }
    __host__ __device__ inline Real Ly() const { return 2.0*h; }
};
struct Params {
    Real Ub   {1.0};
    // --- y-stretch mapping controls ---
    std::string stretch{"none"}; // "none" or "tanh"
    std::string ytable{"none"}; // "none" or "tanh"
    Real beta{0.0};
    // --- Time integration ---
    std::string integrator{"imex3"};  // "ssprk3" (explicit) or "imex" (2nd order) or "imex3" (3rd order)

    Real Reb  {10000.0};
    Real nu_override{-1}; // <0 means "use 2h*Ub/Reb"; otherwise use this absolute ν
    Real nu(Real h) const { return (nu_override>=0 ? nu_override : (2.0*h*Ub)/Reb); }

    Real dt_init {1e-3};
    Real dt_max  {1e-2};
    Real cfl     {0.5};
    Real cvisc   {0.5};
    bool adapt   {true};
    Real t_end   {200.0};
    Real F0      {0.0};
    int  nforce  {1};
    Real lin_drag{0.0};
    std::string progress_path{}; // path to progress.log so utilities can tee messages without changing signatures
    // NEW: resume controls
    bool resume{false};      // if true, start from snapshot time instead of 0
    Real t0{0.0};            // initial time (filled from snapshot 't' when resume=true)

    // --- Initial condition controls ---
    std::string init{"load"};     // laminar | rand | ts | mix | load
    Real amp{0.5};               // amplitude parameter (see notes above)
    Real alpha{1.2};             // TS streamwise wavenumber (default 2π/Lx if 0)
    int  my{1};                  // TS wall-normal index
    Real phase{0.0};             // TS phase [rad]
    Real rand_amp{0.0};          // legacy alias (if init not given)
    unsigned long long seed{123456789ULL};
    // snapshot loading
    std::string load_path{""};
    bool load_use_psi{false}; // if true, read psi and rebuild omega; else read omega
    std::string snap_fields{"all"}; //which fields to write: "all" "omega" "omega_psi"

    // Controls for 'rand' (turbulent vorticity IC)
    int  rand_nx{48};            // # streamwise modes to sample
    int  rand_mymax{8};          // max wall-normal sine index (>=1)
    Real rand_k0{0.0};           // center kx (rad/unit); 0 => 6*(2π/Lx)
    Real rand_sigma{0.6};        // log bandwidth for kx sampling
    Real rand_amp_abs{0.0};      // absolute vorticity amplitude; if 0, falls back to 'amp'
 // --- Manufactured-solution (MMS) verification controls ---
 // If mms=true, the solver should add f_omega(x,y,t) into the vorticity RHS.
 // Typically you also use --init mms (so ω(t=0) matches ω_exact(t=0)).
    bool mms{false};          // enable MMS forcing f_ω
    Real mms_A0{1e-3};        // base amplitude A0
    Real mms_eps{0.1};        // modulation amplitude ε (0 => steady in time)
    Real mms_Om{1.0};         // modulation frequency Ω
    int  mms_kx{1};           // streamwise mode index (k = 2π*mms_kx/Lx)
};

#define CUDA_CHECK(e) do { cudaError_t err__=(e); if (err__!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); std::abort(); } } while(0)
#define CUFFT_CHECK(e) do { cufftResult res__=(e); if (res__!=CUFFT_SUCCESS){ \
  fprintf(stderr,"cuFFT error %s:%d: code %d\n", __FILE__, __LINE__, int(res__)); std::abort(); } } while(0)

inline dim3 blocksFor2D(int NX, int NY, int tx=32, int ty=8){
    return dim3( (NX + tx - 1)/tx, (NY + ty - 1)/ty );
}

/* -------------------- Progress logger -------------------- */
//static bool g_quiet_progress = false;   // set by --quiet
static std::string fmt_hms(double sec){
    if (sec < 0) sec = 0;
    long s = (long)(sec + 0.5);
    int hh = (int)(s / 3600); s %= 3600;
    int mm = (int)(s / 60);  int ss = (int)(s % 60);
    char buf[32]; std::snprintf(buf,sizeof(buf), "%02d:%02d:%02d", hh, mm, ss);
    return std::string(buf);
}
static void progress_append(const std::string& path, const std::string& line){
    std::ofstream f(path, std::ios::app); f << line << "\n";
}
struct EventProgress {
    std::string tag;
    size_t total{0}, done{0};
    std::chrono::steady_clock::time_point t0, last;
    EventProgress() {}
    EventProgress(const std::string& t, size_t T): tag(t), total(T), done(0){
        t0 = std::chrono::steady_clock::now();
        last = t0;
    }
};
struct StepProgress {
    std::string tag;
    size_t samples{0};
    std::chrono::steady_clock::time_point t0, last;
    StepProgress() {}
    StepProgress(const std::string& t): tag(t), samples(0){
        t0 = std::chrono::steady_clock::now();
        last = t0;
    }
};
static inline size_t scheduled_count(double tend, double dt_save){
    if (dt_save <= 0) return 0;
    double n = std::floor(tend / dt_save + 1e-12) + 1.0; // include t=0
    if (n < 1.0) n = 1.0;
    return (size_t)n;
}
static void log_event(EventProgress& P, const std::string& path, double t, double dt){
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
static void log_step(StepProgress& P, const std::string& path, double t, double dt, int every){
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
static inline void tee_progress(const std::string& path, const char* fmt, ...) {
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
static inline bool file_nonempty(const std::string& path){
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    return f.good() && f.tellg() > 0;
}
static inline double time_tol(double t){
    using std::abs; using std::max;
    return 64.0 * std::numeric_limits<double>::epsilon() * max(1.0, abs(t));
}
static inline double sched_tol(double t, double dt){
    using std::abs; using std::max;
    const double tol_abs = time_tol(t);
    const double tol_dt  = 1e-6 * max(1e-300, abs(dt));   // dt-relative tolerance
    return max(tol_abs, tol_dt);
}

static inline double next_after(double t0, double dt){
    if (dt <= 0.0) return 1e300; // "never"
    double k   = std::floor((t0 + time_tol(t0)) / dt);
    double t1  = (k + 1.0) * dt;
    return (t1 <= t0 + time_tol(t0)) ? t1 + dt : t1;
}

/* -------------------- Operators & kernels (numerics preserved) -------------------- */

// given u = p_y, v = -p_x, w = -\Delta p   : uniform Arakawa_J
// this kernel computes advective term u \nabla w = uw_x + vw_y = p_yw_x-p_xw_y=J(w,p)
__global__ void arakawa_J_kernel(const Real* __restrict__ psi, const Real* __restrict__ omg,
    Real* __restrict__ J, int Nx, int Ny, Real inv_dx, Real inv_dy)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix >= Nx || iy >= Ny) return;
    if (iy==0 || iy==Ny-1){ J[(size_t)iy*Nx + ix] = 0.0; return; }

    auto wrap = [&](int j)->int { return (j<0? j+Nx : (j>=Nx? j-Nx : j)); };
    int xm=wrap(ix-1), xp=wrap(ix+1);
    int ym=iy-1, yp=iy+1;
    size_t idx = (size_t)iy*Nx + ix;

    Real psi_ip = psi[(size_t)iy*Nx + xp], psi_im = psi[(size_t)iy*Nx + xm];
    Real psi_jp = psi[(size_t)yp*Nx + ix], psi_jm = psi[(size_t)ym*Nx + ix];
    Real omg_ip = omg[(size_t)iy*Nx + xp], omg_im = omg[(size_t)iy*Nx + xm];
    Real omg_jp = omg[(size_t)yp*Nx + ix], omg_jm = omg[(size_t)ym*Nx + ix];

    Real J1 = ( (psi_ip-psi_im)*(omg_jp-omg_jm) - (psi_jp-psi_jm)*(omg_ip-omg_im) ) * 0.25 * inv_dx * inv_dy;

    Real psi_ip_jp = psi[(size_t)yp*Nx + xp], psi_ip_jm = psi[(size_t)ym*Nx + xp];
    Real psi_im_jp = psi[(size_t)yp*Nx + xm], psi_im_jm = psi[(size_t)ym*Nx + xm];
    Real omg_ip_jp = omg[(size_t)yp*Nx + xp], omg_ip_jm = omg[(size_t)ym*Nx + xp];
    Real omg_im_jp = omg[(size_t)yp*Nx + xm], omg_im_jm = omg[(size_t)ym*Nx + xm];

    Real J2 = (  psi_ip*(omg_ip_jp-omg_ip_jm) - psi_im*(omg_im_jp-omg_im_jm)
               - psi_jp*(omg_ip_jp-omg_im_jp) + psi_jm*(omg_ip_jm-omg_im_jm) ) * 0.25 * inv_dx * inv_dy;

    Real J3 = ( -omg_ip*(psi_ip_jp-psi_ip_jm) + omg_im*(psi_im_jp-psi_im_jm)
               + omg_jp*(psi_ip_jp-psi_im_jp) - omg_jm*(psi_ip_jm-psi_im_jm) ) * 0.25 * inv_dx * inv_dy;

    J[idx] = -(J1 + J2 + J3) / 3.0;
}

// -------------------- y-metrics builder (host-side) --------------------
// Build y, a_node, a_edge, dy_edge, w_node on host; fills G.y* vectors.
// P.stretch: "none"|"tanh"|"table"; P.beta used by tanh; P.ytable used by table.
static void build_y_metrics(const Grid& Gin, const Params& P, Grid& Gout) {
    Gout = Gin; // copy scalar members
    const int Ny = Gin.Ny; const Real h = Gin.h;
    if (Ny <= 2 || P.stretch=="none" || P.stretch=="uniform") {
        // Uniform grid
        Gout.y.resize(Ny);
        const Real dy = Gin.dy();
        for (int j=0; j<Ny; ++j) Gout.y[j] = -h + j*dy;

        // Δη = 2/(Ny-1), y_eta = dy/Δη  => a = 1/y_eta
        const Real deta = 2.0/(Real)(Ny-1);
        const Real y_eta = dy/deta;
        const Real a = 1.0 / y_eta;

        Gout.a_node.assign(Ny,  a);
        Gout.a_edge.assign(Ny-1,a);
        Gout.dy_edge.assign(Ny-1, dy);
        Gout.w_node.assign(Ny,   dy);
        Gout.w_node.front() *= 0.5; Gout.w_node.back() *= 0.5;
        return;
    }

    // Non-uniform
    Gout.y.resize(Ny);
    auto eta = [&](int j)->Real { return -1.0 + 2.0 * (Real)j / (Real)(Ny-1); };

    if (P.stretch=="tanh") {
        const Real beta = (P.beta>0 ? P.beta : 2.5);
        auto y_of_eta = [&](Real e)->Real { return h * std::tanh(beta*e) / std::tanh(beta); };
        for (int j=0; j<Ny; ++j) Gout.y[j] = y_of_eta( eta(j) );

        // y_eta analytically: h*β*sech^2(βη) / tanh(β)
        Gout.a_node.resize(Ny);
        for (int j=0; j<Ny; ++j) {
            Real e = eta(j);
            Real sech = 1.0 / std::cosh(beta*e);
            Real y_eta = h * beta * (sech*sech) / std::tanh(beta);
            Gout.a_node[j] = 1.0 / y_eta;
        }
    } else if (P.stretch=="table") {
#ifndef NO_HDF5
        if (P.ytable.empty()) throw std::runtime_error("--ytable required for --stretch table");
        // Expect dataset /y of length Ny (column or row)
        hid_t f = H5Fopen(P.ytable.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (f<0) throw std::runtime_error("Cannot open --ytable: "+P.ytable);
        if (!h5_has_dataset(f,"y")){ H5Fclose(f); throw std::runtime_error("--ytable missing dataset /y"); }

        // Read as a vector (support [Ny,1] or [1,Ny])
        std::vector<Real> yt(Ny);
        bool ok = false;
        // Try [Ny,1]
        try { h5_read_2d(f,"y", yt.data(), Ny, 1); ok = true; } catch(...) {}
        if (!ok) { // Try [1,Ny]
            try { h5_read_2d(f,"y", yt.data(), 1, Ny); ok = true; } catch(...) {}
        }
        H5Fclose(f);
        if (!ok) throw std::runtime_error("--ytable: could not read /y array");
        Gout.y = std::move(yt);

        // y_eta by finite difference in η
        const Real deta = 2.0/(Real)(Ny-1);
        Gout.a_node.resize(Ny);
        for (int j=0; j<Ny; ++j) {
            Real yeta;
            if (j==0)           yeta = (Gout.y[1]    - Gout.y[0])     / deta;
            else if (j==Ny-1)   yeta = (Gout.y[Ny-1] - Gout.y[Ny-2])  / deta;
            else                yeta = (Gout.y[j+1]  - Gout.y[j-1])   / (2.0*deta);
            Gout.a_node[j] = 1.0 / yeta;
        }
#else
        throw std::runtime_error("Built with NO_HDF5: cannot --stretch table");
#endif
    } else {
        throw std::runtime_error("Unknown --stretch '"+P.stretch+"'");
    }

    // Edges, spacings, trapezoid weights
    Gout.a_edge.resize(Ny-1);
    for (int j=0; j<Ny-1; ++j) Gout.a_edge[j] = 0.5*(Gout.a_node[j] + Gout.a_node[j+1]);

    Gout.dy_edge.resize(Ny-1);
    for (int j=0; j<Ny-1; ++j) Gout.dy_edge[j] = Gout.y[j+1] - Gout.y[j];

    Gout.w_node.assign(Ny, 0.0);
    Gout.w_node[0]      = 0.5 * Gout.dy_edge[0];
    Gout.w_node[Ny-1]   = 0.5 * Gout.dy_edge[Ny-2];
    for (int j=1; j<Ny-1; ++j) Gout.w_node[j] = 0.5*(Gout.dy_edge[j-1]+Gout.dy_edge[j]);
}

// ===================== MAPPED-Y VARIANTS (Phase-1) =====================
__global__ void uv_from_psi_mapped_kernel(const Real* __restrict__ psi,
    Real* __restrict__ u, Real* __restrict__ v,
    int Nx, int Ny, Real inv_dx, Real inv_deta,
    const Real* __restrict__ a_node)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix >= Nx || iy >= Ny) return;
    auto wrap = [&](int j)->int { return (j<0? j+Nx : (j>=Nx? j-Nx : j)); };
    int xm=wrap(ix-1), xp=wrap(ix+1);
    int ym=(iy==0? 0:iy-1), yp=(iy==Ny-1? Ny-1:iy+1);
    Real dpsideta = (psi[(size_t)yp*Nx + ix] - psi[(size_t)ym*Nx + ix]) * 0.5 * inv_deta;
    Real dpsidx   = (psi[(size_t)iy*Nx + xp] - psi[(size_t)iy*Nx + xm]) * 0.5 * inv_dx;
    Real a = a_node[iy];
    u[(size_t)iy*Nx + ix] = a * dpsideta;
    v[(size_t)iy*Nx + ix] = - dpsidx;
}
inline void launch_uv_from_psi_mapped(const Real* psi, Real* u, Real* v, int Nx, int Ny,
                                      Real inv_dx, Real inv_deta, const Real* d_a_node, cudaStream_t s=0)
{
    dim3 tb(32,8), gb = blocksFor2D(Nx,Ny,tb.x,tb.y);
    uv_from_psi_mapped_kernel<<<gb,tb,0,s>>>(psi,u,v,Nx,Ny,inv_dx,inv_deta,d_a_node);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void omega_from_psi_interior_mapped_kernel(Real* __restrict__ w, const Real* __restrict__ psi,
    int Nx, int Ny, Real inv_dx2, Real inv_deta2,
    const Real* __restrict__ a_node,
    const Real* __restrict__ a_edge)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix >= Nx || iy >= Ny) return;
    if (iy==0 || iy==Ny-1) return;
    auto wrap = [&](int j)->int { return (j<0? j+Nx : (j>=Nx? j-Nx : j)); };
    int xm=wrap(ix-1), xp=wrap(ix+1);
    size_t idx = (size_t)iy*Nx + ix;

    Real lapx = psi[(size_t)iy*Nx + xp] - 2.0*psi[idx] + psi[(size_t)iy*Nx + xm];
    // conservative mapped y-Laplacian: a_node[j] * [ a_edge[j+1/2](ψ_{j+1}-ψ_j) - a_edge[j-1/2](ψ_j-ψ_{j-1}) ] / Δη^2
    Real aN = a_node[iy];
    Real ap = a_edge[iy];     // j+1/2
    Real am = a_edge[iy-1];   // j-1/2
    Real lapy = aN * ( ap*(psi[(size_t)(iy+1)*Nx + ix] - psi[idx]) - am*(psi[idx] - psi[(size_t)(iy-1)*Nx + ix]) );

    w[idx] = -(inv_dx2*lapx + inv_deta2*lapy);
}
inline void launch_omega_from_psi_interior_mapped(Real* w, const Real* psi, int Nx, int Ny,
                                                  Real inv_dx2, Real inv_deta2,
                                                  const Real* d_a_node, const Real* d_a_edge, cudaStream_t s=0)
{
    dim3 tb(32,8), gb = blocksFor2D(Nx,Ny,tb.x,tb.y);
    omega_from_psi_interior_mapped_kernel<<<gb,tb,0,s>>>(w,psi,Nx,Ny,inv_dx2,inv_deta2,d_a_node,d_a_edge);
    CUDA_CHECK(cudaGetLastError());
}
__global__ void thom_wall_vorticity_mapped_kernel(Real* __restrict__ w, const Real* __restrict__ psi,
    int Nx, int Ny, const Real* __restrict__ dy_edge, Real psi_bot, Real psi_top)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= Nx) return;

#ifdef FREE_SLIP_WALLS
    //free-slip wall
    w[i] = 0;
    w[(size_t)(Ny-1)*Nx + i] = 0;
#else
    // second-order no-slip wall
    // ---------- bottom wall (j = 0) ----------
    {
	    // distances from wall y0 to first two interior nodes
	    Real dy1 = dy_edge[0];     // y1 - y0
	    Real dy2 = dy_edge[1];     // y2 - y1

	    // coefficients for ψ''(y0) ≈ A*(ψ1-ψ0) + B*(ψ2-ψ0)
	    Real A = (Real)2.0 * ( (Real)1.0/(dy1*dy1) + (Real)1.0/(dy1*dy2) );
	    Real B = (Real)(-2.0) * dy1 / ( dy2 * (dy1 + dy2) * (dy1 + dy2) );

	    size_t j1 = (size_t)1 * Nx + i;
	    size_t j2 = (size_t)2 * Nx + i;

	    Real dpsi1 = psi[j1] - psi_bot;
	    Real dpsi2 = psi[j2] - psi_bot;

	    Real d2psi = A * dpsi1 + B * dpsi2;
	    w[i] = -d2psi;  // ω = -ψ_yy at bottom wall
    }

    // ---------- top wall (j = Ny-1) ----------
    {
	    // distances from top wall y_{Ny-1} downwards
	    Real dy1 = dy_edge[Ny-2];  // y_{Ny-1} - y_{Ny-2}
    Real dy2 = dy_edge[Ny-3];  // y_{Ny-2} - y_{Ny-3}

    Real A = (Real)2.0 * ( (Real)1.0/(dy1*dy1) + (Real)1.0/(dy1*dy2) );
    Real B = (Real)(-2.0) * dy1 / ( dy2 * (dy1 + dy2) * (dy1 + dy2) );

    size_t j1 = (size_t)(Ny-2)*Nx + i;
    size_t j2 = (size_t)(Ny-3)*Nx + i;

    Real dpsi1 = psi[j1] - psi_top;
    Real dpsi2 = psi[j2] - psi_top;

    Real d2psi = A * dpsi1 + B * dpsi2;
    w[(size_t)(Ny-1)*Nx + i] = -d2psi;  // ω at top wall
    }

   // Real dyb = dy_edge[0];  //dy_edge[j] = y[j+1] - y[j]
   // Real dyt = dy_edge[Ny-2];
   // w[i] = -2.0/(dyb*dyb) * (psi[Nx + i] - psi_bot);  //first-order wall
   // w[(size_t)(Ny-1)*Nx + i] = -2.0/(dyt*dyt) * (psi[(size_t)(Ny-2)*Nx + i] - psi_top);
#endif
}
/*__global__ void thom_wall_vorticity_mapped_kernel(Real* __restrict__ w,
                                                  const Real* __restrict__ psi,
                                                  int Nx, int Ny,
                                                  const Real* __restrict__ dy_edge,
                                                  Real psi_bot, Real psi_top)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= Nx) return;

#ifdef FREE_SLIP_WALLS
    // Free-slip: ω_wall = 0
    w[i] = (Real)0;
    w[(size_t)(Ny-1)*Nx + i] = (Real)0;
#else
    // ===== 2nd-order no-slip Thom wall on a mapped, non-uniform y-grid =====
    // Use a 4-point one-sided stencil for ψ'' at the wall:
    //
    //   h1 = y1 - y0,  h2 = y2 - y1,  h3 = y3 - y2
    //
    //   ψ''(y0) ≈ a0 ψ0 + a1 ψ1 + a2 ψ2 + a3 ψ3
    //
    //   a0 = 2 (3h1 + 2h2 + h3) / [ h1 (h1^2 + 2 h1 h2 + h1 h3 + h2^2 + h2 h3) ]
    //   a1 = 2 (-2h1 - 2h2 - h3) / [ h1 h2 (h2 + h3) ]
    //   a2 = 2 ( 2h1 +  h2 + h3) / [ h2 h3 (h1 + h2) ]
    //   a3 = 2 (-2h1 -  h2     ) / [ h3 (h1 h2 + h1 h3 + h2^2 + 2 h2 h3 + h3^2) ]
    //
    // For a uniform grid h1=h2=h3=Δy this becomes
    //   ψ''(0) ≈ ( 2ψ0 - 5ψ1 + 4ψ2 - ψ3 ) / Δy^2.

    // ---- bottom wall (j = 0) ----
    if (Ny >= 4) {
        Real h1b = dy_edge[0];
        Real h2b = dy_edge[1];
        Real h3b = dy_edge[2];

        Real h1b2 = h1b*h1b;
        Real h2b2 = h2b*h2b;
        Real h3b2 = h3b*h3b;

        Real denom0b = h1b * ( h1b2 + (Real)2.0*h1b*h2b + h1b*h3b
                             + h2b2 + h2b*h3b );
        Real denom3b = h3b * ( h1b*h2b + h1b*h3b + h2b2
                             + (Real)2.0*h2b*h3b + h3b2 );

        Real a0b = (Real)2.0 * ( (Real)3.0*h1b + (Real)2.0*h2b + h3b ) / denom0b;
        Real a1b = (Real)2.0 * (-( (Real)2.0*h1b + (Real)2.0*h2b + h3b) )
                   / ( h1b*h2b*(h2b + h3b) );
        Real a2b = (Real)2.0 * ( (Real)2.0*h1b + h2b + h3b )
                   / ( h2b*h3b*(h1b + h2b) );
        Real a3b = (Real)2.0 * (-( (Real)2.0*h1b + h2b) ) / denom3b;

        size_t j0b = (size_t)0   * Nx + i;
        size_t j1b = (size_t)1   * Nx + i;
        size_t j2b = (size_t)2   * Nx + i;
        size_t j3b = (size_t)3   * Nx + i;

        Real psi0b = psi[j0b];   // equals psi_bot
        Real psi1b = psi[j1b];
        Real psi2b = psi[j2b];
        Real psi3b = psi[j3b];

        Real d2psib = a0b*psi0b + a1b*psi1b + a2b*psi2b + a3b*psi3b;
        w[i] = -d2psib;          // ω = −ψ_yy at bottom wall
    } else {
        // Tiny Ny fallback: 1st-order Thom to avoid out-of-bounds
        Real dyb = dy_edge[0];
        w[i] = -(Real)2.0/(dyb*dyb) * ( psi[(size_t)1*Nx + i] - psi_bot );
    }

    // ---- top wall (j = Ny-1) ----
    if (Ny >= 4) {
        Real h1t = dy_edge[Ny-2];  // y_{Ny-1} - y_{Ny-2}
        Real h2t = dy_edge[Ny-3];  // y_{Ny-2} - y_{Ny-3}
        Real h3t = dy_edge[Ny-4];  // y_{Ny-3} - y_{Ny-4}

        Real h1t2 = h1t*h1t;
        Real h2t2 = h2t*h2t;
        Real h3t2 = h3t*h3t;

        Real denom0t = h1t * ( h1t2 + (Real)2.0*h1t*h2t + h1t*h3t
                             + h2t2 + h2t*h3t );
        Real denom3t = h3t * ( h1t*h2t + h1t*h3t + h2t2
                             + (Real)2.0*h2t*h3t + h3t2 );

        Real a0t = (Real)2.0 * ( (Real)3.0*h1t + (Real)2.0*h2t + h3t ) / denom0t;
        Real a1t = (Real)2.0 * (-( (Real)2.0*h1t + (Real)2.0*h2t + h3t) )
                   / ( h1t*h2t*(h2t + h3t) );
        Real a2t = (Real)2.0 * ( (Real)2.0*h1t + h2t + h3t )
                   / ( h2t*h3t*(h1t + h2t) );
        Real a3t = (Real)2.0 * (-( (Real)2.0*h1t + h2t) ) / denom3t;

        size_t j0t = (size_t)(Ny-1)*Nx + i;
        size_t j1t = (size_t)(Ny-2)*Nx + i;
        size_t j2t = (size_t)(Ny-3)*Nx + i;
        size_t j3t = (size_t)(Ny-4)*Nx + i;

        Real psi0t = psi[j0t];   // equals psi_top
        Real psi1t = psi[j1t];
        Real psi2t = psi[j2t];
        Real psi3t = psi[j3t];

        Real d2psit = a0t*psi0t + a1t*psi1t + a2t*psi2t + a3t*psi3t;
        w[(size_t)(Ny-1)*Nx + i] = -d2psit;
    } else {
        Real dyt = dy_edge[Ny-2];
        w[(size_t)(Ny-1)*Nx + i] =
            -(Real)2.0/(dyt*dyt) * ( psi[(size_t)(Ny-2)*Nx + i] - psi_top );
    }
#endif
}*/

inline void launch_thom_wall_vorticity_mapped(Real* w, const Real* psi, int Nx, int Ny,
                                              const Real* d_dy_edge, Real psi_bot, Real psi_top, cudaStream_t s=0)
{
    dim3 tb(256), gb((Nx+tb.x-1)/tb.x);
    thom_wall_vorticity_mapped_kernel<<<gb,tb,0,s>>>(w,psi,Nx,Ny,d_dy_edge,psi_bot,psi_top);
    CUDA_CHECK(cudaGetLastError());
}
// Returns J_adv = u·∇ω = J(ω,ψ) = -J(ψ,ω)
// psi, omg: Nx*Ny (row-major), periodic in x; J is undefined at walls (we zero j=0,Ny-1)
// inv_dx = 1/Δx; inv_deta = 1/Δη on the uniform computational η-grid
// a_node[j] = ∂η/∂y at node j  (so ∂y = a * ∂η)
// Returns J_adv = u·∇ω = - (J1 + J2 + J3)/3 with u=(ψ_y,-ψ_x), and ∂y = a · ∂η.
// 9-point mapped Arakawa; returns J_adv = u·∇ω = -(J1+J2+J3)/3
__global__ void arakawa_J_mapped_kernel(const Real* __restrict__ psi,
                                        const Real* __restrict__ omg,
                                        Real* __restrict__ J,
                                        int Nx, int Ny,
                                        Real inv_dx, Real inv_deta,
                                        const Real* __restrict__ a_node)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix>=Nx || iy>=Ny) return;

    size_t c = (size_t)iy*Nx + ix;
    if (iy==0 || iy==Ny-1){ J[c]=(Real)0; return; }

    auto wrap=[Nx](int i){ return (i<0? i+Nx : (i>=Nx? i-Nx : i)); };
    int xm=wrap(ix-1), xp=wrap(ix+1), ym=iy-1, yp=iy+1;

    size_t cxp=(size_t)iy*Nx+xp, cxm=(size_t)iy*Nx+xm;
    size_t cyp=(size_t)yp*Nx+ix, cym=(size_t)ym*Nx+ix;
    size_t ip_jp=(size_t)yp*Nx+xp, ip_jm=(size_t)ym*Nx+xp;
    size_t im_jp=(size_t)yp*Nx+xm, im_jm=(size_t)ym*Nx+xm;

    const Real a        = a_node[iy];                 // ∂η/∂y at row j
    const Real inv4dxde = (Real)0.25 * inv_dx * inv_deta;

    // J1 = (δxψ)(a·δηω) − (a·δηψ)(δxω)
    Real J1 = ( (psi[cxp]-psi[cxm]) * ( a*(omg[cyp]-omg[cym]) )
              - ( a*(psi[cyp]-psi[cym]) ) * (omg[cxp]-omg[cxm]) ) * inv4dxde;

    // J2 = δx(ψ · a·δηω) − δη(ψ · δxω)    [note: the second leg is multiplied by 'a']
    Real J2 = (  psi[cxp] * ( a*(omg[ip_jp]-omg[ip_jm]) )
              -  psi[cxm] * ( a*(omg[im_jp]-omg[im_jm]) )
              -  a * (  psi[cyp] * (omg[ip_jp]-omg[im_jp])
                     - psi[cym] * (omg[ip_jm]-omg[im_jm]) ) ) * inv4dxde;

    // J3 = δx(ω · a·δηψ) − δη(ω · δxψ)    [FIXED: signs match uniform kernel]
    Real J3 = (- omg[cxp] * ( a*(psi[ip_jp]-psi[ip_jm]) )
              + omg[cxm] * ( a*(psi[im_jp]-psi[im_jm]) )
              + a * (  omg[cyp] * (psi[ip_jp]-psi[im_jp])
                     - omg[cym] * (psi[ip_jm]-psi[im_jm]) ) ) * inv4dxde;

    Real Jbr  = (J1 + J2 + J3) * (Real)(1.0/3.0);
    J[c]      = -Jbr;  // u·∇ω
}

inline void launch_arakawa_J_mapped(const Real* psi, const Real* omg, Real* J, int Nx, int Ny,
                                    Real inv_dx, Real inv_deta, const Real* d_a_node, cudaStream_t s=0)
{
    dim3 tb(32,8), gb = blocksFor2D(Nx,Ny,tb.x,tb.y);
    arakawa_J_mapped_kernel<<<gb,tb,0,s>>>(psi,omg,J,Nx,Ny,inv_dx,inv_deta,d_a_node);
    CUDA_CHECK(cudaGetLastError());
}
// ======================================================================
// 1) Direct advective reference from (u,v): J_ref = u·∇ω
//    (uses the SAME mapped first-derivative scales as the solver)
// ======================================================================
__global__ void compute_adv_from_uv_mapped(const Real* __restrict__ u,
                                           const Real* __restrict__ v,
                                           const Real* __restrict__ omg,
                                           Real* __restrict__ Jadv,
                                           int Nx, int Ny,
                                           Real inv_dx, Real inv_deta,
                                           const Real* __restrict__ a_node)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix>=Nx || iy>=Ny) return;

    const size_t c=(size_t)iy*Nx+ix;
    if (iy==0 || iy==Ny-1){ Jadv[c]=(Real)0; return; }

    auto wrap=[Nx](int i){ return (i<0? i+Nx : (i>=Nx? i-Nx : i)); };
    int xm=wrap(ix-1), xp=wrap(ix+1), ym=iy-1, yp=iy+1;

    const size_t cxm=(size_t)iy*Nx+xm, cxp=(size_t)iy*Nx+xp;
    const size_t cym=(size_t)ym*Nx+ix, cyp=(size_t)yp*Nx+ix;

    const Real inv2dx=(Real)0.5*inv_dx, inv2de=(Real)0.5*inv_deta;
    const Real a=a_node[iy];

    const Real dx_omg = (omg[cxp]-omg[cxm]) * inv2dx;
    const Real dy_omg = a * (omg[cyp]-omg[cym]) * inv2de;

    // u = ψ_y, v = -ψ_x (already computed upstream)
    const Real uc = u[c];
    const Real vc = v[c];

    Jadv[c] = uc*dx_omg + vc*dy_omg;     // u·∇ω  (this is what the PDE uses)
}
__global__ void compute_adv_flux_mapped(const Real* __restrict__ u,
                                        const Real* __restrict__ v,
                                        const Real* __restrict__ omg,
                                        Real* __restrict__ Jadv,
                                        int Nx, int Ny,
                                        Real inv_dx, Real inv_deta,
                                        const Real* __restrict__ a_node)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix>=Nx || iy>=Ny) return;

    const size_t c=(size_t)iy*Nx+ix;
    if (iy==0 || iy==Ny-1){ Jadv[c]=(Real)0; return; }

    auto wrap=[Nx](int i){ return (i<0? i+Nx : (i>=Nx? i-Nx : i)); };
    int xm=wrap(ix-1), xp=wrap(ix+1), ym=iy-1, yp=iy+1;

    size_t cxm=(size_t)iy*Nx+xm, cxp=(size_t)iy*Nx+xp;
    size_t cym=(size_t)ym*Nx+ix, cyp=(size_t)yp*Nx+ix;

    // x–flux at faces i±1/2: Fx = (u_avg)*(ω_avg)
    Real u_ip = (u[cxp] + u[c]) * (Real)0.5;
    Real u_im = (u[c]   + u[cxm]) * (Real)0.5;
    Real w_ip = (omg[cxp] + omg[c]) * (Real)0.5;
    Real w_im = (omg[c]   + omg[cxm]) * (Real)0.5;
    Real dFdx = (u_ip*w_ip - u_im*w_im) * inv_dx;

    // y–flux at faces j±1/2: Fy = (v_avg)*(ω_avg), then ∂y Fy = a * ∂η Fy
    Real v_jp = (v[cyp] + v[c]) * (Real)0.5;
    Real v_jm = (v[c]   + v[cym]) * (Real)0.5;
    Real w_jp = (omg[cyp] + omg[c]) * (Real)0.5;
    Real w_jm = (omg[c]   + omg[cym]) * (Real)0.5;
    Real dFdy = a_node[iy] * ( (v_jp*w_jp - v_jm*w_jm) * ( (Real)0.5*inv_deta*2.0 ) );
    // the last factor simplifies: 0.5*inv_deta*2.0 = inv_deta; written expanded to mirror ∂y=a·∂η.

    Jadv[c] = dFdx + dFdy;                          // conservative u·∇ω
}
// ======================================================================
// 2) Mapped Arakawa A/B/C kernel (ADVECTIVE form), *split* outputs:
//    writes J1, J2, J3 individually and Jadv = u·∇ω = - (J1+J2+J3)/3.
//    Every y-derivative carries the metric a_j = ∂η/∂y.
// ======================================================================
__global__ void arakawa_J_mapped_split_kernel(const Real* __restrict__ psi,
                                              const Real* __restrict__ omg,
                                              Real* __restrict__ J1o,
                                              Real* __restrict__ J2o,
                                              Real* __restrict__ J3o,
                                              Real* __restrict__ Jadv,
                                              int Nx, int Ny,
                                              Real inv_dx, Real inv_deta,
                                              const Real* __restrict__ a_node)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix>=Nx || iy>=Ny) return;

    size_t c = (size_t)iy*Nx + ix;
    if (iy==0 || iy==Ny-1){
        if (J1o)  J1o[c]=(Real)0; if (J2o)  J2o[c]=(Real)0;
        if (J3o)  J3o[c]=(Real)0; if (Jadv) Jadv[c]=(Real)0;
        return;
    }

    auto wrap=[Nx](int i){ return (i<0? i+Nx : (i>=Nx? i-Nx : i)); };
    int xm=wrap(ix-1), xp=wrap(ix+1), ym=iy-1, yp=iy+1;

    size_t cxp=(size_t)iy*Nx+xp, cxm=(size_t)iy*Nx+xm;
    size_t cyp=(size_t)yp*Nx+ix, cym=(size_t)ym*Nx+ix;
    size_t ip_jp=(size_t)yp*Nx+xp, ip_jm=(size_t)ym*Nx+xp;
    size_t im_jp=(size_t)yp*Nx+xm, im_jm=(size_t)ym*Nx+xm;

    const Real a        = a_node[iy];
    const Real inv4dxde = (Real)0.25 * inv_dx * inv_deta;

    Real J1 = ( (psi[cxp]-psi[cxm]) * ( a*(omg[cyp]-omg[cym]) )
              - ( a*(psi[cyp]-psi[cym]) ) * (omg[cxp]-omg[cxm]) ) * inv4dxde;

    Real J2 = (  psi[cxp] * ( a*(omg[ip_jp]-omg[ip_jm]) )
              -  psi[cxm] * ( a*(omg[im_jp]-omg[im_jm]) )
              -  a * (  psi[cyp] * (omg[ip_jp]-omg[im_jp])
                     - psi[cym] * (omg[ip_jm]-omg[im_jm]) ) ) * inv4dxde;

    // J3 [FIXED: signs match uniform kernel]
    Real J3 = ( -omg[cxp] * ( a*(psi[ip_jp]-psi[ip_jm]) )
              + omg[cxm] * ( a*(psi[im_jp]-psi[im_jm]) )
              +  a * (  omg[cyp] * (psi[ip_jp]-psi[im_jp])
                     - omg[cym] * (psi[ip_jm]-psi[im_jm]) ) ) * inv4dxde;

    if (J1o)  J1o[c]=J1;
    if (J2o)  J2o[c]=J2;
    if (J3o)  J3o[c]=J3;

    Real Jbr = (J1 + J2 + J3) * (Real)(1.0/3.0);
    if (Jadv) Jadv[c] = -Jbr; // u·∇ω
}

__global__ void enstrophy_diss_mapped_kernel(const Real* __restrict__ omg,
    int Nx, int Ny, Real inv_dx, Real inv_deta,
    const Real* __restrict__ a_node,
    const Real* __restrict__ w_node,
    Real* __restrict__ sum_w2, Real* __restrict__ sum_gw2)
{
    __shared__ Real sbw[256]; __shared__ Real sbg[256];
    size_t tid=threadIdx.x; Real lw2=0.0, lg2=0.0; size_t N=(size_t)Nx*Ny;
    for (size_t idx=blockIdx.x*blockDim.x + tid; idx<N; idx+=gridDim.x*blockDim.x){
        int ix=idx % Nx, iy=idx / Nx; if (iy==0 || iy==Ny-1) continue;
        int xm=(ix==0? Nx-1:ix-1), xp=(ix==Nx-1? 0:ix+1);
        int ym=iy-1, yp=iy+1;
        Real w = omg[idx];
        Real wx = (omg[(size_t)iy*Nx + xp] - omg[(size_t)iy*Nx + xm]) * 0.5 * inv_dx;
        Real wy = a_node[iy] * (omg[(size_t)yp*Nx + ix] - omg[(size_t)ym*Nx + ix]) * 0.5 * inv_deta;
        Real wt = w_node[iy];
        lw2 += wt * (w*w);
        lg2 += wt * (wx*wx + wy*wy);
    }
    sbw[tid]=lw2; sbg[tid]=lg2; __syncthreads();
    for (unsigned s=blockDim.x/2; s>0; s>>=1){ if (tid<s){ sbw[tid]+=sbw[tid+s]; sbg[tid]+=sbg[tid+s]; } __syncthreads(); }
    if (tid==0){ atomicAdd(sum_w2,sbw[0]); atomicAdd(sum_gw2,sbg[0]); }
}
inline void launch_enstrophy_diss_mapped(const Real* omg, int Nx, int Ny, Real inv_dx, Real inv_deta,
                                         const Real* d_a_node, const Real* d_w_node,
                                         Real* d_sum_w2, Real* d_sum_gw2, cudaStream_t s=0)
{
    int threads=256; int blocks=(int)std::min<size_t>(8192, ((size_t)Nx*Ny + threads - 1)/threads);
    CUDA_CHECK(cudaMemsetAsync(d_sum_w2,0,sizeof(Real),s));
    CUDA_CHECK(cudaMemsetAsync(d_sum_gw2,0,sizeof(Real),s));
    enstrophy_diss_mapped_kernel<<<blocks,threads,0,s>>>(omg,Nx,Ny,inv_dx,inv_deta,d_a_node,d_w_node,d_sum_w2,d_sum_gw2);
    CUDA_CHECK(cudaGetLastError());
}
// Full-domain enstrophy and palinstrophy (includes wall rows).
// Uses centered differences in the interior and 2nd-order one-sided in η at the walls.
// ω_x: centered periodic in x everywhere.
// ω_y: ω_y = a_node[j] * ω_η;  ω_η is one-sided at j=0,Ny-1 and centered otherwise.
__global__ void enstrophy_diss_mapped_full_kernel(const Real* __restrict__ omg,
    int Nx, int Ny, Real inv_dx, Real inv_deta,
    const Real* __restrict__ a_node,
    const Real* __restrict__ w_node,
    Real* __restrict__ sum_w2_full, Real* __restrict__ sum_gw2_full)
{
    __shared__ Real sbw[256]; __shared__ Real sbg[256];
    size_t tid=threadIdx.x; Real lw2=0.0, lg2=0.0; size_t N=(size_t)Nx*Ny;
    for (size_t idx=blockIdx.x*blockDim.x + tid; idx<N; idx+=gridDim.x*blockDim.x){
        int ix=idx % Nx, iy=(int)(idx / Nx);
        int xm=(ix==0? Nx-1:ix-1), xp=(ix==Nx-1? 0:ix+1);

        const Real w = omg[idx];

        // ω_x: centered periodic
        const Real wx = (omg[(size_t)iy*Nx + xp] - omg[(size_t)iy*Nx + xm]) * (Real)0.5 * inv_dx;

        // ω_y in physical y: ω_y = a(j) * ω_η
        Real weta;
        if (iy == 0){
            // 2nd-order one-sided in η at bottom wall (uses j=0,1,2)
            const Real w0 = omg[(size_t)0*Nx + ix];
            const Real w1 = omg[(size_t)1*Nx + ix];
            const Real w2 = omg[(size_t)2*Nx + ix];
            weta = (-3.0*w0 + 4.0*w1 - 1.0*w2) * ((Real)0.5 * inv_deta);
        } else if (iy == Ny-1){
            // 2nd-order one-sided in η at top wall (uses j=Ny-1,Ny-2,Ny-3)
            const Real wN   = omg[(size_t)(Ny-1)*Nx + ix];
            const Real wNm1 = omg[(size_t)(Ny-2)*Nx + ix];
            const Real wNm2 = omg[(size_t)(Ny-3)*Nx + ix];
            weta = ( 3.0*wN - 4.0*wNm1 + 1.0*wNm2) * ((Real)0.5 * inv_deta);
        } else {
            int ym=iy-1, yp=iy+1;
            weta = (omg[(size_t)yp*Nx + ix] - omg[(size_t)ym*Nx + ix]) * (Real)0.5 * inv_deta;
        }
        const Real wy = a_node[iy] * weta;

        const Real wt = w_node[iy];
        lw2 += wt * (w*w);
        lg2 += wt * (wx*wx + wy*wy);
    }
    sbw[tid]=lw2; sbg[tid]=lg2; __syncthreads();
    for (unsigned s=blockDim.x/2; s>0; s>>=1){
        if (tid<s){ sbw[tid]+=sbw[tid+s]; sbg[tid]+=sbg[tid+s]; }
        __syncthreads();
    }
    if (tid==0){ atomicAdd(sum_w2_full,sbw[0]); atomicAdd(sum_gw2_full,sbg[0]); }
}

inline void launch_enstrophy_diss_mapped_full(const Real* omg, int Nx, int Ny, Real inv_dx, Real inv_deta,
                                              const Real* d_a_node, const Real* d_w_node,
                                              Real* d_sum_w2_full, Real* d_sum_gw2_full, cudaStream_t s=0)
{
    int threads=256; int blocks=(int)std::min<size_t>(8192, ((size_t)Nx*Ny + threads - 1)/threads);
    CUDA_CHECK(cudaMemsetAsync(d_sum_w2_full, 0, sizeof(Real), s));
    CUDA_CHECK(cudaMemsetAsync(d_sum_gw2_full, 0, sizeof(Real), s));
    enstrophy_diss_mapped_full_kernel<<<blocks,threads,0,s>>>(omg,Nx,Ny,inv_dx,inv_deta,d_a_node,d_w_node,d_sum_w2_full,d_sum_gw2_full);
    CUDA_CHECK(cudaGetLastError());
}
// Enstrophy dissipation using -ω Δω (wall-balanced, works for no-slip or free-slip)
__global__ void enstrophy_diss_balance_mapped_kernel(
    const Real* __restrict__ omega,      // ω
    const Real* __restrict__ lap_omega,  // ∇²ω with the same stencil as RHS
    int Nx, int Ny,
    const Real* __restrict__ w_node,     // quadrature weights in y
    Real* __restrict__ sum_w2,           // accumulator for ∑ w_node ω²
    Real* __restrict__ sum_eta)          // accumulator for ∑ w_node (-ω Δω)
{
    __shared__ Real sbw[256];
    __shared__ Real sbe[256];

    const int    tid = threadIdx.x;
    const size_t N   = (size_t)Nx * (size_t)Ny;

    Real lw2  = 0.0;
    Real leta = 0.0;

    // grid‑stride loop over all nodes
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + tid;
         idx < N;
         idx += (size_t)gridDim.x * blockDim.x)
    {
        int j = (int)(idx / Nx);          // y-index

        // match Ω definition: interior only (j=1..Ny-2)
        if (j == 0 || j == Ny - 1) continue;

        Real w   = omega[idx];
        Real lap = lap_omega[idx];
        Real wt  = w_node[j];

        lw2  += wt * (w * w);
        // + sign: η = ⟨ω Δω⟩
        leta += wt * (w * lap);
    }

    sbw[tid] = lw2;
    sbe[tid] = leta;
    __syncthreads();

    // block reduction
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sbw[tid] += sbw[tid + s];
            sbe[tid] += sbe[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum_w2,  sbw[0]);
        atomicAdd(sum_eta, sbe[0]);
    }
}

__global__ void kinetic_energy_mapped_kernel(const Real* __restrict__ psi,
    int Nx, int Ny, Real inv_dx, Real inv_deta, const Real* __restrict__ a_node,
    const Real* __restrict__ w_node, Real* __restrict__ out_sum)
{
    __shared__ Real sb[256];
    size_t tid=threadIdx.x; Real sum=0.0; size_t N=(size_t)Nx*Ny;
    for (size_t idx=blockIdx.x*blockDim.x + tid; idx<N; idx+=gridDim.x*blockDim.x){
        int ix=idx % Nx, iy=idx / Nx;
        // Skip wall rows 
        if (iy == 0 || iy == Ny-1) continue;

        int xm=(ix==0? Nx-1:ix-1), xp=(ix==Nx-1? 0:ix+1);
       	//int ym=(iy==0? 0:iy-1), yp=(iy==Ny-1? Ny-1:iy+1);
        int ym = iy-1, yp = iy + 1;
	Real u=(psi[(size_t)yp*Nx + ix]-psi[(size_t)ym*Nx + ix]) * 0.5 * inv_deta * a_node[iy];
        Real v=-(psi[(size_t)iy*Nx + xp]-psi[(size_t)iy*Nx + xm]) * 0.5 * inv_dx;
	
        sum += 0.5*(u*u + v*v) * w_node[iy];
    }
    sb[tid]=sum; __syncthreads();
    for (unsigned s=blockDim.x/2; s>0; s>>=1){ if (tid<s) sb[tid]+=sb[tid+s]; __syncthreads(); }
    if (tid==0) atomicAdd(out_sum,sb[0]);
}
inline void launch_kinetic_energy_mapped(const Real* psi, int Nx, int Ny, Real inv_dx, Real inv_deta,
                                         const Real* d_a_node, const Real* d_w_node, Real* d_sum, cudaStream_t s=0)
{
    int threads=256; int blocks=(int)std::min<size_t>(8192, ((size_t)Nx*Ny + threads - 1)/threads);
    CUDA_CHECK(cudaMemsetAsync(d_sum,0,sizeof(Real),s));
    kinetic_energy_mapped_kernel<<<blocks,threads,0,s>>>(psi,Nx,Ny,inv_dx,inv_deta,d_a_node,d_w_node,d_sum);
    CUDA_CHECK(cudaGetLastError());
}
// Index helper
#ifndef IDX
#define IDX(j,i) ((size_t)(j) * (size_t)Nx + (size_t)(i))
#endif
// energy_dissipation_mapped_kernel:
// Computes sum_yx w_node[j] * ( |∇u|^2 + |∇v|^2 ), where
//   u = ψ_y = a(j) * ψ_η,  v = -ψ_x
// with:
//   u_x = a(j) * ∂η(ψ_x)          (cross derivative, j±1 only)
//   v_y = -u_x                    (same cross stencil → discrete incompressibility)
//   v_x = -ψ_xx                   (true second derivative in x)
//   u_y = a(j)*[ a_{j+1/2}(ψ_{j+1}-ψ_j) - a_{j-1/2}(ψ_j-ψ_{j-1}) ] / Δη²   (fully conservative)
//
// Notes:
// - Skips wall rows j=0 and j=Ny-1 (no-slip). If you prefer to include walls,
//   you can add one-sided variants, though contribution is tiny there.
// - Uses dynamic shared memory sized to blockDim.x for reduction.
// - Requires: a_node[j] (∂η/∂y at nodes), a_edge[j] (∂η/∂y at j+1/2), w_node[j] (∼dy trapezoid).
__global__ void energy_dissipation_mapped_kernel(
    const Real* __restrict__ psi,     // [Ny*Nx]
    const Real* __restrict__ omega,   // [Ny*Nx] (needed for wall contribution via u_y|wall = -ω_wall)
    int Nx, int Ny,
    Real inv_dx, Real inv_deta,
    const Real* __restrict__ a_node,  // [Ny]
    const Real* __restrict__ a_edge,  // [Ny-1], where a_edge[j] = a at (j+1/2)
    const Real* __restrict__ w_node,  // [Ny] trapezoid-like y weights(~dy)
    Real* __restrict__ sum_eps)       // scalar accumulator (device)
{
    extern __shared__ Real sb[];      // sb[blockDim.x]
    const int tid = threadIdx.x;
    const size_t N = (size_t)Nx * (size_t)Ny;

    Real loc = 0.0;
    const Real inv_dx2   = inv_dx   * inv_dx;
    const Real inv_deta2 = inv_deta * inv_deta;

    // Grid-stride loop
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + tid;
         idx < N;
         idx += (size_t)gridDim.x * blockDim.x)
    {
        const int j = (int)(idx / Nx);
        const int i = (int)(idx - (size_t)j * Nx);

        // Include wall rows in the quadrature using no-slip kinematics:
        // u=v=0 on y=±h ⇒ u_x=v_x=0 and by incompressibility v_y=0, so |∇u|^2 reduces to (u_y)^2.
        // With ω = v_x - u_y and v_x|wall=0 ⇒ u_y|wall = -ω_wall.
        if (j==0 || j==Ny-1){
            const Real uy = -omega[idx];
            loc += (uy*uy) * w_node[j];
            continue;
        }       
        const int im = (i == 0     ? Nx - 1 : i - 1);
        const int ip = (i == Nx-1  ? 0      : i + 1);
        // Neighbor linear indices
        const size_t c     = IDX(j, i);
        const size_t c_ip  = IDX(j, ip);
        const size_t c_im  = IDX(j, im);
        const size_t jp_i  = IDX(j+1, i);
        const size_t jm_i  = IDX(j-1, i);
        const size_t jp_ip = IDX(j+1, ip);
        const size_t jp_im = IDX(j+1, im);
        const size_t jm_ip = IDX(j-1, ip);
        const size_t jm_im = IDX(j-1, im);
        
        // Mapping metrics
        const Real a  = a_node[j];
        const Real ap = a_edge[j];     // j+1/2
        const Real am = a_edge[j-1];   // j-1/2

        // ψ_x at j±1 (periodic in x)
        const Real psi_x_jp = (psi[jp_ip] - psi[jp_im]) * (0.5 * inv_dx);
        const Real psi_x_jm = (psi[jm_ip] - psi[jm_im]) * (0.5 * inv_dx);

        // Cross derivatives: u_x = a * ∂η(ψ_x),  v_y = -u_x
        const Real ux = a * (psi_x_jp - psi_x_jm) * (0.5 * inv_deta);
        const Real vy = -ux;

        // v_x = -ψ_xx (true second derivative in x)
        const Real vx = -(psi[c_ip] - 2.0*psi[c] + psi[c_im]) * inv_dx2;

        // u_y = a * [ ap*(ψ_{j+1}-ψ_j) - am*(ψ_j-ψ_{j-1}) ] / Δη²   (fully conservative)
        const Real uy = a * ( ap * (psi[jp_i] - psi[c]) - am * (psi[c] - psi[jm_i]) ) * inv_deta2;

        // Accumulate contribution with y-quadrature weight
        const Real g2 = ux*ux + uy*uy + vx*vx + vy*vy;
        loc += g2 * w_node[j];
    }

    // block reduction (sum)
    sb[tid] = loc;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sb[tid] += sb[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(sum_eps, sb[0]);
}
inline void launch_energy_dissipation_mapped(
    const Real* d_psi, const Real* d_omega, int Nx, int Ny,
    Real inv_dx, Real inv_deta,
    const Real* d_a_node,  // [Ny]
    const Real* d_a_edge,  // [Ny-1]
    const Real* d_w_node,  // [Ny]
    Real* d_sum_eps,       // device scalar
    cudaStream_t s)
{
    const int threads = 256;
    const size_t N    = (size_t)Nx * (size_t)Ny;
    const int blocks  = (int)std::min<size_t>(8192, (N + threads - 1) / threads);
    const size_t shmem = threads * sizeof(Real);

    CUDA_CHECK(cudaMemsetAsync(d_sum_eps, 0, sizeof(Real), s));
    energy_dissipation_mapped_kernel<<<blocks, threads, shmem, s>>>(
        d_psi, d_omega, Nx, Ny, inv_dx, inv_deta, d_a_node, d_a_edge, d_w_node, d_sum_eps);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void power_input_mapped_kernel(const Real* __restrict__ psi,
    int Nx, int Ny, Real inv_deta, Real h, Real F0, int nforce,
    const Real* __restrict__ a_node, const Real* __restrict__ y_node,
    const Real* __restrict__ w_node, Real* __restrict__ sum)
{
    __shared__ Real sb[256]; size_t tid=threadIdx.x; Real loc=0.0; size_t N=(size_t)Nx*Ny;
    for (size_t idx=blockIdx.x*blockDim.x + tid; idx<N; idx+=gridDim.x*blockDim.x){
        int ix=idx % Nx, iy=idx / Nx;
        int ym=(iy==0? 0:iy-1), yp=(iy==Ny-1? Ny-1:iy+1);
        Real u = (psi[(size_t)yp*Nx + ix]-psi[(size_t)ym*Nx + ix]) * 0.5 * inv_deta * a_node[iy];
        Real y = y_node[iy];
        Real Fx = F0 * sin((Real)nforce * CUDART_PI * ((y + h)/(2.0*h)));
        loc += u*Fx * w_node[iy];
    }
    sb[tid]=loc; __syncthreads();
    for (unsigned s=blockDim.x/2; s>0; s>>=1){ if (tid<s) sb[tid]+=sb[tid+s]; __syncthreads(); }
    if (tid==0) atomicAdd(sum,sb[0]);
}
inline void launch_power_input_mapped(const Real* psi, int Nx, int Ny, Real inv_deta, Real h, Real F0, int nforce,
                                      const Real* d_a_node, const Real* d_y_node, const Real* d_w_node,
                                      Real* d_sum, cudaStream_t s=0)
{
    int threads=256; int blocks=(int)std::min<size_t>(8192, ((size_t)Nx*Ny + threads - 1)/threads);
    CUDA_CHECK(cudaMemsetAsync(d_sum,0,sizeof(Real),s));
    power_input_mapped_kernel<<<blocks,threads,0,s>>>(psi,Nx,Ny,inv_deta,h,F0,nforce,d_a_node,d_y_node,d_w_node,d_sum);
    CUDA_CHECK(cudaGetLastError());
}
// Full-domain enstrophy input P_Ω = ⟨ ω f_ω ⟩_A (includes wall rows for consistent area quadrature).
__global__ void enstrophy_input_mapped_full_kernel(
    const Real* __restrict__ omega,
    int Nx, int Ny, Real h, Real F0, int nforce,
    const Real* __restrict__ y_node,
    const Real* __restrict__ w_node,
    Real* __restrict__ sum)
{
    __shared__ Real sb[256];
    size_t tid = threadIdx.x;
    Real loc = 0.0;
    size_t N = (size_t)Nx * (size_t)Ny;

    for (size_t idx = blockIdx.x*blockDim.x + tid; idx < N; idx += gridDim.x*blockDim.x){
        int iy = (int)(idx / Nx);
        const Real w = omega[idx];
        const Real y = y_node[iy];

        // f_ω = -∂F_x/∂y where F_x = F0 sin(nπ(y+h)/(2h))
        const Real arg = (Real)nforce * CUDART_PI * ((y + h)/(2.0*h));
        const Real f_omega = -F0 * ((Real)nforce*CUDART_PI/(2.0*h)) * cos(arg);

        loc += w * f_omega * w_node[iy];
    }

    sb[tid] = loc;
    __syncthreads();
    for (unsigned ssz=blockDim.x/2; ssz>0; ssz>>=1){
        if (tid<ssz) sb[tid] += sb[tid+ssz];
        __syncthreads();
    }
    if (tid==0) atomicAdd(sum, sb[0]);
}

inline void launch_enstrophy_input_mapped_full(
    const Real* omega, int Nx, int Ny, Real h, Real F0, int nforce,
    const Real* d_y_node, const Real* d_w_node,
    Real* d_sum, cudaStream_t s=0)
{
    int threads=256;
    int blocks=(int)std::min<size_t>(8192, ((size_t)Nx*Ny + threads - 1)/threads);
    CUDA_CHECK(cudaMemsetAsync(d_sum, 0, sizeof(Real), s));
    enstrophy_input_mapped_full_kernel<<<blocks,threads,0,s>>>(
        omega, Nx, Ny, h, F0, nforce, d_y_node, d_w_node, d_sum);
    CUDA_CHECK(cudaGetLastError());
}
__global__ void enstrophy_input_mapped_kernel(
    const Real* __restrict__ omega, 
    int Nx, int Ny, Real h, Real F0, int nforce,
    const Real* __restrict__ y_node,
    const Real* __restrict__ w_node, 
    Real* __restrict__ sum)
{
    __shared__ Real sb[256]; 
    size_t tid=threadIdx.x; 
    Real loc=0.0; 
    size_t N=(size_t)Nx*Ny;
    
    for (size_t idx=blockIdx.x*blockDim.x + tid; idx<N; idx+=gridDim.x*blockDim.x){
	int iy=idx / Nx;
	if (iy==0 || iy==Ny-1) continue;  // interior only, matches Ω/η	
        Real w = omega[idx];
        Real y = y_node[iy];
        
        // f_ω = -∂F_x/∂y where F_x = F0 sin(nπ(y+h)/(2h))
        // f_ω = -F0 * (nπ/(2h)) * cos(nπ(y+h)/(2h))
        Real arg = (Real)nforce * CUDART_PI * ((y + h)/(2.0*h));
        Real f_omega = -F0 * ((Real)nforce*CUDART_PI/(2.0*h)) * cos(arg);
        loc += w * f_omega * w_node[iy];
    }
    
    sb[tid]=loc; 
    __syncthreads();
    for (unsigned s=blockDim.x/2; s>0; s>>=1){ 
        if (tid<s) sb[tid]+=sb[tid+s]; 
        __syncthreads(); 
    }
    if (tid==0) atomicAdd(sum, sb[0]);
}
inline void launch_enstrophy_input_mapped(
    const Real* omega, int Nx, int Ny, Real h, Real F0, int nforce,
    const Real* d_y_node, const Real* d_w_node,
    Real* d_sum, cudaStream_t s=0)
{
    int threads=256; 
    int blocks=(int)std::min<size_t>(8192, ((size_t)Nx*Ny + threads - 1)/threads);
    CUDA_CHECK(cudaMemsetAsync(d_sum,0,sizeof(Real),s));
    enstrophy_input_mapped_kernel<<<blocks,threads,0,s>>>(
        omega,Nx,Ny,h,F0,nforce,d_y_node,d_w_node,d_sum);
    CUDA_CHECK(cudaGetLastError());
}
__global__ void row_moments_from_psi_mapped_kernel(
    const Real* __restrict__ psi, int Nx, int Ny, Real inv_dx, Real inv_deta,
    const Real* __restrict__ a_node,
    Real* __restrict__ sum_u, Real* __restrict__ sum_v, Real* __restrict__ sum_u2, Real* __restrict__ sum_v2, Real* __restrict__ sum_uv)
{
    int iy = blockIdx.x;
    if (iy >= Ny) return;
    extern __shared__ Real s[];
    Real* su=s; Real* sv=s+blockDim.x; Real* su2=s+2*blockDim.x; Real* sv2=s+3*blockDim.x; Real* suv=s+4*blockDim.x;
    size_t tid=threadIdx.x;
    Real lu=0.0, lv=0.0, lu2=0.0, lv2=0.0, luv=0.0;
    for (int ix=threadIdx.x; ix<Nx; ix+=blockDim.x){
        int xm=(ix==0? Nx-1:ix-1), xp=(ix==Nx-1? 0:ix+1);
        int ym=(iy==0? 0:iy-1), yp=(iy==Ny-1? Ny-1:iy+1);
        Real u = (psi[(size_t)yp*Nx + ix] - psi[(size_t)ym*Nx + ix]) * 0.5 * inv_deta * a_node[iy];
        Real v = -(psi[(size_t)iy*Nx + xp]-psi[(size_t)iy*Nx + xm]) * 0.5 * inv_dx;
        lu+=u; lv+=v; lu2+=u*u; lv2+=v*v; luv+=u*v;
    }
    su[tid]=lu; sv[tid]=lv; su2[tid]=lu2; sv2[tid]=lv2; suv[tid]=luv; __syncthreads();
    for (unsigned ssz=blockDim.x/2; ssz>0; ssz>>=1){
        if (tid<ssz){ su[tid]+=su[tid+ssz]; sv[tid]+=sv[tid+ssz]; su2[tid]+=su2[tid+ssz]; sv2[tid]+=sv2[tid+ssz]; suv[tid]+=suv[tid+ssz]; }
        __syncthreads();
    }
    if (tid==0){ sum_u[iy]=su[0]; sum_v[iy]=sv[0]; sum_u2[iy]=su2[0]; sum_v2[iy]=sv2[0]; sum_uv[iy]=suv[0]; }
}
inline void launch_row_moments_from_psi_mapped(const Real* psi, int Nx, int Ny, Real inv_dx, Real inv_deta,
                                               const Real* d_a_node, Real* su, Real* sv, Real* su2, Real* sv2, Real* suv, cudaStream_t s=0)
{
    int threads=256; dim3 gb(Ny); size_t sh=threads*sizeof(Real)*5;
    row_moments_from_psi_mapped_kernel<<<gb,threads,sh,s>>>(psi,Nx,Ny,inv_dx,inv_deta,d_a_node,su,sv,su2,sv2,suv);
    CUDA_CHECK(cudaGetLastError());
}
__global__ void mean_wall_omega_kernel(const Real* __restrict__ omega, int Nx, int Ny, Real* __restrict__ mean_bot, Real* __restrict__ mean_top){
    extern __shared__ Real s[]; Real* sb=s; Real* st=s+blockDim.x;
    size_t tid=threadIdx.x; Real lb=0.0, lt=0.0;
    for (int ix=tid; ix<Nx; ix+=blockDim.x){ lb+=omega[ix]; lt+=omega[(size_t)(Ny-1)*Nx + ix]; }
    sb[tid]=lb; st[tid]=lt; __syncthreads();
    for (unsigned ssz=blockDim.x/2; ssz>0; ssz>>=1){ if (tid<ssz){ sb[tid]+=sb[tid+ssz]; st[tid]+=st[tid+ssz]; } __syncthreads(); }
    if (tid==0){ *mean_bot = sb[0]/(Real)Nx; *mean_top = st[0]/(Real)Nx; }
}
inline void launch_mean_wall_omega(const Real* omega, int Nx, int Ny, Real* d_mean_bot, Real* d_mean_top, cudaStream_t s=0){
    int threads=256; size_t sh=threads*sizeof(Real)*2;
    mean_wall_omega_kernel<<<1,threads,sh,s>>>(omega,Nx,Ny,d_mean_bot,d_mean_top);
    CUDA_CHECK(cudaGetLastError());
}
// Mean wall enstrophy-flux integrand ⟨ ω ω_y ⟩_x at bottom/top walls (mapped grid).
// ω_y is computed using 2nd-order one-sided differences in η and the metric: ω_y = a_node[j] * ω_η.
// Outputs mean_bot = (1/Nx) Σ_i ω_bot(i) * ω_y_bot(i), and similarly for mean_top.
__global__ void mean_wall_omega_omegay_mapped_kernel(
    const Real* __restrict__ omega,
    int Nx, int Ny,
    Real inv_deta,
    const Real* __restrict__ a_node,
    Real* __restrict__ mean_bot,
    Real* __restrict__ mean_top)
{
    extern __shared__ Real s[];
    Real* sb = s;
    Real* st = s + blockDim.x;

    const int tid = threadIdx.x;
    Real lb = 0.0;
    Real lt = 0.0;

    for (int ix = tid; ix < Nx; ix += blockDim.x){
        // bottom wall j=0 : ω_η ≈ (-3ω0 + 4ω1 - ω2)/(2Δη)
        const Real w0 = omega[(size_t)0*Nx + ix];
        const Real w1 = omega[(size_t)1*Nx + ix];
        const Real w2 = omega[(size_t)2*Nx + ix];
        const Real weta_bot = (-3.0*w0 + 4.0*w1 - 1.0*w2) * ((Real)0.5 * inv_deta);
        const Real wy_bot   = a_node[0] * weta_bot;
        lb += w0 * wy_bot;

        // top wall j=Ny-1 : ω_η ≈ (3ωN - 4ωN-1 + ωN-2)/(2Δη)
        const size_t jN = (size_t)(Ny-1);
        const size_t j1 = (size_t)(Ny-2);
        const size_t j2 = (size_t)(Ny-3);
        const Real wN   = omega[jN*(size_t)Nx + ix];
        const Real wNm1 = omega[j1*(size_t)Nx + ix];
        const Real wNm2 = omega[j2*(size_t)Nx + ix];
        const Real weta_top = ( 3.0*wN - 4.0*wNm1 + 1.0*wNm2) * ((Real)0.5 * inv_deta);
        const Real wy_top   = a_node[Ny-1] * weta_top;
        lt += wN * wy_top;
    }

    sb[tid] = lb;
    st[tid] = lt;
    __syncthreads();

    for (unsigned ssz = blockDim.x/2; ssz > 0; ssz >>= 1){
        if (tid < ssz){
            sb[tid] += sb[tid + ssz];
            st[tid] += st[tid + ssz];
        }
        __syncthreads();
    }

    if (tid == 0){
        *mean_bot = sb[0] / (Real)Nx;
        *mean_top = st[0] / (Real)Nx;
    }
}

inline void launch_mean_wall_omega_omegay_mapped(
    const Real* omega, int Nx, int Ny,
    Real inv_deta,
    const Real* d_a_node,
    Real* d_mean_bot, Real* d_mean_top,
    cudaStream_t s=0)
{
    int threads = 256;
    size_t sh   = (size_t)threads * sizeof(Real) * 2;
    mean_wall_omega_omegay_mapped_kernel<<<1, threads, sh, s>>>(
        omega, Nx, Ny, inv_deta, d_a_node, d_mean_bot, d_mean_top);
    CUDA_CHECK(cudaGetLastError());
}
// ============= ADD THIS NEW KERNEL =============
// q_lap = ∇² q  (mapped y, periodic x). Writes interior j=1..Ny-2; sets walls = 0.
__global__ void laplacian_fd_mapped_kernel(const Real* __restrict__ q,
                                           Real* __restrict__ q_lap,
                                           int Nx, int Ny,
                                           Real inv_dx2, Real inv_deta2,
                                           const Real* __restrict__ a_node,  // Ny
                                           const Real* __restrict__ a_edge)  // Ny-1
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix>=Nx || iy>=Ny) return;

    size_t id = (size_t)iy*Nx + ix;
    if (iy==0 || iy==Ny-1){ q_lap[id] = (Real)0; return; }

    auto wrap=[Nx](int i){ return (i<0? i+Nx : (i>=Nx? i-Nx : i)); };
    int xm = wrap(ix-1), xp = wrap(ix+1);

    // δxx (uniform in x)
    Real dxx = ( q[(size_t)iy*Nx+xp] - 2*q[id] + q[(size_t)iy*Nx+xm] ) * inv_dx2;

    // mapped δyy = a_j * [ a_{j+1/2}(q_{j+1}-q_j) - a_{j-1/2}(q_j-q_{j-1}) ] / Δη²
    Real a   = a_node[iy];
    Real ap  = a_edge[iy];     // j+1/2
    Real am  = a_edge[iy-1];   // j-1/2
    Real dyy = a * ( ap*( q[(size_t)(iy+1)*Nx + ix] - q[id] )
                   - am*( q[id] - q[(size_t)(iy-1)*Nx + ix] ) ) * inv_deta2;

    q_lap[id] = dxx + dyy;
}

inline void launch_laplacian_fd_mapped(const Real* d_q, Real* d_out,
                                       int Nx, int Ny,
                                       Real inv_dx2, Real inv_deta2,
                                       const Real* d_a_node, const Real* d_a_edge,
                                       cudaStream_t s=0)
{
    dim3 tb(32,4), gb((Nx+31)/32, (Ny+3)/4);
    laplacian_fd_mapped_kernel<<<gb,tb,0,s>>>(d_q, d_out, Nx, Ny,
                                              inv_dx2, inv_deta2,
                                              d_a_node, d_a_edge);
    CUDA_CHECK(cudaGetLastError());
}
// Wall‑balanced enstrophy dissipation using -ω Δω
inline void launch_enstrophy_diss_balance_mapped(
    const Real* d_omega,        // [Ny*Nx] vorticity field
    int Nx, int Ny,
    Real inv_dx2, Real inv_deta2,
    const Real* d_a_node,       // mapping metrics
    const Real* d_a_edge,
    const Real* d_w_node,       // quadrature weights in y
    Real* d_sum_w2,             // device scalar
    Real* d_sum_eta,            // device scalar
    cudaStream_t s = 0)
{
    const size_t N = (size_t)Nx * (size_t)Ny;

    // Lazily allocated scratch for ∇²ω
    static Real*  d_lap     = nullptr;
    static size_t capacity  = 0;
    if (N > capacity) {
        if (d_lap) CUDA_CHECK(cudaFree(d_lap));
        CUDA_CHECK(cudaMalloc(&d_lap, N * sizeof(Real)));
        capacity = N;
    }

    // ∇²ω with the *same* stencil as the PDE (includes mapped y & walls)
    launch_laplacian_fd_mapped(
        d_omega, d_lap, Nx, Ny,
        inv_dx2, inv_deta2,
        d_a_node, d_a_edge, s);

    // Reduction of Ω and η = ν⟨ω Δω⟩
    const int threads = 256;
    const int blocks  =
        (int)std::min<size_t>(8192, (N + threads - 1) / threads);

    CUDA_CHECK(cudaMemsetAsync(d_sum_w2,  0, sizeof(Real), s));
    CUDA_CHECK(cudaMemsetAsync(d_sum_eta, 0, sizeof(Real), s));

    enstrophy_diss_balance_mapped_kernel<<<blocks, threads, 0, s>>>(
        d_omega, d_lap, Nx, Ny, d_w_node, d_sum_w2, d_sum_eta);
    CUDA_CHECK(cudaGetLastError());
}
//compute diffusion test: d Omega/dt = <omega*rhs> 
__global__ void omega_rhs_balance_mapped_kernel(
    const Real* __restrict__ omega,   // size Nx*Ny
    const Real* __restrict__ rhs,     // size Nx*Ny (current RHS)
    int Nx, int Ny,
    const Real* __restrict__ w_node,  // size Ny, y-weights
    Real* __restrict__ sum_omega_rhs) // single scalar accumulator
{
    extern __shared__ Real sdata[];

    const int tid = threadIdx.x;
    Real local_sum = (Real)0;

    // Flat index over the full grid
    for (int idx = blockIdx.x * blockDim.x + tid;
         idx < Nx * Ny;
         idx += blockDim.x * gridDim.x)
    {
        int j = idx / Nx;

        // interior only, like your enstrophy diagnostic
        //if (j == 0 || j == Ny - 1) continue;
        Real wj   = w_node[j];
        Real omg  = omega[idx];
        Real rhsv = rhs[idx];

        // contribution to ⟨ω·rhs⟩ with same y-weights
        local_sum += wj * omg * rhsv;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    // block reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum_omega_rhs, sdata[0]);
    }
}
inline void launch_omega_rhs_balance_mapped(
    const Real* d_omega,
    const Real* d_rhs,
    int Nx, int Ny,
    const Real* d_w_node,
    Real* d_sum_omega_rhs,
    cudaStream_t stream = 0)
{
    // zero accumulator
    CUDA_CHECK( cudaMemsetAsync(d_sum_omega_rhs, 0, sizeof(Real), stream) );

    const int threads = 256;
    const int blocks  = (Nx * Ny + threads - 1) / threads;
    const size_t shmem = threads * sizeof(Real);

    omega_rhs_balance_mapped_kernel<<<blocks, threads, shmem, stream>>>(
        d_omega, d_rhs, Nx, Ny, d_w_node, d_sum_omega_rhs);

    CUDA_CHECK( cudaGetLastError() );
}

// ============= RHS mapped kernel =============
__global__ void rhs_mapped_kernel(
    const Real* __restrict__ J, const Real* __restrict__ lap, 
    const Real* __restrict__ omg, Real* __restrict__ rhs, 
    int Nx, int Ny, Real nu, Real F0, int nforce, Real h, Real lin_drag,
    const Real* __restrict__ y_node)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix >= Nx || iy >= Ny) return;
    size_t idx = (size_t)iy*Nx + ix;

    Real f = 0.0;
    if (F0 != 0.0){
        Real y = y_node[iy];  // Use actual grid coordinate
        Real arg = (Real)nforce * CUDART_PI * ((y + h)/(2.0*h));
        f = -F0 * ((Real)nforce*CUDART_PI/(2.0*h)) * cos(arg);
    }
    rhs[idx] = -J[idx] + nu*lap[idx] + f - lin_drag*omg[idx];
    //rhs[idx] = nu*lap[idx];
}

inline void launch_rhs_mapped(
    const Real* J, const Real* lap, const Real* omg, Real* rhs,
    int Nx, int Ny, Real nu, Real F0, int nforce, Real h, Real lin_drag,
    const Real* d_y_node, cudaStream_t s=0)
{
    dim3 tb(32,8), gb = blocksFor2D(Nx,Ny,tb.x,tb.y);
    rhs_mapped_kernel<<<gb,tb,0,s>>>(J,lap,omg,rhs,Nx,Ny,nu,F0,nforce,h,lin_drag,d_y_node);
    CUDA_CHECK(cudaGetLastError());
}
// Add manufactured forcing f_omega(x,y,t) to RHS for MMS verification.
//
// Manufactured streamfunction:
//   psi_ex = A(t) * sin(k x) * sin^2(theta),   theta = pi (y+h)/(2h)
// where A(t)=A0*(1 + eps*sin(Om*t)),  k = 2*pi*kx_mode/Lx.
//
// Governing PDE (your solver):
//   omega_t = -J(psi,omega) + nu * Lap(omega) + f_omega - r*omega
//
// MMS forcing definition (continuous):
//   f_omega = omega_t + J(psi,omega) - nu * Lap(omega) + r*omega
//
// This kernel computes f_omega analytically and does: rhs += f_omega.
__global__ void mms_add_forcing_kernel(
    Real* __restrict__ rhs,          // [Ny*Nx] (in/out)
    int Nx, int Ny, Real Lx, Real h, Real t, Real nu, Real r,
    Real A0, Real eps, Real Om, int  kx_mode,
    const Real* __restrict__ y_node) // [Ny] physical y
{
    const size_t N = (size_t)Nx * (size_t)Ny;
    const Real dx = Lx / (Real)Nx;

    // Wavenumbers / constants
    const Real k = (Real)(2.0) * (Real)CUDART_PI * (Real)kx_mode / Lx;
    const Real c = (Real)CUDART_PI / ((Real)2.0 * h);  // theta = c*(y+h)

    // Time dependence
    const Real A      = A0 * ((Real)1.0 + eps * sin(Om * t));
    const Real Aprime = A0 * (eps * Om * cos(Om * t));

    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < N;
         idx += (size_t)gridDim.x * blockDim.x)
    {
        const int j = (int)(idx / (size_t)Nx);
        const int i = (int)(idx - (size_t)j * (size_t)Nx);

        const Real y = y_node[j];
        const Real x = (Real)i * dx;

        // theta, 2theta
        const Real theta  = c * (y + h);
        const Real two_th = (Real)2.0 * theta;

        const Real s_th   = sin(theta);
        const Real s      = s_th * s_th;     // sin^2(theta)
        const Real cos2   = cos(two_th);
        const Real sin2   = sin(two_th);

        // g(y) where omega_ex = A * sin(kx) * g(y)
        // For s=sin^2(theta):
        //   s'' = 2 c^2 cos(2theta)
        //   g   = k^2 s - s'' = k^2 s - 2 c^2 cos(2theta)
        const Real c2 = c * c;
        const Real g  = (k*k) * s - (Real)2.0 * c2 * cos2;

        // g'' = 2 c^2 cos(2theta) * (k^2 + 4 c^2)
        const Real gpp = (Real)2.0 * c2 * cos2 * ( (k*k) + (Real)4.0 * c2 );

        // sin(kx), sin(2kx)
        const Real sk   = sin(k * x);
        const Real s2k  = sin((Real)2.0 * k * x);

        // omega_t term: Aprime * sin(kx) * g
        // -nu Lap(omega) term: nu * A * sin(kx) * (k^2 g - g'')
        // +r omega term: r * A * sin(kx) * g
        const Real term_lin = Aprime * g + (nu * A) * ((k*k) * g - gpp) + (r  * A) * g;

        // IMPORTANT:
        //   The solver's Arakawa kernel returns J_adv = u·∇ω, and RHS uses -J_adv.
        //   MMS must therefore add +J_adv to cancel that term.
        //   For this manufactured solution:
        //     J(ψ,ω) = A^2 k c^3 sin(2kx) sin(2θ)
        //     J_adv  = u·∇ω = -J(ψ,ω)
        const Real c3 = c2 * c;
        const Real term_adv = -(A*A) * k * c3 * s2k * sin2;  // u·∇ω

        const Real fomega = sk * term_lin + term_adv;
        rhs[idx] += fomega;
    }
}

inline void launch_mms_add_forcing(
    Real* d_rhs, int Nx, int Ny, Real Lx, Real h,
    Real t, Real nu, Real r,
    Real A0, Real eps, Real Om, int kx_mode,
    const Real* d_y_node,
    cudaStream_t s=0)
{
    const size_t N = (size_t)Nx * (size_t)Ny;
    int threads = 256;
    int blocks  = (int)std::min<size_t>(8192, (N + threads - 1) / threads);
    mms_add_forcing_kernel<<<blocks, threads, 0, s>>>(
        d_rhs, Nx, Ny, Lx, h, t, nu, r, A0, eps, Om, kx_mode, d_y_node);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void axpy_inplace_kernel(Real* __restrict__ y, const Real* __restrict__ x, const Real* __restrict__ r, Real dt, size_t N){
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) y[i] = x[i] + dt*r[i];
}
__global__ void ssprk2_combine_kernel(Real* __restrict__ w2, const Real* __restrict__ w, const Real* __restrict__ w1, const Real* __restrict__ rhs, Real dt, size_t N){
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) w2[i] = 0.75*w[i] + 0.25*(w1[i] + dt*rhs[i]);
}
__global__ void ssprk3_combine_kernel(Real* __restrict__ wnp1, const Real* __restrict__ w, const Real* __restrict__ w2, const Real* __restrict__ rhs, Real dt, size_t N){
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) wnp1[i] = (1.0/3.0)*w[i] + (2.0/3.0)*(w2[i] + dt*rhs[i]);
}
inline void launch_axpy(Real* y, const Real* x, const Real* r, Real dt, size_t N, cudaStream_t s=0){
    dim3 tb(256), gb((N+tb.x-1)/tb.x);
    axpy_inplace_kernel<<<gb,tb,0,s>>>(y,x,r,dt,N);
    CUDA_CHECK(cudaGetLastError());
}
inline void launch_ssprk2(Real* w2, const Real* w, const Real* w1, const Real* rhs, Real dt, size_t N, cudaStream_t s=0){
    dim3 tb(256), gb((N+tb.x-1)/tb.x);
    ssprk2_combine_kernel<<<gb,tb,0,s>>>(w2,w,w1,rhs,dt,N);
    CUDA_CHECK(cudaGetLastError());
}
inline void launch_ssprk3(Real* wnp1, const Real* w, const Real* w2, const Real* rhs, Real dt, size_t N, cudaStream_t s=0){
    dim3 tb(256), gb((N+tb.x-1)/tb.x);
    ssprk3_combine_kernel<<<gb,tb,0,s>>>(wnp1,w,w2,rhs,dt,N);
}

// =======================================================
// Max |u|, |v| on a mapped (stretched) grid
// u =  ψ_y = a(j) * ψ_η,   v = -ψ_x
// - x is periodic
// - y derivatives use j±1 (one‑sided at the walls by skipping them)
// - two‑stage reduction (per‑block -> final scalar)
// - temporary block buffers are managed inside the launcher
// =======================================================
#ifndef IDX
#define IDX(j,i) ((size_t)(j) * (size_t)Nx + (size_t)(i))
#endif
// ---------------------------
// Stage 1: per‑block maxima
// ---------------------------
__global__ void max_uv_blocks_mapped_kernel( const Real* __restrict__ psi, int Nx, int Ny,
    Real inv_dx, Real inv_deta, const Real* __restrict__ a_node, Real* __restrict__ umax_blk, Real* __restrict__ vmax_blk,
    Real* __restrict__ vmaxm_blk)
{
    extern __shared__ unsigned char smem[];
    Real* sU = reinterpret_cast<Real*>(smem);
    Real* sV = sU + blockDim.x;
    Real* sVm = sV + blockDim.x;

    const size_t tid = threadIdx.x;
    const size_t N   = (size_t)Nx * (size_t)Ny;

    Real umax = 0.0;
    Real vmax = 0.0;
    Real vmaxm = 0.0; // max|v|*a(j), used for mapped-grid y-CFL in eta
   
    // Grid‑stride loop over all cells
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + tid;
         idx < N;
         idx += (size_t)gridDim.x * blockDim.x)
    {
        const int j = (int)(idx / Nx);
        const int i = (int)(idx - (size_t)j * Nx);

        // Skip wall rows for max (no‑slip -> u=v=0 at walls).
        // If you prefer, you can keep walls and use a one‑sided y‑derivative there.
        if (j == 0 || j == Ny - 1) {
            continue;
        }

        const int ip = (i + 1 < Nx) ? (i + 1) : 0;
        const int im = (i > 0)      ? (i - 1) : (Nx - 1);

        // Derivatives of ψ
        const Real dpsideta = (psi[IDX(j+1,i)] - psi[IDX(j-1,i)]) * (0.5 * inv_deta);
        const Real dpsidx   = (psi[IDX(j,ip)] - psi[IDX(j,im)])   * (0.5 * inv_dx);

        // Map to physical u, v
        const Real a = a_node[j];
        const Real u = a * dpsideta;    // ψ_y
        const Real v = -dpsidx;         // -ψ_x

        // Track maxima
        const Real au = fabs(u);
        const Real av = fabs(v);
	const Real avm = av * a;  // v/dy ~ |v|*a/deta (eta-space advective speed)
        if (au > umax) umax = au;
        if (av > vmax) vmax = av;
	if (avm > vmaxm) vmaxm = avm;
    }

    // In‑block reduction
    sU[tid] = umax;
    sV[tid] = vmax;
    sVm[tid] = vmaxm;
    __syncthreads();

    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            if (sU[tid + s] > sU[tid]) sU[tid] = sU[tid + s];
            if (sV[tid + s] > sV[tid]) sV[tid] = sV[tid + s];
            if (sVm[tid + s] > sVm[tid]) sVm[tid] = sVm[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        umax_blk[blockIdx.x] = sU[0];
        vmax_blk[blockIdx.x] = sV[0];
        vmaxm_blk[blockIdx.x] = sVm[0];
    }
}

// ---------------------------------------
// Stage 2: reduce block maxima to scalars
// (single‑block reduction; blocks ≤ 8192)
// ---------------------------------------
__global__ void reduce_max_triple_kernel(const Real* __restrict__ umax_blk, const Real* __restrict__ vmax_blk,
    const Real* __restrict__ vmaxm_blk, int n, Real* __restrict__ umax_out, Real* __restrict__ vmax_out, Real* __restrict__ vmaxm_out)
{
    extern __shared__ unsigned char smem[];
    Real* sU = reinterpret_cast<Real*>(smem);
    Real* sV = sU + blockDim.x;
    Real* sVm = sV + blockDim.x;

    Real umax = 0.0, vmax = 0.0, vmaxm = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const Real u = umax_blk[i];
        const Real v = vmax_blk[i];
        const Real vm = vmaxm_blk[i];
        if (u > umax) umax = u;
        if (v > vmax) vmax = v;
        if (vm > vmaxm) vmaxm = vm;
    }

    sU[threadIdx.x] = umax;
    sV[threadIdx.x] = vmax;
    sVm[threadIdx.x] = vmaxm;
    __syncthreads();

    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sU[threadIdx.x + s] > sU[threadIdx.x]) sU[threadIdx.x] = sU[threadIdx.x + s];
            if (sV[threadIdx.x + s] > sV[threadIdx.x]) sV[threadIdx.x] = sV[threadIdx.x + s];
            if (sVm[threadIdx.x + s] > sVm[threadIdx.x]) sVm[threadIdx.x] = sVm[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *umax_out = sU[0];
        *vmax_out = sV[0];
        *vmaxm_out = sVm[0];
    }
}

// -------------------------------------------------------
// Host launcher: same call signature you already use
// -------------------------------------------------------
inline void launch_max_uv_from_psi_mapped( const Real* d_psi, int Nx, int Ny, Real inv_dx, Real inv_deta,
    const Real* d_a_node, Real* d_umax, Real* d_vmax, Real* d_vmaxm,       // device scalars (size 1 each)
    cudaStream_t stream = 0)
{
    // Sanity guard for tiny grids
    if (Nx < 2 || Ny < 3) {
        const Real zero = 0;
        CUDA_CHECK(cudaMemcpyAsync(d_umax, &zero, sizeof(Real), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_vmax, &zero, sizeof(Real), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_vmaxm, &zero, sizeof(Real), cudaMemcpyHostToDevice, stream));
        return;
    }

    const int threads = 256;
    const size_t N    = (size_t)Nx * (size_t)Ny;

    // Cap block count to keep shared memory small; 8192 works well on all GPUs
    const int blocks  = (int)std::min<size_t>(8192, (N + threads - 1) / threads);
    const size_t shmem_stage1 = 3 * threads * sizeof(Real);
    const size_t shmem_stage2 = 3 * threads * sizeof(Real);

    // Persistent scratch for per‑block maxima
    static int   cap = 0;
    static Real* d_umax_blk = nullptr;
    static Real* d_vmax_blk = nullptr;
    static Real* d_vmaxm_blk = nullptr;
    if (blocks > cap) {
        if (d_umax_blk) CUDA_CHECK(cudaFree(d_umax_blk));
        if (d_vmax_blk) CUDA_CHECK(cudaFree(d_vmax_blk));
        if (d_vmaxm_blk) CUDA_CHECK(cudaFree(d_vmaxm_blk));
        CUDA_CHECK(cudaMalloc(&d_umax_blk, blocks * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_vmax_blk, blocks * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_vmaxm_blk, blocks * sizeof(Real)));
        cap = blocks;
    }

    // Stage 1: compute per‑block maxima over all Nx*Ny cells
    max_uv_blocks_mapped_kernel<<<blocks, threads, shmem_stage1, stream>>>(
        d_psi, Nx, Ny, inv_dx, inv_deta, d_a_node, d_umax_blk, d_vmax_blk, d_vmaxm_blk);
    CUDA_CHECK(cudaGetLastError());

    // Stage 2: reduce per‑block maxima to single scalars
    reduce_max_triple_kernel<<<1, threads, shmem_stage2, stream>>>(
        d_umax_blk, d_vmax_blk, d_vmaxm_blk, blocks, d_umax, d_vmax, d_vmaxm);
    CUDA_CHECK(cudaGetLastError());
}

/* -------------------- Z2Z Poisson (x-FFT + y-tridiagonal) -------------------- */
using Cx = cufftDoubleComplex;
__global__ void real_to_complex_kernel(const Real* __restrict__ r, Cx* __restrict__ z, size_t N){
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N){ z[i].x = r[i]; z[i].y = 0.0; }
}
__global__ void complex_to_real_scaled_kernel(const Cx* __restrict__ z, Real* __restrict__ r, size_t N, Real s){
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N){ r[i] = s * z[i].x; }
}
__global__ void fill_kx2_z2z_kernel(Real* kx2, int Nx, Real Lx){
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= Nx) return;
    int kphys_i = (k <= Nx/2 ? k : k - Nx);
    Real kphys = (2.0*CUDART_PI * (Real)kphys_i) / Lx;
    kx2[k] = kphys*kphys;
}
__global__ void precompute_cp_kernel(Real* __restrict__ cp, const Real* __restrict__ kx2, int Nx, int Ny, Real inv_dy2){
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= Nx) return;
    int Ni = Ny-2;
    Real a = inv_dy2, c = inv_dy2, b = -2.0*inv_dy2 - kx2[k];
    Real cp_prev = c / b;
    cp[(size_t)k*Ni + 0] = cp_prev;
    for (int j=1; j<Ni; ++j){
        Real den = b - a*cp_prev;
        Real cpi = c/den;
        cp[(size_t)k*Ni + j] = cpi;
        cp_prev = cpi;
    }
}

__global__ void precompute_cp_mapped_kernel(Real* __restrict__ cp,
    const Real* __restrict__ a_node, const Real* __restrict__ a_edge, const Real* __restrict__ kx2,
    int Nx, int Ny, Real inv_deta2)
{
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= Nx) return;
    int Ni = Ny - 2;
    // Row j=1 (index 0)
    int j = 1;
    Real an = a_node[j];
    Real ae_p = a_edge[j];     // j+1/2
    Real ae_m = a_edge[j-1];   // j-1/2
    Real a_sub = an * ae_m * inv_deta2;
    Real c_sup = an * ae_p * inv_deta2;
    Real b = - (a_sub + c_sup) - kx2[k];
    Real cp_prev = c_sup / b;
    cp[(size_t)k*Ni + 0] = cp_prev;
    for (int r=1; r<Ni; ++r){
        j = r+1;
        an = a_node[j];
        ae_p = a_edge[j];
        ae_m = a_edge[j-1];
        a_sub = an * ae_m * inv_deta2;
        c_sup = an * ae_p * inv_deta2;
        b = - (a_sub + c_sup) - kx2[k];
        Real den = b - a_sub * cp_prev;
        Real cpi = c_sup / den;
        cp[(size_t)k*Ni + r] = cpi;
        cp_prev = cpi;
    }
}

__global__ void poisson_tridiag_batched_mapped_kernel(
    const Cx* __restrict__ omega_hat, Cx* __restrict__ psi_hat,
    const Real* __restrict__ cp, const Real* __restrict__ a_node, const Real* __restrict__ a_edge,
    const Real* __restrict__ kx2, int Nx, int Ny, Real inv_deta2, Real psi_bot, Real psi_top)
{
    int k = blockIdx.x;
    if (k >= Nx) return;
    int Ni = Ny - 2;
    const Cx* wh = omega_hat + k;
    Cx* ph       = psi_hat   + k;

    // Row j=1 (index 0)
    int j = 1;
    Real an = a_node[j];
    Real ae_p = a_edge[j];
    Real ae_m = a_edge[j-1];
    Real a_sub = an * ae_m * inv_deta2;
    Real c_sup = an * ae_p * inv_deta2;
    Real b = - (a_sub + c_sup) - kx2[k];

    Cx dp_prev; dp_prev.x=0.0; dp_prev.y=0.0;
    Cx rhs0 = wh[Nx*j]; rhs0.x = -rhs0.x; rhs0.y = -rhs0.y;
    // *** FIX: wall ψ contribution only for kx = 0 ***
    if (k == 0){
        rhs0.x -= a_sub * psi_bot * (Real)Nx;
    }

    Real den0 = b;
    Cx dp0; dp0.x = (rhs0.x - a_sub*dp_prev.x)/den0; dp0.y = (rhs0.y - a_sub*dp_prev.y)/den0;
    ph[Nx*j] = dp0; dp_prev = dp0;

    for (int r=1; r<Ni; ++r){
        j = r+1;
        an = a_node[j];
        ae_p = a_edge[j];
        ae_m = a_edge[j-1];
        a_sub = an * ae_m * inv_deta2;
        c_sup = an * ae_p * inv_deta2;
        b = - (a_sub + c_sup) - kx2[k];
        Cx rhs = wh[Nx*j]; rhs.x = -rhs.x; rhs.y = -rhs.y;

        // *** FIX: top-wall contribution only for kx = 0, last row ***
        if (k == 0 && r == Ni-1){
            rhs.x -= c_sup * psi_top * (Real)Nx;
        }

        Real cpj = cp[(size_t)k*Ni + r];
        Real den = c_sup / cpj; // equals b - a_sub * cp_prev
        Cx dpj; dpj.x = (rhs.x - a_sub*dp_prev.x)/den; dpj.y = (rhs.y - a_sub*dp_prev.y)/den;
        ph[Nx*j] = dpj; dp_prev = dpj;
    }
    Cx xnext = ph[Nx*(Ni)];
    for (int r=Ni-2; r>=0; --r){
        j = r+1;
        Real cpj = cp[(size_t)k*Ni + r];
        Cx dpj = ph[Nx*j];
        Cx xj; xj.x = dpj.x - cpj*xnext.x; xj.y = dpj.y - cpj*xnext.y;
        ph[Nx*j] = xj;
        xnext = xj;
    }
    Cx zero; zero.x=0.0; zero.y=0.0;
    ph[0] = (k==0 ? Cx{ (Real)Nx*psi_bot, 0.0 } : zero);
    ph[Nx*(Ny-1)] = (k==0 ? Cx{ (Real)Nx*psi_top, 0.0 } : zero);
}

// -------------------- Poisson (FFT in x; tri-diagonal in y) --------------------
struct Poisson2D_Z2Z {
    int Nx, Ny;
    cufftHandle fwd{}, inv{};
    Real *d_kx2{nullptr}, *d_cp{nullptr};
    Cx *d_hat_in{nullptr}, *d_hat_out{nullptr};
    bool profile{false};
    float acc_fft_ms{0.0f};
    float acc_tridiag_ms{0.0f};

    // mapped support (external variable coefficients; not owned)
    bool mapped{false};
    bool cp_mapped_ready{false};
    const Real *d_a_node{nullptr}, *d_a_edge{nullptr};
    Real inv_deta2{0.0};

    // Constructor: build FFT plans, kx^2, and cp for a uniform grid
    Poisson2D_Z2Z(const Grid& G, cudaStream_t s=0) : Nx(G.Nx), Ny(G.Ny){
   	    // 1D FFT in x, batched over Ny rows
	    CUFFT_CHECK(cufftPlan1d(&fwd, Nx, CUFFT_Z2Z, Ny));
	    CUFFT_CHECK(cufftPlan1d(&inv, Nx, CUFFT_Z2Z, Ny));
	    CUFFT_CHECK(cufftSetStream(fwd,s));
	    CUFFT_CHECK(cufftSetStream(inv,s));
	    // kx^2 array
	    CUDA_CHECK(cudaMalloc(&d_kx2, (size_t)Nx*sizeof(Real)));
	    int threads = 256, blocks = (Nx + threads - 1)/threads;
	    fill_kx2_z2z_kernel<<<blocks,threads,0,s>>>(d_kx2, Nx, G.Lx);
	    CUDA_CHECK(cudaGetLastError());
            // tri-diagonal forward thomas coefficient cp(k,j), j=1...Ny-2
	    const int Ni = Ny - 2;
	    CUDA_CHECK(cudaMalloc(&d_cp, (size_t)Nx*Ni*sizeof(Real)));

	    // Fallback uniform coefficients using physical dy; for stretched grids we overwrite cp in enable_mapped()
	    precompute_cp_kernel<<<blocks,threads,0,s>>>( d_cp, d_kx2, Nx, Ny, 1.0/(G.dy()*G.dy()));
	    CUDA_CHECK(cudaGetLastError());
            // spectral work buffers
	    CUDA_CHECK(cudaMalloc(&d_hat_in,  (size_t)Nx*Ny*sizeof(Cx)));
	    CUDA_CHECK(cudaMalloc(&d_hat_out, (size_t)Nx*Ny*sizeof(Cx)));
    }
    ~Poisson2D_Z2Z() {
        if (d_hat_out) cudaFree(d_hat_out);
        if (d_hat_in)  cudaFree(d_hat_in);
        if (d_cp)      cudaFree(d_cp);
        if (d_kx2)     cudaFree(d_kx2);
        if (fwd) cufftDestroy(fwd);
        if (inv) cufftDestroy(inv);
    }

    inline void set_profile(bool on){ profile = on; }
    inline void profile_reset(){ acc_fft_ms = 0.0f; acc_tridiag_ms = 0.0f; }
    inline void profile_get(float& fft_ms, float& tridiag_ms){ fft_ms = acc_fft_ms; tridiag_ms = acc_tridiag_ms; }
    
    // attach mapped metrics and precompute cp(k,j) for the mapped laplacian.
    // inv_deta is 1/Delta eta on the computational eta in [-1, 1] grid.
    inline void enable_mapped(const Real* a_node, const Real* a_edge, Real inv_deta, cudaStream_t s=0){
	    mapped = true; d_a_node = a_node; d_a_edge = a_edge; inv_deta2 = inv_deta*inv_deta;

	    int threads = 256, blocks = (Nx + threads - 1)/threads;
	    // Precompute cp(k,j) ONCE for the mapped Laplacian; reused every solve().
	    precompute_cp_mapped_kernel<<<blocks,threads,0,s>>>(d_cp, d_a_node, d_a_edge, d_kx2, Nx, Ny, inv_deta2);
	    CUDA_CHECK(cudaGetLastError());

	    cp_mapped_ready = true;
    }

    // NOTE: psi_bot/top Dirichlet enforced on kx=0 mode in spectral space Solve -∇²ψ = ω with ψ(y=±h) = psi_bot/top (Dirichlet in kx=0 mode).
    void solve(const Real* d_omega, Real* d_psi, Real dy, Real psi_bot, Real psi_top, cudaStream_t s=0){
		// dy is for API compatibility of uniform case, not used in mapped path
        size_t N = (size_t)Nx*Ny;
        int threads=256, blocks=(N+threads-1)/threads;

        cudaEvent_t e0,e1,e2,e3; if (profile){ cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3); }

        // pack real->complex
        real_to_complex_kernel<<<blocks,threads,0,s>>>(d_omega, d_hat_in, N);
        CUDA_CHECK(cudaGetLastError());

        // forward FFT in x
        if (profile) cudaEventRecord(e0, s);
        CUFFT_CHECK(cufftExecZ2Z(fwd, d_hat_in, d_hat_in, CUFFT_FORWARD));
        if (profile){ cudaEventRecord(e1, s); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); acc_fft_ms += ms; }

        // batched tri-diagonal solve in y for each kx
        if (profile) cudaEventRecord(e2, s);
	dim3 gb(Nx), tb(1);
	// cp already precomputed (uniform constructor or enable_mapped());
        // just use the mapped tridiagonal solve
	poisson_tridiag_batched_mapped_kernel<<<gb,tb,0,s>>>(d_hat_in, d_hat_in, d_cp, d_a_node, d_a_edge, d_kx2, Nx, Ny, inv_deta2, psi_bot, psi_top);
        CUDA_CHECK(cudaGetLastError());
        if (profile){ cudaEventRecord(e3, s); cudaEventSynchronize(e3); float ms; cudaEventElapsedTime(&ms, e2, e3); acc_tridiag_ms += ms; }

        // inverse FFT
        if (profile) cudaEventRecord(e2, s);
        CUFFT_CHECK(cufftExecZ2Z(inv, d_hat_in, d_hat_out, CUFFT_INVERSE));
        if (profile){ cudaEventRecord(e3, s); cudaEventSynchronize(e3); float ms; cudaEventElapsedTime(&ms, e2, e3); acc_fft_ms += ms; }

        // unpack and scale by 1/Nx (cuFFT convention)
        complex_to_real_scaled_kernel<<<blocks,threads,0,s>>>(d_hat_out, d_psi, N, (Real)(1.0/(Real)Nx));
        CUDA_CHECK(cudaGetLastError());

        if (profile){ cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3); }
    }
};

/* -------------------- HDF5 helpers -------------------- */
enum SnapFmt { SNAP_H5, SNAP_BIN, SNAP_BOTH };

#ifndef NO_HDF5
static void h5_write_attr_double(hid_t where, const char* name, double v){
    hid_t sid = H5Screate(H5S_SCALAR);
    hid_t aid = H5Acreate2(where, name, H5T_NATIVE_DOUBLE, sid, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(aid, H5T_NATIVE_DOUBLE, &v); H5Aclose(aid); H5Sclose(sid);
}
static void h5_write_attr_int(hid_t where, const char* name, int v){
    hid_t sid = H5Screate(H5S_SCALAR);
    hid_t aid = H5Acreate2(where, name, H5T_NATIVE_INT, sid, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(aid, H5T_NATIVE_INT, &v); H5Aclose(aid); H5Sclose(sid);
}
static void h5_write_2d(hid_t file, const char* dname, const Real* data, int Nx, int Ny){
    hsize_t dims[2] = { (hsize_t)Ny, (hsize_t)Nx };
    hid_t space = H5Screate_simple(2, dims, NULL);
    hid_t dset  = H5Dcreate2(file, dname, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(dset); H5Sclose(space);
}
#endif

/* -------------------- Snapshots -------------------- */
static void save_snapshot_any(const std::string& outdir, int Nx, int Ny, Real t,
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
static void write_profiles_csv(const std::string& path,
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

/* -------------------- 1D spectra in x -------------------- */
__global__ void fill_kx_array_kernel(Real* kx, int Nxc, Real Lx){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < Nxc){
        kx[i] = (2.0*CUDART_PI * (Real)i) / Lx; // i = 0..Nx/2
    }
}
__global__ void spectra1d_rowReduce_kernel(
    const cufftDoubleComplex* __restrict__ Uhat,
    const cufftDoubleComplex* __restrict__ Vhat,
    int Ny, int Nxc, Real norm1d,
    Real* __restrict__ Eu_tmp, Real* __restrict__ Ev_tmp)
{
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= Nxc) return;
    double wx = (k>0 && k<Nxc-1)? 2.0 : 1.0; // R2C half-spectrum weight
    double sumU = 0.0, sumV = 0.0;
    for (int j=0;j<Ny;j++){
        size_t idx = (size_t)j*Nxc + k;
        double ur=Uhat[idx].x, ui=Uhat[idx].y;
        double vr=Vhat[idx].x, vi=Vhat[idx].y;
        sumU += (ur*ur + ui*ui);
        sumV += (vr*vr + vi*vi);
    }
    Eu_tmp[k] = norm1d * wx * sumU;
    Ev_tmp[k] = norm1d * wx * sumV;
}
__global__ void vec_add_kernel(Real* acc, const Real* tmp, int n){
    int i=blockIdx.x*blockDim.x + threadIdx.x;
    if (i<n) acc[i] += tmp[i];
}

struct Spectra1D {
    int Nx, Ny, Nxc;
    cufftHandle plan{};
    cufftDoubleComplex *d_Uhat{nullptr}, *d_Vhat{nullptr};
    Real *d_kx{nullptr};
    Real *d_Eu_acc{nullptr}, *d_Ev_acc{nullptr};
    Real *d_Eu_tmp{nullptr}, *d_Ev_tmp{nullptr};
    Real norm1d{0};
    int samples{0};

    void init(const Grid& G){
        Nx = G.Nx; Ny = G.Ny; Nxc = Nx/2 + 1;
        CUFFT_CHECK(cufftPlan1d(&plan, Nx, CUFFT_D2Z, Ny)); // batch Ny
        CUDA_CHECK(cudaMalloc(&d_Uhat, (size_t)Ny*Nxc*sizeof(cufftDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_Vhat, (size_t)Ny*Nxc*sizeof(cufftDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_kx, Nxc*sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_Eu_acc, Nxc*sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_Ev_acc, Nxc*sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_Eu_tmp, Nxc*sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_Ev_tmp, Nxc*sizeof(Real)));
        CUDA_CHECK(cudaMemset(d_Eu_acc, 0, Nxc*sizeof(Real)));
        CUDA_CHECK(cudaMemset(d_Ev_acc, 0, Nxc*sizeof(Real)));
        int tb=256, gb=(Nxc+tb-1)/tb;
        fill_kx_array_kernel<<<gb,tb>>>(d_kx, Nxc, G.Lx);
        CUDA_CHECK(cudaGetLastError());
        // ∫∫ 0.5(u^2+v^2) dx dy = sum_k [ 0.5 * (dx*dy/Nx) * (|Û|^2 + |V̂|^2)_full ]
        // Area-averaged spectra with cuFFT UNnormalized forward DFT (diagnostics §8.5):
        // Eu(km) = wm/(Nx^2*Ny) * sum_j |û_j(km)|^2   (and similarly for Ev)
        // so that ∑_m E(km) = <u^2+v^2>_A = 2K.
        norm1d = 1.0 / ((Real)G.Nx * (Real)G.Nx * (Real)G.Ny);
        samples = 0;
    }
    ~Spectra1D(){
        if (d_Ev_tmp) cudaFree(d_Ev_tmp);
        if (d_Eu_tmp) cudaFree(d_Eu_tmp);
        if (d_Ev_acc) cudaFree(d_Ev_acc);
        if (d_Eu_acc) cudaFree(d_Eu_acc);
        if (d_kx) cudaFree(d_kx);
        if (d_Vhat) cudaFree(d_Vhat);
        if (d_Uhat) cudaFree(d_Uhat);
        if (plan) cufftDestroy(plan);
    }
    void accumulate(Real* d_u, Real* d_v){
        CUFFT_CHECK(cufftExecD2Z(plan, (cufftDoubleReal*)d_u, d_Uhat));
        CUFFT_CHECK(cufftExecD2Z(plan, (cufftDoubleReal*)d_v, d_Vhat));
        int tb=256, gb=(Nxc+tb-1)/tb;
        spectra1d_rowReduce_kernel<<<gb,tb>>>(d_Uhat,d_Vhat,Ny,Nxc,norm1d,d_Eu_tmp,d_Ev_tmp);
        CUDA_CHECK(cudaGetLastError());
        vec_add_kernel<<<gb,tb>>>(d_Eu_acc,d_Eu_tmp,Nxc);
        vec_add_kernel<<<gb,tb>>>(d_Ev_acc,d_Ev_tmp,Nxc);
        samples++;
    }
    void write_csv(const std::string& path){
        std::vector<Real> Hkx(Nxc), Heu(Nxc), Hev(Nxc);
        CUDA_CHECK(cudaMemcpy(Hkx.data(), d_kx, Nxc*sizeof(Real), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(Heu.data(), d_Eu_acc, Nxc*sizeof(Real), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(Hev.data(), d_Ev_acc, Nxc*sizeof(Real), cudaMemcpyDeviceToHost));
        if (samples>0){
            for (int k=0;k<Nxc;k++){ Heu[k]/=samples; Hev[k]/=samples; }
        }
        std::ofstream f(path);
        f.setf(std::ios::scientific); f.precision(10);
        f << "kx,E_u,E_v,E_tot\n";
        for (int k=0;k<Nxc;k++){
            Real Etot = Heu[k] + Hev[k];
            f << Hkx[k] << "," << Heu[k] << "," << Hev[k] << "," << Etot << "\n";
        }
    }
};

/* -------------------- CLI & init -------------------- */
static void parse_args(int argc, char** argv, Grid& G, Params& P,
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
static void init_omega_with_IC(const Grid& G, const Params& P, std::vector<Real>& h_w){
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

#ifdef EXPORT_TEST_API
// ========= IMEX‑ARK3 scalar ODE helpers (test only) =========
// N = λ_E u,  L = λ_I u  on a vector of length Ntot
__global__ void linear_split_NL_kernel(
    const Real* __restrict__ u,
    Real* __restrict__ N,
    Real* __restrict__ L,
    Real lambdaE, Real lambdaI,
    size_t Ntot)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Ntot) {
        Real v = u[i];
        N[i] = lambdaE * v;
        L[i] = lambdaI * v;
    }
}

inline void launch_linear_split_NL(
    const Real* u, Real* N, Real* L,
    Real lambdaE, Real lambdaI,
    size_t Ntot, cudaStream_t s = 0)
{
    int threads = 256;
    int blocks  = (int)((Ntot + threads - 1) / threads);
    linear_split_NL_kernel<<<blocks,threads,0,s>>>(
        u, N, L, lambdaE, lambdaI, Ntot);
    CUDA_CHECK(cudaGetLastError());
}

// Solve (I - γ ν Δt λ_I) u = rhs  elementwise
__global__ void linear_helmholtz_solve_kernel(
    Real* __restrict__ u_out,
    const Real* __restrict__ rhs,
    Real lambdaI, Real nu, Real dt, Real gamma,
    size_t Ntot)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Ntot) {
        Real denom = (Real)1.0 - gamma * nu * dt * lambdaI;
        u_out[i]   = rhs[i] / denom;
    }
}

inline void launch_linear_helmholtz_solve(
    Real* u_out, const Real* rhs,
    Real lambdaI, Real nu, Real dt, Real gamma,
    size_t Ntot, cudaStream_t s = 0)
{
    int threads = 256;
    int blocks  = (int)((Ntot + threads - 1) / threads);
    linear_helmholtz_solve_kernel<<<blocks,threads,0,s>>>(
        u_out, rhs, lambdaI, nu, dt, gamma, Ntot);
    CUDA_CHECK(cudaGetLastError());
}

// ========= C‑linkage test entry: IMEX‑ARK3 scalar ODE =========
// Integrates u' = λ_E u + λ_I u on [0,T], T = nsteps*dt,
// and returns |u_num(T) - u_exact(T)| in *err_out.
extern "C" void testapi_imex_ark3_linear_ode(
    Real dt, int nsteps, Real* err_out)
{
    const size_t Ntot   = 1;            // one DOF is enough
    const Real   lambdaE = (Real)(-1.0);   // explicit part
    const Real   lambdaI = (Real)(-10.0);  // implicit (stiff) part
    const Real   nu      = (Real)1.0;      // so F_I = nu*lambdaI*u = lambdaI*u

    // Device buffers
    Real *d_u_n   = nullptr;  // holds u^n and then u^{n+1}
    Real *d_u_old = nullptr;  // saved u^n
    Real *d_u2    = nullptr;  // u^(2)
    Real *d_u3    = nullptr;  // u^(3)
    Real *d_u4    = nullptr;  // u^(4)
    Real *d_rhs   = nullptr;

    Real *d_N1=nullptr, *d_N2=nullptr, *d_N3=nullptr, *d_N4=nullptr;
    Real *d_L1=nullptr, *d_L2=nullptr, *d_L3=nullptr, *d_L4=nullptr;

    CUDA_CHECK(cudaMalloc(&d_u_n,   Ntot*sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_u_old, Ntot*sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_u2,    Ntot*sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_u3,    Ntot*sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_u4,    Ntot*sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_rhs,   Ntot*sizeof(Real)));

    CUDA_CHECK(cudaMalloc(&d_N1, Ntot*sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_N2, Ntot*sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_N3, Ntot*sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_N4, Ntot*sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_L1, Ntot*sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_L2, Ntot*sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_L3, Ntot*sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_L4, Ntot*sizeof(Real)));

    // Initial condition
    const Real u0 = (Real)1.23456789;
    CUDA_CHECK(cudaMemcpy(d_u_n, &u0, sizeof(Real), cudaMemcpyHostToDevice));

    cudaStream_t s = 0;

    for (int n = 0; n < nsteps; ++n) {

        // Save u^n in a separate buffer so we can use it in the final update
        CUDA_CHECK(cudaMemcpyAsync(
            d_u_old, d_u_n, Ntot*sizeof(Real),
            cudaMemcpyDeviceToDevice, s));

        // ---- Stage 1: Y1 = u^n ----
        // N1 = λ_E Y1, L1 = λ_I Y1
        launch_linear_split_NL(
            d_u_old, d_N1, d_L1, lambdaE, lambdaI, Ntot, s);

        // ---- Stage 2 ----
        // rhs2 = u^n + dt * ( aE21 N1 + ν aI21 L1 )
        launch_ark3_stage2_rhs(
            d_rhs, d_u_old, d_N1, d_L1,
            dt, nu, Ntot, s);

        // Y2 via scalar "Helmholtz" solve: (I - γ ν dt λ_I) Y2 = rhs2
        launch_linear_helmholtz_solve(
            d_u2, d_rhs,
            lambdaI, nu, dt, (Real)ARK3::gamma, Ntot, s);

        // N2, L2 from Y2
        launch_linear_split_NL(
            d_u2, d_N2, d_L2, lambdaE, lambdaI, Ntot, s);

        // ---- Stage 3 ----
        // rhs3 = u^n + dt*(aE31 N1 + aE32 N2 + ν(aI31 L1 + aI32 L2))
        launch_ark3_stage3_rhs(
            d_rhs, d_u_old,
            d_N1, d_N2, d_L1, d_L2,
            dt, nu, Ntot, s);

        // Y3
        launch_linear_helmholtz_solve(
            d_u3, d_rhs,
            lambdaI, nu, dt, (Real)ARK3::gamma, Ntot, s);

        // N3, L3 from Y3
        launch_linear_split_NL(
            d_u3, d_N3, d_L3, lambdaE, lambdaI, Ntot, s);

        // ---- Stage 4 ----
        // rhs4 = u^n + dt*(aE41 N1 + aE42 N2 + aE43 N3)
        //              + ν dt*(aI41 L1 + aI42 L2 + aI43 L3)
        launch_ark3_stage4_rhs(
            d_rhs, d_u_old,
            d_N1, d_N2, d_N3,
            d_L1, d_L2, d_L3,
            dt, nu, Ntot, s);

        // Y4
        launch_linear_helmholtz_solve(
            d_u4, d_rhs,
            lambdaI, nu, dt, (Real)ARK3::gamma, Ntot, s);

        // N4, L4 from Y4
        launch_linear_split_NL(
            d_u4, d_N4, d_L4, lambdaE, lambdaI, Ntot, s);

        // ---- Final 3rd‑order ARK3 update ----
        // u^{n+1} = u^n + dt Σ b_i N_i + ν dt Σ b_i L_i
        launch_ark3_final_weighted(
            d_u_n,        // omega_out (u^{n+1})
            d_u_old,      // omega_n  (u^n)
            d_N1, d_N2, d_N3, d_N4,
            d_L1, d_L2, d_L3, d_L4,
            dt, nu, Ntot, s);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Compare against exact solution u(T) = u0 * exp((λ_E + λ_I) T)
    Real uT;
    CUDA_CHECK(cudaMemcpy(&uT, d_u_n, sizeof(Real), cudaMemcpyDeviceToHost));
    const Real   T      = dt * (Real)nsteps;
    const double lamTot = (double)lambdaE + (double)lambdaI;
    const double u_ex   = (double)u0 * std::exp(lamTot * (double)T);
    const double err    = std::abs((double)uT - u_ex);

    if (err_out) *err_out = (Real)err;

    // Cleanup
    CUDA_CHECK(cudaFree(d_L4));
    CUDA_CHECK(cudaFree(d_L3));
    CUDA_CHECK(cudaFree(d_L2));
    CUDA_CHECK(cudaFree(d_L1));
    CUDA_CHECK(cudaFree(d_N4));
    CUDA_CHECK(cudaFree(d_N3));
    CUDA_CHECK(cudaFree(d_N2));
    CUDA_CHECK(cudaFree(d_N1));
    CUDA_CHECK(cudaFree(d_rhs));
    CUDA_CHECK(cudaFree(d_u4));
    CUDA_CHECK(cudaFree(d_u3));
    CUDA_CHECK(cudaFree(d_u2));
    CUDA_CHECK(cudaFree(d_u_old));
    CUDA_CHECK(cudaFree(d_u_n));
}
//================End of ARK3 time integration test=======================

extern "C" void testapi_uv_from_psi_mapped(
    const Real* d_psi, Real* d_u, Real* d_v,
    int Nx, int Ny,
    Real inv_dx, Real inv_deta,          // scalars first
    const Real* d_a_node,                // pointer last
    cudaStream_t s)
{
    launch_uv_from_psi_mapped(d_psi, d_u, d_v,
                              Nx, Ny,
                              inv_dx, inv_deta, d_a_node, s);
}

extern "C" void testapi_omega_from_psi_interior_mapped(
    Real* d_w, const Real* d_psi,
    int Nx, int Ny,
    Real inv_dx2, Real inv_deta2,     // scalars
    const Real* d_a_node,             // Ny
    const Real* d_a_edge,             // Ny-1
    cudaStream_t s)
{
    launch_omega_from_psi_interior_mapped(d_w, d_psi, Nx, Ny,
                                          inv_dx2, inv_deta2, d_a_node, d_a_edge, s);
}

extern "C" void testapi_thom_wall_vorticity_mapped(
    Real* d_w, const Real* d_psi,
    int Nx, int Ny,
    const Real* d_dy_edge,            // Ny-1 (physical Δy at edges)
    Real psi_bot, Real psi_top,
    cudaStream_t s)
{
    // no inv_deta2 or d_y needed; Thom uses physical Δy near the walls
    launch_thom_wall_vorticity_mapped(d_w, d_psi, Nx, Ny,
                                      d_dy_edge, psi_bot, psi_top, s);
}

extern "C" void testapi_arakawa_J_mapped(
    const Real* d_psi, const Real* d_w, Real* d_J,
    int Nx, int Ny,
    Real inv_dx, Real inv_deta,       // scalars
    const Real* d_a_node,             // Ny
    cudaStream_t s)
{
    launch_arakawa_J_mapped(d_psi, d_w, d_J, Nx, Ny,
                            inv_dx, inv_deta, d_a_node, s);
}

extern "C" void testapi_compute_adv_from_uv_mapped(
    const Real* d_u, const Real* d_v, const Real* d_omg, Real* d_Jref,
    int Nx, int Ny,
    Real inv_dx, Real inv_deta,
    const Real* d_a_node, cudaStream_t s)
{
    dim3 tb(32,4), gb((Nx+31)/32,(Ny+3)/4);
    compute_adv_from_uv_mapped<<<gb,tb,0,s>>>(
        d_u, d_v, d_omg, d_Jref, Nx, Ny, inv_dx, inv_deta, d_a_node);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void testapi_compute_adv_flux_mapped(
    const Real* d_u, const Real* d_v, const Real* d_omg, Real* d_Jref,
    int Nx, int Ny,
    Real inv_dx, Real inv_deta,
    const Real* d_a_node, cudaStream_t s)
{
    dim3 tb(32,4), gb((Nx+31)/32,(Ny+3)/4);
    compute_adv_flux_mapped<<<gb,tb,0,s>>>(
        d_u, d_v, d_omg, d_Jref, Nx, Ny, inv_dx, inv_deta, d_a_node);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void testapi_arakawa_J_mapped_split(
    const Real* d_psi, const Real* d_omg,
    Real* d_J1, Real* d_J2, Real* d_J3, Real* d_Jadv,
    int Nx, int Ny,
    Real inv_dx, Real inv_deta,
    const Real* d_a_node,
    cudaStream_t s)
{
    dim3 tb(32,4), gb((Nx+31)/32,(Ny+3)/4);
    arakawa_J_mapped_split_kernel<<<gb,tb,0,s>>>(
        d_psi, d_omg, d_J1, d_J2, d_J3, d_Jadv,
        Nx, Ny, inv_dx, inv_deta, d_a_node);
    CUDA_CHECK(cudaGetLastError());
}
#endif
