// Part of nse2d-channel-gpu: GPU DNS of 2D channel turbulence (vorticity-streamfunction).
// Run log, scheduling helpers, HDF5 snapshots and CSV writers.
#pragma once
#include "common.cuh"

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

enum SnapFmt { SNAP_H5, SNAP_BIN, SNAP_BOTH };

bool h5_has_dataset(hid_t file, const char* name);
double h5_read_attr_double(hid_t file, const char* name, double defv);
int h5_read_attr_int(hid_t file, const char* name, int defv);
void h5_read_2d(hid_t file, const char* name, Real* out, int Nx, int Ny);
bool dir_exists(const std::string& p);
bool make_one(const std::string& p);
bool ensure_dir_p(const std::string& p);
std::string fmt_hms(double sec);
void progress_append(const std::string& path, const std::string& line);
size_t scheduled_count(double tend, double dt_save);
void log_event(EventProgress& P, const std::string& path, double t, double dt);
void log_step(StepProgress& P, const std::string& path, double t, double dt, int every);
void tee_progress(const std::string& path, const char* fmt, ...);
bool file_nonempty(const std::string& path);
double time_tol(double t);
double sched_tol(double t, double dt);
double next_after(double t0, double dt);
void h5_write_attr_double(hid_t where, const char* name, double v);
void h5_write_attr_int(hid_t where, const char* name, int v);
void h5_write_2d(hid_t file, const char* dname, const Real* data, int Nx, int Ny);
void save_snapshot_any(const std::string& outdir, int Nx, int Ny, Real t,
                              const Real* d_psi, const Real* d_w,
                              const Real* d_u,   const Real* d_v,
                              SnapFmt fmt, const Grid& G, const Params& P);
void write_profiles_csv(const std::string& path,
                               const Grid& G, const Params& P,
                               double utau_mean, double T_accum,
                               const std::vector<Real>& H_int_u,
                               const std::vector<Real>& H_int_v,
                               const std::vector<Real>& H_int_u2,
                               const std::vector<Real>& H_int_v2,
                               const std::vector<Real>& H_int_uv,
                               const std::vector<Real>& y_node,
                               const std::vector<Real>& a_node);
