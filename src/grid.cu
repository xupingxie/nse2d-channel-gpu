// Part of nse2d-channel-gpu: GPU DNS of 2D channel turbulence (vorticity-streamfunction).
// Wall-normal grid metrics (uniform, tanh, tabulated).
#include "grid.cuh"
#include "io.cuh"

// P.stretch: "none"|"tanh"|"table"; P.beta used by tanh; P.ytable used by table.
void build_y_metrics(const Grid& Gin, const Params& P, Grid& Gout) {
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

