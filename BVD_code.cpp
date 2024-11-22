// © 2024 Hiro Wakimura, Feng Xiao
// MUSCL-THINC-BVD scheme
//------------------------------------------------------------

#include<iostream>
#include<fstream>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#define _USE_MATH_DEFINES
#include<cmath>
#include<algorithm>
#include<vector>
#include<valarray>
#include<iomanip>

#define eps 1.0e-15
#define GNUPLOT "gnuplot -persist"

using std::cin;
using std::cout;
using std::isnan;
using std::vector;
using std::endl;
using std::string;

typedef std::vector<int> vec1i;
typedef std::vector<vec1i> vec2i;
typedef std::vector<vec2i> vec3i;
typedef std::vector<double> vec1d;
typedef std::vector<vec1d> vec2d;
typedef std::vector<vec2d> vec3d;

// global variables
int problem_type;
int scheme_type;
string scheme_name;
double gamma_;
int num_var;
int nx, ng, NX;
int ns;
double x_range[2], dx;
vec1d xc, xb;
double t, t_end, dt, dt_last;
int t_step;
vec3d U;
double beta, T1_THINC;
int BVD, BVD_s;
vec3i BVD_active;
vec3d W_L, W_R;
vec2d F;
double CFL;
int k, RK_stage;
vec2d RK_alpha;
vec1d RK_beta;

void parameter();
void initial_condition();
void initialize_BVD_active();
void boundary_condition();
void reconstruction();
void eigen_vectors_Euler_cons_L_x(vec2d&, int, int);
void MUSCL(double*, double*, const vec1d&);
double Phi_MUSCL(double);
void THINC(double*, double*, const vec1d&);
void BVD_selection();
void eigen_vectors_Euler_cons_R_x(int, int, int);
void Riemann_solver_Euler_HLLC(double*);
void cal_dt(double);
void update();
void output_result();
void plot_result();
double sign(double);

int main(void){
    
    int i, m, per;
    double percent, MWS_x;
    
    parameter();
    
    initial_condition();
    
    t = 0.; t_step = 0; per = 1; dt_last = 0.; dt = 0.0; // initialization
    while (1){
        // adjust value of dt in final time step
        if (t + dt > t_end) dt_last = t_end - t;
        
        if (BVD > 1) initialize_BVD_active();
        
        for (k = 0; k < RK_stage; k++){
            // set values in ghost cells
            boundary_condition();
            
            // interpolate cell boundary values
            reconstruction();
            
            // calculate numerical flux
            Riemann_solver_Euler_HLLC(&MWS_x);
            
            // update numerical solution
            cal_dt(MWS_x);
            update();

        }
        t += dt;
        t_step++;
        
        // display of calculation progress
        percent = t / t_end * 100.0;
        if (percent >= per * 10 - eps){
            cout << per * 10 << "%  ";
            fflush(stdout);
            per++;
        }
        
        if (t >= t_end) break;
    }
    
    cout << "\nt_step = " << t_step << ", t = " << t << endl;
    
    output_result();
    plot_result();
    
    return 0;
}

void parameter(){
    int i;
    
    // set benchmark test
    cout << "Press number of benchmark test." << endl;
    cout << "1: Sod problem, 2: Le Blanc problem" << endl;
    cin >> problem_type;
    
    // set numerical scheme
    cout << "Press number of scheme." << endl;
    cout << "1: MUSCL, 2: THINC, 3: MUSCL-THINC-BVD" << endl;
    cin >> scheme_type;
    
    if      (scheme_type == 1) scheme_name = "MUSCL";
    else if (scheme_type == 2) scheme_name = "THINC";
    else if (scheme_type == 3) scheme_name = "MUSCL-THINC-BVD";
    else {
        cout << "Not implemented." << endl;
        exit(0);
    }
    
    if (scheme_name == "MUSCL" || scheme_name == "THINC") BVD = 1; // number of cacndidate reconstruction schemes
    else if (scheme_name == "MUSCL-THINC-BVD") BVD = 2;
    
    num_var = 3; // number of variables in Euler equation (rho, rhou, rhoE)
    
    BVD_s = BVD - 1; // number of cells to be expanded outside the computational domain for the BVD scheme
    ns = 3; // number of cells in stencil for reconstruction
    ng = 2 + BVD_s; // number of ghost cells
    
    nx = 200; // number of cell in computational domain
    NX = nx + ng * 2; // total number of cell including ghost cell
    
    // Sod problem
    if (problem_type == 1){
        x_range[0] = 0.0; // x_coordinate at left boundary of computational domain
        x_range[1] = 1.0; // x_coordinate at right boundary of computational domain
    }
    // Le Blanc problem
    else if (problem_type == 2){
        x_range[0] = 0.0; // x_coordinate at left boundary of computational domain
        x_range[1] = 9.0; // x_coordinate at right boundary of computational domain
    }
    
    dx = (x_range[1] - x_range[0]) / nx; // cell length
    xc = vec1d(nx); // x-coordinate at cell center
    for (i = 0; i < nx; i++){
        xc[i] = x_range[0] + dx / 2.0 + dx * i;
    }
    xb = vec1d(nx + 1); // x-coordinate at cell boundary
    for (i = 0; i < nx + 1; i++){
        xb[i] = x_range[0] + dx * i;
    }
    
    // coefficients of 3-stage 3rd-order Runge Kutta method
    RK_stage = 3;
    RK_alpha = vec2d(RK_stage, vec1d(RK_stage));
    RK_alpha[0][0] = 1.0; RK_alpha[0][1] = 0.0; RK_alpha[0][2] = 0.0;
    RK_alpha[1][0] = 0.75; RK_alpha[1][1] = 0.25; RK_alpha[1][2] = 0.0;
    RK_alpha[2][0] = 1.0/3.0; RK_alpha[2][1] = 0.0; RK_alpha[2][2] = 2.0/3.0;
    RK_beta = vec1d(RK_stage);
    RK_beta[0] = 1.0; RK_beta[1] = 0.25; RK_beta[2] = 2.0/3.0;
    // RK_alpha = {{1.0, 0.0, 0.0},
    //             {0.75, 0.25, 0.0},
    //             {1.0/3.0, 0.0, 2.0/3.0}};
    // RK_beta = {1.0, 0.25, 2.0/3.0};
    
    CFL = 0.5; // Courant number
    
    // allocate memory
    U = vec3d(RK_stage, vec2d(num_var, vec1d(NX, 0.0))); // conservative variables
    W_L = vec3d(BVD, vec2d(num_var, vec1d(NX + 1, 0.0))); // reconstructed left-side cell boundary value
    W_R = vec3d(BVD, vec2d(num_var, vec1d(NX + 1, 0.0))); // reconstructed right-side cell boundary value
    BVD_active = vec3i(RK_stage, vec2i(num_var, vec1i(NX, 0))); // record cells where MUSCL has been replaced by THINC
    F = vec2d(num_var, vec1d(NX + 1, 0.0)); // flux vector
    
    // for THINC scheme
    beta = 1.6; // gradient parameter
    T1_THINC = tanh(beta / 2.0); // T1 should be precalculated for reducing computational cost
}

void initial_condition(){
    double rho_L, rho_R, u_L, u_R, p_L, p_R, x_border;
    int i;
    
    // Sod shock tube problem
    if (problem_type == 1){
        rho_L = 1.0; rho_R = 0.125; // density
        u_L = 0.0; u_R = 0.0; // velocity
        p_L = 1.0; p_R = 0.1; // pressure
        x_border = 0.5; // border of initial condition
        gamma_ = 1.4; // specific heat ratio
        t_end = 0.25; // time to end calculation
    }
    // Le Blanc problem
    else if (problem_type == 2){
        rho_L = 1.0; rho_R = 0.001; // density
        u_L = 0.0; u_R = 0.0; // velocity
        p_L = 0.1*(2./3.); p_R = 1.e-10*(2./3.); // pressure
        x_border = 3.0; // border of initial condition
        gamma_ = 5./3.; // specific heat ratio
        t_end = 6.0; // time to end calculation
    }

    for (i = 0; i < nx; i++){
        if (xc[i] < x_border){
            U[0][0][ng + i] = rho_L; // rho
            U[0][1][ng + i] = rho_L * u_L; // rhou
            U[0][2][ng + i] = p_L / (gamma_ - 1.0) + 0.5 * rho_L * u_L * u_L; //rhoE
        }
        else {
            U[0][0][ng + i] = rho_R; // rho
            U[0][1][ng + i] = rho_R * u_R; // rhou
            U[0][2][ng + i] = p_R / (gamma_ - 1.0) + 0.5 * rho_R * u_R * u_R; // rhoE
        }
    }
}

void initialize_BVD_active(){
    int i, m;
    for (i = 0; i < NX; i++){
        for (k = 0; k < RK_stage; k++){
            for (m = 0; m < num_var; m++){
                BVD_active[k][m][i] = 0;
            }
        }
    }
}

void boundary_condition(){
    int i, m;
    
    // outflow condition
    for (i = 0; i < ng; i++){
        for (m = 0; m < num_var; m++){
            U[k][m][i] = U[k][m][ng];
            U[k][m][NX - 1 - i] = U[k][m][nx + ng - 1];
        }
    }
}

void reconstruction(){
    int i, m, xi;
    double q_L, q_R;
    vec2d stencil(num_var, vec1d(ns + 1));
    
    for (i = ng - BVD_s; i < ng + nx + 1 + BVD_s; i++){ // each cell boundary
        for (m = 0; m < num_var; m++){ // each variables
            for (xi = 0; xi < ns + 1; xi++){
                stencil[m][xi]=U[k][m][i - 1 - (ns - 1) / 2 + xi]; // stencil for reconstruction
            }
        }
        
        eigen_vectors_Euler_cons_L_x(stencil, i - 1, i); // multiply stencil by left eigenvector
        
        // calculate cell boundary values
        // q_L: left-side cell boundary value at x_{i-1/2}
        // q_R: right-side cell boundary value at x_{i-1/2}
        for (m = 0; m < num_var; m++){
            
            if (scheme_type == 1){
                MUSCL(&q_L, &q_R, stencil[m]);
                W_L[0][m][i] = q_L;
                W_R[0][m][i] = q_R;
            }
            else if (scheme_type == 2){
                THINC(&q_L, &q_R, stencil[m]);
                W_L[0][m][i] = q_L;
                W_R[0][m][i] = q_R;
            }
            else if (scheme_type == 3){
                MUSCL(&q_L, &q_R, stencil[m]);
                W_L[0][m][i] = q_L;
                W_R[0][m][i] = q_R;
                THINC(&q_L, &q_R, stencil[m]);
                W_L[1][m][i] = q_L;
                W_R[1][m][i] = q_R;
            }
            
        }
    }
    
    if (BVD > 1){ // MUSCL or THINC scheme is selected at each cell following BVD selection algorithm
        BVD_selection();
    }
    
    for (i = ng; i < ng + nx + 1; i++){ // each cell boundary
        eigen_vectors_Euler_cons_R_x(i - 1, i, i); // multiply cell boundary values by right eigenvector
    }
}

void MUSCL(double *qL, double *qR, const vec1d& q){
    
    double Phi_L, Phi_R, r_i;
    
    if (fabs(q[2] - q[1]) > eps){ // avoid zero-division
        r_i = (q[1] - q[0]) / (q[2] - q[1]);
        Phi_L = Phi_MUSCL(r_i);
        *qL = q[1] + 0.5 * Phi_L * (q[2] - q[1]); // left-side cell boundary value
        
        r_i = (q[3] - q[2]) / (q[2] - q[1]);
        Phi_R = Phi_MUSCL(r_i);
        *qR = q[2] - 0.5 * Phi_R * (q[2] - q[1]); // right-side cell boundary value
    }
    else {
        *qL = q[1];
        *qR = q[2];
    }

}

double Phi_MUSCL(double r_i){
    return (r_i + fabs(r_i)) / (1.0 + fabs(r_i)); // van leer limiter
}

void THINC(double *qL, double *qR, const vec1d& q){
    double q_p, q_m, alpha, eps_THINC = 1e-15, T1, T2;
    
    T1 = T1_THINC;
    
    // left-side cell boundary value
    if ((q[1] - q[0]) * (q[2] - q[1]) > eps_THINC){
        q_p = (q[0] + q[2]) / 2.0;
        q_m = (q[2] - q[0]) / 2.0;
        alpha = (q[1] - q_p) / q_m;
        T2 = tanh(alpha * beta / 2.0);
        *qL = q_p + q_m * (T1 + T2 / T1) / (1.0 + T2);
    }
    else {
        *qL = q[1];
    }
    // right-side cell boundary value
    if ((q[2] - q[1]) * (q[3] - q[2]) > eps_THINC){
        q_p = (q[1] + q[3]) / 2.0;
        q_m = (q[3] - q[1]) / 2.0;
        alpha = (q[2] - q_p) / q_m;
        T2 = tanh(alpha * beta / 2.0);
        *qR = q_p - q_m * (T1 - T2 / T1) / (1.0 - T2);
    }
    else {
        *qR = q[2];
    }
}

void BVD_selection(){
    int i, m;
    double TBV[2];
    
    for (m = 0; m < num_var; m++){
        // calculate TBV (Total Boundary Variation) and compare
        for (i = ng - BVD_s; i < ng + nx + BVD_s; i++){ // each cell
            TBV[0] = fabs(W_L[0][m][i] - W_R[0][m][i]) + fabs(W_L[0][m][i + 1] - W_R[0][m][i + 1]); // TBV of MUSCL
            TBV[1] = fabs(W_L[1][m][i] - W_R[1][m][i]) + fabs(W_L[1][m][i + 1] - W_R[1][m][i + 1]); // TBV of THINC
            if (TBV[0] > TBV[1]){
                BVD_active[k][m][i] = 1;
            }
        }
        
        // select numerical scheme which has smaller TBV value
        for (i = ng - BVD_s; i < ng + nx + BVD_s; i++){ // each cell
            if (BVD_active[k][m][i] == 1){
                W_L[0][m][i + 1] = W_L[1][m][i + 1];
                W_R[0][m][i] = W_R[1][m][i];
            }
        }
    }
}

void eigen_vectors_Euler_cons_L_x(vec2d &q, int IL, int IR){
    int m, xi;
    double rhoL, rhoR, srL, srR, uL, uR, HL, HR, ub, Hb, cb, b1, b2, L_x[3][3];
    vec2d q_copy = q;
    
    // calculate Roe average
    rhoL = U[k][0][IL];
    rhoR = U[k][0][IR];
    uL = U[k][1][IL] / rhoL;
    uR = U[k][1][IR] / rhoR;
    HL = gamma_ * U[k][2][IL] / rhoL - (gamma_ - 1.0) * 0.5 * uL * uL;
    HR = gamma_ * U[k][2][IR] / rhoR - (gamma_ - 1.0) * 0.5 * uR * uR;
    srL = sqrt(rhoL);
    srR = sqrt(rhoR);
    ub = (srL * uL + srR * uR) / (srL + srR);
    Hb = (srL * HL + srR * HR) / (srL + srR);
    cb = sqrt((gamma_ - 1.0) * (Hb - 0.5 * ub * ub));
    
    // calculate left eigenvectors
    b2 = (gamma_ - 1.0) / (cb * cb);
    b1 = 0.5 * ub * ub * b2;
    L_x[0][0] = 0.5 * (b1 + ub / cb); L_x[0][1] = - 0.5 * (1.0 / cb + b2 * ub); L_x[0][2] = 0.5 * b2;
    L_x[1][0] = 1.0 - b1;             L_x[1][1] = b2 * ub;                      L_x[1][2] = - b2;
    L_x[2][0] = 0.5 * (b1 - ub / cb); L_x[2][1] = 0.5 * (1.0 / cb - b2 * ub);   L_x[2][2] = 0.5 * b2;
    
    // multiply stencil by left eigenvectors
    for (xi = 0; xi < ns + 1; xi++){
        for (m = 0; m < num_var; m++){
            q[m][xi] = L_x[m][0] * q_copy[0][xi] + L_x[m][1] * q_copy[1][xi] + L_x[m][2] * q_copy[2][xi];
        }
    }
}

void eigen_vectors_Euler_cons_R_x(int IL, int IR, int Ix){
    int m;
    double rhoL, rhoR, srL, srR, uL, uR, HL, HR, ub, Hb, cb, b1, b2, R_x[3][3];
    double VL_copy[3], VR_copy[3];
    
    // calculate Roe average
    rhoL = U[k][0][IL];
    rhoR = U[k][0][IR];
    uL = U[k][1][IL] / rhoL;
    uR = U[k][1][IR] / rhoR;
    HL = gamma_ * U[k][2][IL] / rhoL - (gamma_ - 1.0) * 0.5 * uL * uL;
    HR = gamma_ * U[k][2][IR] / rhoR - (gamma_ - 1.0) * 0.5 * uR * uR;
    srL = sqrt(rhoL);
    srR = sqrt(rhoR);
    ub = (srL * uL + srR * uR) / (srL + srR);
    Hb = (srL * HL + srR * HR) / (srL + srR);
    cb = sqrt((gamma_ - 1.0) * (Hb - 0.5 * ub * ub));
    
    // calculate right eigenvectors
    R_x[0][0] = 1.0;          R_x[0][1] = 1.0;           R_x[0][2] = 1.0;
    R_x[1][0] = ub - cb;      R_x[1][1] = ub;            R_x[1][2] = ub + cb;
    R_x[2][0] = Hb - ub * cb; R_x[2][1] = 0.5 * ub * ub; R_x[2][2] = Hb + ub * cb;
    
    // multiply cell boundary values by right eigenvectors
    for (m = 0; m < num_var; m++){
        VL_copy[m] = W_L[0][m][Ix];
        VR_copy[m] = W_R[0][m][Ix];
    }
    for (m = 0; m < num_var; m++){
        W_L[0][m][Ix] = (R_x[m][0] * VL_copy[0] + R_x[m][2] * VL_copy[2]) + R_x[m][1] * VL_copy[1];
        W_R[0][m][Ix] = (R_x[m][0] * VR_copy[0] + R_x[m][2] * VR_copy[2]) + R_x[m][1] * VR_copy[1];
    }
}

void Riemann_solver_Euler_HLLC(double *MWS_x){
    int i;
    double mws_x = 0.0;
    
    double rho_L, rhou_L, rhoE_L, u_L, p_L, c_L, rho_R, rhou_R, rhoE_R, u_R, p_R, c_R;
    double f_rho_L, f_rhou_L, f_rhoE_L, f_rho_R, f_rhou_R, f_rhoE_R;
    double rho_bar, c_bar, p_star, rho_star, q_L, q_R;
    double S_L, S_R, S_star;
    double rho_Lstar, rhou_Lstar, rhoE_Lstar, rho_Rstar, rhou_Rstar, rhoE_Rstar;
    double S_ratio_L, S_ratio_R;
    double S_star_plus, S_star_minus, S_L_minus, S_R_plus;
    
    for (i = ng; i < ng + nx + 1; i++){ // each cell boundary
        // conservative variables
        rho_L = W_L[0][0][i]; rho_R = W_R[0][0][i];
        rhou_L = W_L[0][1][i]; rhou_R = W_R[0][1][i];
        rhoE_L = W_L[0][2][i]; rhoE_R = W_R[0][2][i];
        // primitive variables
        u_L = rhou_L / rho_L; u_R = rhou_R / rho_R;
        p_L = (gamma_ - 1.0) * (rhoE_L - 0.5 * rhou_L * u_L); p_R = (gamma_ - 1.0) * (rhoE_R - 0.5 * rhou_R * u_R);
        c_L = sqrt(gamma_ * p_L / rho_L); c_R = sqrt(gamma_ * p_R / rho_R);
        // flux
        f_rho_L = rhou_L; f_rho_R = rhou_R;
        f_rhou_L = rhou_L * u_L + p_L; f_rhou_R = rhou_R * u_R + p_R;
        f_rhoE_L = (rhoE_L + p_L) * u_L; f_rhoE_R = (rhoE_R + p_R) * u_R;

        // PVRS (Primitive Variable Riemann Solver)
        rho_bar = 0.5 * (rho_L + rho_R);
        c_bar = 0.5 * (c_L + c_R);
        p_star = fmax(0.0, 0.5 * (p_L + p_R) - 0.5 * (u_R - u_L) * rho_bar * c_bar);
        if (p_star < p_L) q_L = 1.0;
        else q_L = sqrt(1.0 + (gamma_ + 1.0) / (2.0 * gamma_) * (p_star / p_L - 1.0));
        S_L = u_L - c_L * q_L;
        if (p_star < p_R) q_R = 1.0;
        else q_R = sqrt(1.0 + (gamma_ + 1.0) / (2.0 * gamma_) * (p_star / p_R-1.0));
        S_R = u_R + c_R * q_R;
        
        // Max Wave Speed
        mws_x = fmax(mws_x, fmax(fabs(S_L), fabs(S_R)));

        // speed of contact discontinuity
        S_star = ((p_R - p_L) + (rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)))
                  /
                  (rho_L * (S_L - u_L) - rho_R * (S_R - u_R));

        // conservative variables at intermediate region
        S_ratio_L = (S_L - u_L) / (S_L - S_star); S_ratio_R = (S_R - u_R) / (S_R - S_star);
        rho_Lstar = rho_L * S_ratio_L;
        rho_Rstar = rho_R * S_ratio_R;
        rhou_Lstar = rho_Lstar * S_star;
        rhou_Rstar = rho_Rstar * S_star;
        rhoE_Lstar = rho_Lstar * (rhoE_L / rho_L + (S_star - u_L) * (S_star + p_L / rho_L / (S_L - u_L)));
        rhoE_Rstar = rho_Rstar * (rhoE_R / rho_R + (S_star - u_R) * (S_star + p_R / rho_R / (S_R - u_R)));

        // calculate flux
        S_star_plus = 0.5 * (1.0 + sign(S_star));
        S_star_minus = 0.5 * (1.0 - sign(S_star));
        S_L_minus = fmin(S_L , 0.0);
        S_R_plus = fmax(S_R , 0.0);
        F[0][i] = S_star_plus * (f_rho_L + S_L_minus * (rho_Lstar - rho_L))
                 + S_star_minus * (f_rho_R + S_R_plus * (rho_Rstar - rho_R));
        F[1][i] = S_star_plus * (f_rhou_L + S_L_minus * (rhou_Lstar - rhou_L))
                 + S_star_minus * (f_rhou_R + S_R_plus * (rhou_Rstar - rhou_R));
        F[2][i] = S_star_plus * (f_rhoE_L + S_L_minus * (rhoE_Lstar - rhoE_L))
                 + S_star_minus * (f_rhoE_R + S_R_plus * (rhoE_Rstar - rhoE_R));
    }
    *MWS_x = mws_x;
}

void cal_dt(double MWS_x){
    
    // calculate dt
    if (dt_last > 0.0){
        dt = dt_last;
    }
    else {
        dt = CFL * dx / (MWS_x + eps);
    }
}

void update(){
    int i, m, s, k_next = (k + 1) % RK_stage;
    double L, rho, rhou, rhoE, u, p;
    
    for (i = ng; i < ng + nx; i++){
        for (m = 0; m < num_var; m++){
            // calculate spatial discrete operator
            L = -(F[m][i + 1] - F[m][i]) / dx;
            
            // obtain numerical solution at next sub-step in Runge Kutta method
            U[k_next][m][i] = RK_alpha[k][0] * U[0][m][i];
            for (s = 1; s <= k; s++){
                U[k_next][m][i] += RK_alpha[k][s] * U[s][m][i];
            }
            U[k_next][m][i] += RK_beta[k] * L * dt;
        }
        
        // check positivity
        rho = U[k_next][0][i];
        rhou = U[k_next][1][i];
        rhoE = U[k_next][2][i];
        u = rhou / rho;
        p = (gamma_ - 1.0) * (rhoE - 0.5 * rho * u * u);
        if (rho <= 0.0 || p <= 0.0){
            cout << "rho or p < 0" << endl;
            cout << "rho = " << rho << ", p = " << p << endl;
            getchar();
        }
    }
}

void output_result(){
    int i, m;
    double rho, rhou, rhoE, u, p;
    int BVD_func;
    
    std::ofstream file_result("./result.csv");
        
    for (i = 0; i < nx; i++){
        
        rho = U[0][0][ng + i];
        rhou = U[0][1][ng + i];
        rhoE = U[0][2][ng + i];
        u = rhou / rho;
        p = (gamma_ - 1.0) * (rhoE - 0.5 * rhou * u);
        
        file_result << std::setprecision(15);
        file_result << xc[i]<<" ";
        file_result
            << rho << " "
            << u << " "
            << p << " ";
        BVD_func = 0;
        for (k = 0; k < RK_stage; k++){
            for (m = 0; m < num_var; m++){
                BVD_func = fmax(BVD_func, BVD_active[k][m][ng + i]);
            }
        }
        file_result << BVD_func << " ";
        file_result << "\n";
    }
    
    file_result.close();
}

void plot_result(){
    
    FILE *gp;
    string variable_name;
    double yr_min, yr_max;
    
    int plot_var = 1; // 1: density, 2: velocity, 3: pressure, 4: BVD_active
    if      (plot_var == 1) variable_name = "density";
    else if (plot_var == 2) variable_name = "velocity";
    else if (plot_var == 3) variable_name = "pressure";
    else if (plot_var == 4) variable_name = "BVD_active";
    
    if (problem_type == 1){ // Sod problem
        yr_min = 0.0; yr_max = 1.2;
    }
    else if (problem_type == 2){ // Le Blanc problem
        yr_min = 1.0e-3; yr_max = 1.0;
    }
    
    #if defined(_WIN32) || defined(_WIN64)
        // for windows pc
        gp = _popen(GNUPLOT, "w");
        if (!gp){
            cout << "unable to start gnuplot" << endl;
        }
        else {
            // plot numerical result using gnuplot
            fprintf(gp, "set term wxt 1\n");
            fprintf(gp, "set size ratio 1\n");
            fprintf(gp, "set yr[%f:%f]\n", yr_min, yr_max);
            fprintf(gp, "set xl \"x\"\n");
            fprintf(gp, "set yl \"%s\"\n", variable_name.c_str());
            fprintf(gp, "set title \"1D Euler\"\n");
            fprintf(gp, "plot \"./result.csv\" using %d:%d title \"%s\" w lp lt 7 ps 1\n", 1, plot_var + 1, scheme_name.c_str());
            fflush(gp);
            _pclose(gp);
        }
        
    #elif defined(__APPLE__) && defined(__MACH__)
        // for mac pc
        gp = popen(GNUPLOT, "w");
        if (!gp){
            cout << "unable to start gnuplot" << endl;
        }
        else {
            // plot numerical result using gnuplot
            // fprintf(gp, "set term wxt 1\n");
            fprintf(gp, "set size ratio 1\n");
            fprintf(gp, "set yr[%f:%f]\n", yr_min, yr_max);
            fprintf(gp, "set xl \"x\"\n");
            fprintf(gp, "set yl \"%s\"\n", variable_name.c_str());
            fprintf(gp, "set title \"1D Euler\"\n");
            fprintf(gp, "plot \"./result.csv\" using %d:%d title \"%s\" w lp lt 7 ps 1\n", 1, plot_var + 1, scheme_name.c_str());
            fflush(gp);
            pclose(gp);
        }
        
    #elif defined(__linux__)
        // for linux pc
        gp = popen(GNUPLOT, "w");
        if (!gp){
            cout << "unable to start gnuplot" << endl;
        }
        else {
            // plot numerical result using gnuplot
            fprintf(gp, "set term wxt 1\n");
            fprintf(gp, "set size ratio 1\n");
            // fprintf(gp, "set logscale y\n");
            fprintf(gp, "set yr[%f:%f]\n", yr_min, yr_max);
            fprintf(gp, "set xl \"x\"\n");
            fprintf(gp, "set yl \"%s\"\n", variable_name.c_str());
            fprintf(gp, "set title \"1D Euler\"\n");
            fprintf(gp, "plot \"./result.csv\" using %d:%d title \"%s\" w lp lt 7 ps 1\n", 1, plot_var + 1, scheme_name.c_str());
            fflush(gp);
            pclose(gp);
        }
        
    #endif
        
}

double sign(double a){
    if (a > 0.0)      return 1.0;
    else if (a < 0.0) return -1.0;
    else              return 0.0;
}