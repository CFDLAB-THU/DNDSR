#pragma once

#include "DNDS/HardEigen.hpp"
#include "DNDS/Defines.hpp"

namespace DNDS::Linear
{

    template <class TDATA>
    class GMRES_LeftPreconditioned
    {
        uint32_t kSubspace;

        std::vector<TDATA> Vs;
        TDATA V_temp;
        TDATA MLb;

    public:
        template <class Finit>
        GMRES_LeftPreconditioned(
            uint32_t NkSubspace,
            Finit &&finit = [](TDATA &) {}) : kSubspace(NkSubspace)
        {
            Vs.resize(kSubspace + 1);
            for (auto &i : Vs)
                finit(i);
            finit(V_temp);
            finit(MLb);
        }

        /**
         * @brief
         *
         * @tparam TFA
         * @tparam TFML
         * @tparam TFstop
         * @param FA  void FA(TDATA &x, TDATA &Ax)
         * @param FML void FML(TDATA &x, TDATA &MLx)
         * @param b   rhs
         * @param x   input and output
         * @param nRestart
         * @param FStop bool FStop(iRestart, res, resB)
         */
        template <class TFA, class TFML, class TFstop>
        bool solve(TFA &&FA, TFML &&FML, TDATA &b, TDATA &x, uint32_t nRestart, TFstop &&FStop)
        {
            FML(b, MLb); // MLb = ML * b
            real scale_MLb = MLb.norm2();
            Eigen::MatrixXd h;
            h.setZero(kSubspace + 1, kSubspace);
            uint32_t iRestart;
            for (iRestart = 0; iRestart <= nRestart; iRestart++)
            {
                FA(x, V_temp);                        // V_temp = A * x
                FML(V_temp, Vs[0]);                   // Vs[0] = ML * A * x
                Vs[0].addTo(MLb, -1.0);               // Vs[0] = ML * A * x - ML * b = -r
                real beta = Vs[0].norm2();            // beta = norm2(r)
                if (FStop(iRestart, beta, scale_MLb)) // see if converge
                    break;
                if (std::abs(beta) < verySmallReal || // beta is floating-point negligible
                    (iRestart == nRestart))
                    break;
                Vs[0] *= -1.0 / beta; // Vs[0] = r/norm2(r)

                uint32_t j = 0;
                for (; j < kSubspace; j++) // Arnoldi
                {
                    FA(Vs[j], V_temp);      // V_temp = A * Vs[j]
                    FML(V_temp, Vs[j + 1]); // Vs[j + 1] = ML * A * Vs[j]
                    for (uint32_t i = 0; i <= j; i++)
                    {
                        // Gram-Schmidt, calculate projected lengths
                        h(i, j) = Vs[j + 1].dot(Vs[i]); // h(i, j) = dot(ML * A * Vs[j],Vs[i])
                    }
                    for (uint32_t i = 0; i <= j; i++)
                    {
                        // Gram-Schmidt, subtract projections
                        Vs[j + 1].addTo(Vs[i], -h(i, j)); // Vs[j + 1] = ML * A * Vs[j] - sum_{i=0,1,...j}(h(i,j) *  Vs[i])
                    }
                    h(j + 1, j) = Vs[j + 1].norm2(); //
                    if (h(j + 1, j) < scale_MLb * 1e-32)
                    {
                        std::cout << "early stop" << std::endl;
                        break;
                    }
                    Vs[j + 1] *= 1.0 / h(j + 1, j); // normalize
                }
                // std::cout << beta << std::endl;

                Eigen::VectorXd eBeta;
                eBeta.resize(j + 1);
                eBeta.setZero();
                eBeta(0) = beta; // eBeta = e_1 * beta
                if(j < 1)
                    break;

                { // the QR method
                  //  auto QR = h.colPivHouseholderQr();
                  //  QR.setThreshold(std::sqrt(scale_MLb) * 1e-32);
                  //  if (QR.rank() != h.rows())
                  //      DNDS_assert_info(false, "GMRES not good");
                  //  Eigen::VectorXd y = QR.solve(eBeta);
                }

                // auto sol = h(Eigen::seq(0, j + 1 - 1), Eigen::seq(0, j - 1)).bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
                // Eigen::VectorXd y = sol.solve(eBeta); //! warning: this gmres is dumb and do not lucky stop
                Eigen::MatrixXd y;
                auto rank = HardEigen::EigenLeastSquareSolve(h(Eigen::seq(0, j + 1 - 1), Eigen::seq(0, j - 1)), eBeta, y);

                // int rank;
                // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                // if (rank == 0)
                //     std::cout << h << std::endl;

                for (uint32_t jj = 0; jj < j; jj++) // x = V(:, 0,1,2,...kSubspace-1) * y
                {
                    x.addTo(Vs[jj], y(jj));
                    // std::cout << iRestart << "::" << Vs[j].transpose() << "::" << y(j) << std::endl;
                }
                if (rank < h.cols())
                {
                   
                    break; // do not restart
                }
            }
            return iRestart < nRestart;
        }
    };
}
