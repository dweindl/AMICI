#include "amici/symbolic_functions.h"
#include "amici/defines.h"

#include <algorithm>

namespace amici {
namespace model_model_neuron_py {

void deltasx_model_neuron_py(realtype *deltasx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const int ip, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *sx, const realtype *stau, const realtype *tcl, const realtype *x_old){
    const realtype dvdt_ = xdot[0];
    const realtype dudt_ = xdot[1];
    const realtype xdot_old0_ = xdot_old[0];
    const realtype xdot_old1_ = xdot_old[1];
    const realtype sx0_ = sx[0];
    const realtype stau0_ = stau[0];

    switch(ie) {
        case 0:
            switch(ip) {
                case 0:
                case 1:
                    deltasx[0] = stau0_*xdot_old0_ + stau0_*(dvdt_ - xdot_old0_) - sx0_;
                    deltasx[1] = stau0_*(dudt_ - xdot_old1_);
                    break;
                case 2:
                    deltasx[0] = stau0_*xdot_old0_ + stau0_*(dvdt_ - xdot_old0_) - sx0_ - 1;
                    deltasx[1] = stau0_*(dudt_ - xdot_old1_);
                    break;
                case 3:
                    deltasx[0] = stau0_*xdot_old0_ + stau0_*(dvdt_ - xdot_old0_) - sx0_;
                    deltasx[1] = stau0_*(dudt_ - xdot_old1_) + 1;
                    break;
            }
            break;
    }
}

} // namespace model_model_neuron_py
} // namespace amici
