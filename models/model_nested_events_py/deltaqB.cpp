#include "amici/symbolic_functions.h"
#include "amici/defines.h"

#include <algorithm>

namespace amici {
namespace model_model_nested_events_py {

void deltaqB_model_nested_events_py(realtype *deltaqB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *dx, const int ip, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *x_old, const realtype *xB){
    const realtype dVirusdt_ = xdot[0];
    const realtype xdot_old0_ = xdot_old[0];
    const realtype xB0_ = xB[0];

    switch(ie) {
        case 2:
            switch(ip) {
                case 1:
                    deltaqB[0] = xB0_;
                    break;
                case 2:
                    deltaqB[0] = xB0_*(-dVirusdt_ + xdot_old0_);
                    break;
            }
            break;
    }
}

} // namespace model_model_nested_events_py
} // namespace amici
