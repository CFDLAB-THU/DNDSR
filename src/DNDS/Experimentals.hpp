#pragma once
#include <string>

#define USE_NORM_FUNCTIONAL
// #define USE_THIN_NORM_FUNCTIONAL

// #define USE_ECCENTRIC_COMB_POW_2

// #define USE_LOCAL_COORD_CURVILINEAR
// #define PRINT_EVERY_VR_JACOBI_ITER_INCREMENT

// #define USE_ISOTROPIC_OPTHQM

// #define USE_FLUX_BALANCE_TERM // TODO: decide how to flux balance for moving mesh

#define USE_ENTROPY_FIXED_LAMBDA_IN_SA

#define USE_FIX_ZERO_SA_NUT_AT_WALL

#define USE_TOTAL_REDUCED_ORDER_CELL

// #define USE_DISABLE_DIST_GRP_FIX_AT_WALL

// #define USE_NO_RIEMANN_ON_WALL

// #define USE_SIGN_MINUS_AT_ROE_M4_FLUX

#define USE_FIRST_ORDER_VISCOUS_WALL_DELTA_IN_VR_WEIGHT

#define USE_FIRST_ORDER_WALL_DIST

// #define USE_NS_SA_NEGATIVE_MODEL

// #define USE_NS_SA_NUT_REDUCED_ORDER

// #define USE_NS_SA_ALLOW_NEGATIVE_MEAN

#define USE_ABS_VELO_IN_ROTATION

/*-------------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------*/

static const std::string DNDS_Experimentals_State = std::string("DNDS_Experimentals ")
#ifdef USE_NORM_FUNCTIONAL
                                                    + " USE_NORM_FUNCTIONAL "
#endif
#ifdef USE_THIN_NORM_FUNCTIONAL
                                                    + " USE_THIN_NORM_FUNCTIONAL "
#endif
#ifdef USE_ECCENTRIC_COMB_POW_2
                                                    + " USE_ECCENTRIC_COMB_POW_2 "
#endif
#ifdef USE_LOCAL_COORD_CURVILINEAR
                                                    + " USE_LOCAL_COORD_CURVILINEAR "
#endif
#ifdef PRINT_EVERY_VR_JACOBI_ITER_INCREMENT
                                                    + " PRINT_EVERY_VR_JACOBI_ITER_INCREMENT "
#endif
#ifdef USE_FLUX_BALANCE_TERM
                                                    + " USE_FLUX_BALANCE_TERM "
#endif
#ifdef USE_ENTROPY_FIXED_LAMBDA_IN_SA
                                                    + " USE_ENTROPY_FIXED_LAMBDA_IN_SA "
#endif
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
                                                    + " USE_FIX_ZERO_SA_NUT_AT_WALL "
#endif
#ifdef USE_TOTAL_REDUCED_ORDER_CELL
                                                    + " USE_TOTAL_REDUCED_ORDER_CELL "
#endif
#ifdef USE_DISABLE_DIST_GRP_FIX_AT_WALL
                                                    + " USE_DISABLE_DIST_GRP_FIX_AT_WALL "
#endif
#ifdef USE_NO_RIEMANN_ON_WALL
                                                    + " USE_NO_RIEMANN_ON_WALL "
#endif
#ifdef USE_SIGN_MINUS_AT_ROE_M4_FLUX
                                                    + " USE_SIGN_MINUS_AT_ROE_M4_FLUX "
#endif
#ifdef USE_FIRST_ORDER_VISCOUS_WALL_DELTA_IN_VR_WEIGHT
                                                    + " USE_FIRST_ORDER_VISCOUS_WALL_DELTA_IN_VR_WEIGHT "
#endif
#ifdef USE_FIRST_ORDER_WALL_DIST
                                                    + " USE_FIRST_ORDER_WALL_DIST "
#endif
#ifdef USE_NS_SA_NEGATIVE_MODEL
                                                    + " USE_NS_SA_NEGATIVE_MODEL "
#endif
#ifdef USE_NS_SA_NUT_REDUCED_ORDER
                                                    + " USE_NS_SA_NUT_REDUCED_ORDER "
#endif
#ifdef USE_NS_SA_ALLOW_NEGATIVE_MEAN
                                                    + " USE_NS_SA_ALLOW_NEGATIVE_MEAN "
#endif
#ifdef USE_ISOTROPIC_OPTHQM
                                                    + " USE_ISOTROPIC_OPTHQM "
#endif
#ifdef USE_ABS_VELO_IN_ROTATION
                                                    + " USE_ABS_VELO_IN_ROTATION "
#endif
    ;
