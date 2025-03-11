#pragma once
#include <string>

// #define USE_ECCENTRIC_COMB_POW_2

// #define USE_LOCAL_COORD_CURVILINEAR
// #define PRINT_EVERY_VR_JACOBI_ITER_INCREMENT

// #define USE_ISOTROPIC_OPTHQM

// #define USE_FLUX_BALANCE_TERM // TODO: decide how to flux balance for moving mesh
// TODO: this option has been deleted, re-implement in the evaluate rhs code

#define USE_ENTROPY_FIXED_LAMBDA_IN_SA

#define USE_FIX_ZERO_SA_NUT_AT_WALL

// #define USE_SIGN_MINUS_AT_ROE_M4_FLUX

#define USE_FIRST_ORDER_VISCOUS_WALL_DELTA_IN_VR_WEIGHT

#define USE_FIRST_ORDER_WALL_DIST

// #define USE_MG_O1_NO_VISCOUS

// #define USE_MG_O1_LLF_FLUX

#define USE_NS_SA_NEGATIVE_MODEL

// #define USE_NS_SA_NUT_REDUCED_ORDER

// #define USE_NS_SA_ALLOW_NEGATIVE_MEAN

#define USE_ABS_VELO_IN_ROTATION

/*-------------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------*/

static const std::string DNDS_Experimentals_State = std::string("DNDS_Experimentals ")
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
#ifdef USE_SIGN_MINUS_AT_ROE_M4_FLUX
                                                    + " USE_SIGN_MINUS_AT_ROE_M4_FLUX "
#endif
#ifdef USE_FIRST_ORDER_VISCOUS_WALL_DELTA_IN_VR_WEIGHT
                                                    + " USE_FIRST_ORDER_VISCOUS_WALL_DELTA_IN_VR_WEIGHT "
#endif
#ifdef USE_FIRST_ORDER_WALL_DIST
                                                    + " USE_FIRST_ORDER_WALL_DIST "
#endif
#ifdef USE_MG_O1_NO_VISCOUS
                                                    + " USE_MG_O1_NO_VISCOUS "
#endif
#ifdef USE_MG_O1_LLF_FLUX
                                                    + " USE_MG_O1_LLF_FLUX "
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
