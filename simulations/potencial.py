import sympy as sp
from scipy.optimize import fsolve
import pickle


def get_expressions(values, type='numeric', substituted_I=False):  
    X1, X2 = sp.symbols('X1 X2')
    tau, phi_p, phi_pp, phi_ppp, s0, c, g, phi_Ip = sp.symbols('tau phi_p phi_pp phi_ppp s0 c g phi_Ip')
    I_L, I_C, I_R, s_L, s_C, s_R, R = sp.symbols('I_L I_C I_R s_L s_C s_R R')
    x = sp.symbols('x', real=True)
    R_I, I0, I_I = sp.symbols('R_I I0 I_I', real=True)

    with open('U.pkl', 'rb') as f:
        U_simpl = pickle.load(f)
    with open('field_components.pkl', 'rb') as f:
        exprs = pickle.load(f)
    with open('phis.pkl', 'rb') as f:
        phis = pickle.load(f)

    F1, F2 = exprs['F1'], 3*exprs['F2']
    phi_X0 = phis['phi_X0']
    phi_prime_X0 = phis['phi_X0_p']
    phi_double_prime_X0 = phis['phi_X0_pp']
    phi_triple_prime_X0 = phis['phi_X0_ppp']
    phi_I = phis['phi_I']
    phi_I_prime = phis['phi_Ip']

    
    eq1 = s0*phi_prime_X0-1
    eq2 = R - phi_X0 
    eq3 = R_I - phi_I
    subs_eq1= {
        sym: values[sym.name]
        for sym in eq1.free_symbols
        if sym.name in values
    }
    subs_eq2= {
        sym: values[sym.name]
        for sym in eq2.free_symbols
        if sym.name in values
    }
    subs_eq3= {
        sym: values[sym.name]
        for sym in eq3.free_symbols
        if sym.name in values
    }

    eq1 = eq1.subs(subs_eq1)
    eq2 = eq2.subs(subs_eq2)
    eq3 = eq3.subs(subs_eq3)
    F_so = sp.lambdify(
        (R, R_I, I0),
        (eq1, eq2, eq3),
        'numpy'
    )
    

    def fun(vars):
        R_val, R_I_val, I0_val = vars
        return F_so(R_val, R_I_val, I0_val)

    x0 = [1/4, 1/4, 1/2]
    R_val, R_I_val, I0_val = fsolve(fun, x0)


    phi_p_val = sp.N(phi_prime_X0.subs({R: R_val, R_I: R_I_val, I0: I0_val, s0: values['s0'], c: values['c'], g: values['g']}))
    phi_pp_val = sp.N(phi_double_prime_X0.subs({R: R_val, R_I: R_I_val, I0: I0_val, s0: values['s0'], c: values['c'], g: values['g']}))
    phi_ppp_val = sp.N(phi_triple_prime_X0.subs({R: R_val, R_I: R_I_val, I0: I0_val, s0: values['s0'], c: values['c'], g: values['g']}))
    phi_Ip_val = sp.N(phi_I_prime.subs({R: R_val, R_I: R_I_val, I0: I0_val, s0: values['s0'], c: values['c'], g: values['g'], I_I: values['I_I']}))

    dict_I = {
        phi_p:   phi_p_val,
        phi_pp:  phi_pp_val,
        phi_ppp: phi_ppp_val,
        phi_Ip:  phi_Ip_val,
        s0:      values['s0'],
        I0:      I0_val,
        c:       values['c'],
        g:       values['g'],
        R:       R_val,
        I_L:     values['IL'],
        I_C:     values['IC'],
        I_R:     values['IR'],
        s_L:     values['sL'],
        s_C:     values['sC'],
        s_R:     values['sR'],
        R_I:     R_I_val,
    }
    dict_without_I = {
        phi_p:   phi_p_val,
        phi_pp:  phi_pp_val,
        phi_ppp: phi_ppp_val,
        phi_Ip:  phi_Ip_val,
        s0:      values['s0'],
        I0:      I0_val,
        c:       values['c'],
        g:       values['g'],
        R:       R_val,
        s_L:     values['sL'],
        s_C:     values['sC'],
        s_R:     values['sR'],
        R_I:     R_I_val,
    }


    if substituted_I:
        subs_dict =  dict_I
        lamdifyed = (X1, X2)
    else:
        subs_dict = dict_without_I
        lamdifyed = (X1, X2, I_L, I_C, I_R)

   

    H_sim = sp.hessian(U_simpl.subs(subs_dict), (X1, X2))
    U_sim = sp.simplify(U_simpl.subs(subs_dict))
    F1_sim = sp.simplify(F1.subs(subs_dict))
    F2_sim = sp.simplify(F2.subs(subs_dict))

    H_num  = sp.lambdify(lamdifyed, H_sim,  'numpy')
    U_num  = sp.lambdify(lamdifyed, U_sim,  'numpy')
    F1_num = sp.lambdify(lamdifyed, F1_sim, 'numpy')
    F2_num = sp.lambdify(lamdifyed, F2_sim, 'numpy')

    if(type == 'numeric'):
        return U_num, F1_num, F2_num, H_num
    elif(type == 'symbolic'):
        return U_sim, F1_sim, F2_sim, H_sim
    else: 
        raise ValueError("Invalid type. Use 'numeric' or 'symbolic'.")
    