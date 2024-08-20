import numpy as np
from typing import Dict, List, Type

import sde

__known_models__: List[Type[sde.SDEModel]] = []


class Adak2020(sde.SDEModel):
    """From 10.1016/j.chaos.2020.110381"""

    name = 'adak_2020'
    variable_names = ['S', 'L', 'I', 'R', 'D', 'N']
    parameter_defaults = {
        'Lam': 1165.0,
        'lam': 0.18,
        'delta': 3.37E-5,
        'sigma': 0.4,
        'omega': 0.1924,
        'gamma': 0.095,
        'mu': 0.11
    }

    @staticmethod
    def _step(current_vals: np.ndarray,
              dt: float,
              parameters: Dict[str, float]) -> np.ndarray:
        result = np.zeros_like(current_vals)

        p = parameters
        s, l, i, r, d, n = current_vals.tolist()

        d_s = (p['Lam'] - p['delta'] * s) * dt
        d_l = - p['delta'] * l * dt
        d_i = - p['delta'] * i * dt
        d_r = - p['delta'] * r * dt

        d_li = p['omega'] * l * dt
        d_ir = p['gamma'] * i * dt
        d_id = p['mu'] * i * dt

        d_sl = s * i / n * (p['lam'] * dt + p['sigma'] * np.random.normal(0, np.sqrt(dt)))
        d_sl = min(d_sl, s + d_s)
        d_sl = max(d_sl, d_li - l - d_l)

        result[0] = d_s - d_sl
        result[1] = d_l + d_sl - d_li
        result[2] = d_i + d_li - d_ir - d_id
        result[3] = d_r + d_ir
        result[4] = d_id
        result[5] = sum(result[0:5])
        return current_vals + result

    def step(self,
             current_time: float,
             current_vals: np.ndarray,
             dt: float,
             parameters: Dict[str, float]) -> np.ndarray:
        return self._step(current_vals, dt, parameters)


__known_models__.append(Adak2020)


class Adak2020Fitted(sde.SDEModel):
    """From 10.1016/j.chaos.2020.110381"""

    name = 'adakfitted_2020'
    variable_names = Adak2020.variable_names
    parameter_defaults = {k: v for k, v in Adak2020.parameter_defaults.items() if k != 'lam'}

    @staticmethod
    def get_lam(current_time: float):
        lam_fitted = (0.542, 0.315, 0.192, 0.151, 0.164, 0.183, 0.271, 0.193)
        for i, lf in enumerate(lam_fitted):
            if current_time < 28 * (i + 1):
                return lam_fitted[i]
        return lam_fitted[-1]

    def step(self,
             current_time: float,
             current_vals: np.ndarray,
             dt: float,
             parameters: Dict[str, float]) -> np.ndarray:
        p = parameters.copy() if parameters is not None else {}
        p['lam'] = self.get_lam(current_time)
        return Adak2020._step(current_vals, dt, p)


__known_models__.append(Adak2020Fitted)


class Din2020(sde.SDEModel):
    """From 10.1016/j.chaos.2020.110036"""

    name = 'din_2020'
    variable_names = ['S', 'I', 'Q', 'N']
    parameter_defaults = {
        'lam': 0.3,
        'beta': 0.5,
        'mu0': 0.2,
        'mu1': 0.2,
        'gam': 0.3,
        'sig': 0.2,
        'mu': 0.1,
        'noise_S': 0.5,
        'noise_I': 0.4,
        'noise_Q': 0.2
    }

    def step(self,
             current_time: float,
             current_vals: np.ndarray,
             dt: float,
             parameters: Dict[str, float]) -> np.ndarray:
        p = parameters
        s, i, q, n = current_vals.tolist()
        result = np.zeros_like(current_vals)
        sqrt_dt = np.sqrt(dt)

        result[0] = (p['lam'] - p['beta'] * s * i / n - p['mu0'] * s) * dt + p['noise_S'] * s * np.random.normal(0, sqrt_dt)
        result[1] = (p['beta'] * s * i / n - (p['gam'] + p['mu1'] + p['mu0']) * i + p['sig'] * q) * dt + p['noise_I'] * i * np.random.normal(0, sqrt_dt)
        result[2] = (p['gam'] * i - (p['mu0'] + p['mu'] + p['sig']) * q) * dt + p['noise_Q'] * q * np.random.normal(0, sqrt_dt)
        result[3] = np.sum(result[0:3])
        return current_vals + result


__known_models__.append(Din2020)


class Faranda2020(sde.SDEModel):
    """From 10.1063/5.0015943"""

    name = 'faranda_2020'
    variable_names = ['S', 'E', 'I', 'R', 'N', 'lam', 'alp', 'gam', 'R0', 'C']
    parameter_defaults = {
        'lam_0': 1.0,
        'alp_0': 0.27,
        'gam_0': 0.37,
        'lam_std': 0.2,
        'noise_alp': 0.2,
        'noise_gam': 0.2,
        'int_time_0': -1.0,
        'int_val_0': 0.0,
        'int_time_1': -1.0,
        'int_val_1': 0.0
    }

    def step(self,
             current_time: float,
             current_vals: np.ndarray,
             dt: float,
             parameters: Dict[str, float]) -> np.ndarray:

        p = parameters
        s, e, i, r, n, lam, alp, gam, r0, c = current_vals.tolist()
        result = np.zeros_like(current_vals)

        int_time_0 = p['int_time_0']
        int_time_1 = p['int_time_1']

        lam_mean = p['lam_0']
        if 0 < int_time_1 < current_time:
            lam_mean *= p['int_val_1']
        elif 0 < int_time_0 < current_time:
            lam_mean *= p['int_val_0']
        lam_std = lam_mean * p['lam_std']
        alp_mean = p['alp_0']
        alp_std = alp_mean * p['noise_alp']
        gam_mean = p['gam_0']
        gam_std = gam_mean * p['noise_gam']

        d_s_e = lam * s * i / n * dt
        d_e_i = alp * e * dt
        d_i_r = gam * i * dt

        result[0] = - d_s_e
        result[1] = d_s_e - d_e_i
        result[2] = d_e_i - d_i_r
        result[3] = d_i_r
        result[5] = np.exp(np.random.normal(loc=np.log(lam_mean - lam_std * lam_std / 2), scale=lam_std)) - lam
        result[6] = np.random.normal(loc=alp_mean, scale=alp_std) - alp
        result[7] = np.random.normal(loc=gam_mean, scale=gam_std) - gam
        result[8] = (lam + result[5]) / (gam + result[7]) - r0
        result[9] = d_e_i

        return current_vals + result


__known_models__.append(Faranda2020)


class _Mamis2023SEIR(sde.SDEModel):
    """From 10.1098/rspa.2022.0568"""

    name = 'mamis_2023_seir_'
    variable_names = ['S', 'E', 'I', 'R', 'N', 'lam']
    parameter_defaults = {
        'alp': 1. / 3.5,
        'gam': 1. / 1.2,
        'mu': 1.85 / 1.2,
        'sig': 0.1 * 1.54,
        'tau': 7.
    }

    def _rates(self, _dt: float, _current_vals: np.ndarray, _parameters: Dict[str, float]):
        p = _parameters
        s, e, i, r, n, lam = _current_vals.tolist()
        result = np.zeros_like(_current_vals)

        d_s_e = lam / n * s * i
        d_e_i = p['alp'] * e
        d_i_r = p['gam'] * i
        xi = (lam - p['mu']) / p['sig']

        result[0] = - d_s_e
        result[1] = d_s_e - d_e_i
        result[2] = d_e_i - d_i_r
        result[3] = d_i_r
        result[4] = sum(result[0:4]) / _dt
        result[5] = (p['mu'] + p['sig'] * self.lam_noise(dt=_dt, xi=xi, **p) - lam) / _dt

        return result

    def step(self,
             current_time: float,
             current_vals: np.ndarray,
             dt: float,
             parameters: Dict[str, float]) -> np.ndarray:

        dt_done = 0.0
        result = np.array(current_vals)
        while dt_done < dt:
            dt_actual = dt - dt_done
            working = True
            rates = self._rates(dt_actual, result, parameters)
            while working:
                working = False
                for i in range(rates.shape[0] - 1):
                    if rates[i] * dt_actual < - result[i]:
                        dt_actual = min(dt_actual, - result[i] / rates[i] * 0.95)
                        working = True
                if working:
                    rates = self._rates(dt_actual, result, parameters)
            result += rates * dt_actual
            dt_done += dt_actual

        return result

    @classmethod
    def lam_noise(cls, dt: float, xi: float, **kwargs) -> float:
        raise NotImplementedError


class Mamis2023SEIRWN(_Mamis2023SEIR):
    """From 10.1098/rspa.2022.0568 with white noise"""

    name = _Mamis2023SEIR.name + 'wn'

    @classmethod
    def lam_noise(cls, dt: float, xi: float, **kwargs) -> float:
        return np.random.normal(loc=0.0, scale=np.sqrt(dt))


__known_models__.append(Mamis2023SEIRWN)


class Mamis2023SEIROU(_Mamis2023SEIR):
    """From 10.1098/rspa.2022.0568 with Ornstein–Uhlenbeck noise"""

    name = _Mamis2023SEIR.name + 'ou'

    @classmethod
    def lam_noise(cls, dt: float, xi: float, **kwargs) -> float:
        return xi + (np.random.normal() - xi) * dt / kwargs['tau']


__known_models__.append(Mamis2023SEIROU)


class NinoTorres2022(sde.SDEModel):
    """From 10.1016/j.idm.2021.12.008"""

    name = "ninotorres_2022"
    variable_names = ['S', 'E', 'IA', 'IS', 'Q', 'H', 'R', 'C', 'D', 'N']
    parameter_defaults = {
        'beta': 1.063147,
        'omega': 0.476190,
        'xi': 0.200000,
        'alpR': 0.105263,
        'alpD': 0.131579,
        'nu': 0.731120,
        'chi': 0.900000,
        'kappa': 1./4.6,
        'gamma': 1./8.,
        'p': 0.100,
        'q': 0.190,
        'a': 0.934,  # l in Table 1; confinement when t=tm
        'b': 0.011,  # m in Table 1; reenter when t=tl
        'eps': 5.000,
        'ta': 18.0,  # tm in Table 1; lockdown time
        'tb': 66.0,  # tl in Table 1; release time
        'sig1': 0.2,
        'sig2': 0.4
    }

    def step(self,
             current_time: float,
             current_vals: np.ndarray,
             dt: float,
             parameters: Dict[str, float]) -> np.ndarray:
        result = np.zeros_like(current_vals)

        p = parameters
        s, e, i_a, i_s, q, h, r, c, d, n = current_vals.tolist()
        sqrt_dt = np.sqrt(dt)

        k_c = int(current_time / dt)
        k_a = int(p['ta'] / dt)
        k_b = int(p['tb'] / dt)
        phi = p['a'] if k_c == k_a else 0.0
        psi = p['b'] if k_c == k_b else 0.0
        betac = p['beta'] * (1.0 if k_c <= k_a else (1.0 + p['eps']))

        d_s_c = phi * s * dt
        d_s_e = betac * s / n * (i_a + i_s) * dt
        d_e_i_a = (1. - p['p']) * p['kappa'] * e * dt
        d_e_i_s = p['p'] * p['kappa'] * e * dt
        d_i_a_i_s = (1. - p['nu']) * p['omega'] * i_a * dt
        d_i_a_r = p['nu'] * p['omega'] * i_a * dt
        d_i_s_q = p['xi'] * i_s * dt
        d_q_h = p['q'] * p['gamma'] * q * dt
        d_q_r = (1. - p['q']) * p['gamma'] * q * dt
        d_h_r = (1. - p['chi']) * p['alpR'] * h * dt
        d_h_d = p['chi'] * p['alpD'] * h * dt
        d_c_s = psi * c * dt

        d_s_i_a = p['sig1'] * s * i_a / n * sqrt_dt * np.random.normal()
        # i_a >= 0 ; assume s will stay positive
        d_s_i_a = max(d_s_i_a, - (i_a + d_e_i_a - d_i_a_i_s - d_i_a_r))

        d_s_i_s = p['sig2'] * s * i_s / n * sqrt_dt * np.random.normal()
        # i_s >= 0 ; assume s will stay positive
        d_s_i_s = max(d_s_i_s, - (i_s + d_e_i_s + d_i_a_i_s - d_i_s_q))

        result[0] = - d_s_c - d_s_e + d_c_s - d_s_i_a - d_s_i_s
        result[1] = d_s_e - d_e_i_a - d_e_i_s
        result[2] = d_e_i_a - d_i_a_i_s - d_i_a_r + d_s_i_a
        result[3] = d_e_i_s + d_i_a_i_s - d_i_s_q + d_s_i_s
        result[4] = d_i_s_q - d_q_h - d_q_r
        result[5] = d_q_h - d_h_r - d_h_d
        result[6] = d_i_a_r + d_q_r + d_h_r
        result[7] = d_s_c - d_c_s
        result[8] = d_h_d
        result[9] = sum(result[0:9])

        return result + current_vals


__known_models__.append(NinoTorres2022)


class Tesfaye2021(sde.SDEModel):
    """From 10.1186/s13662-021-03597-1"""

    name = 'tesfaye_2020'
    variable_names = ['S', 'V', 'I', 'T', 'R']
    parameter_defaults = {
        'phi': 0.008,
        'sig1': 0.15,
        'sig2': 0.0005,
        'alpha': 0.0143,
        'theta': 0.01,
        'mu': 0.016,
        'delta': 0.004,
        'tau': 0.017,
        'rho': 0.0012,
        'omega': 0.002,
        'beta1': 0.0,
        'beta2': 0.0,
        'beta3': 0.0,
        'beta4': 0.0,
        'beta5': 0.0
    }

    def step(self,
             current_time: float,
             current_vals: np.ndarray,
             dt: float,
             parameters: Dict[str, float]) -> np.ndarray:

        result = np.zeros_like(current_vals)
        p = parameters
        s, v, i, t, r = current_vals.tolist()
        sqrt_dt = np.sqrt(dt)

        result[0] = (p['phi'] + p['sig1'] * r + p['sig2'] * v - (p['alpha'] * i + p['theta'] + p['mu']) * s) * dt + p['beta1'] * s * np.random.normal() * sqrt_dt
        result[1] = (p['theta'] * s - (p['mu'] + p['sig2']) * v) * dt + p['beta2'] * v * np.random.normal() * sqrt_dt
        result[2] = (p['alpha'] * s * i - (p['mu'] + p['delta'] + p['tau']) * i) * dt + p['beta3'] * i * np.random.normal() * sqrt_dt
        result[3] = (p['delta'] * i - (p['mu'] + p['rho'] + p['omega']) * t) * dt + p['beta4'] * t * np.random.normal() * sqrt_dt
        result[4] = (p['rho'] * t - (p['mu'] + p['sig1']) * r) * dt + p['beta5'] * r * np.random.normal() * sqrt_dt

        return current_vals + result


__known_models__.append(Tesfaye2021)
