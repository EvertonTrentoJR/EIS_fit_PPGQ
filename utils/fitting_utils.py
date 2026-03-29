from utils import data_types, equivalent_circuit
import numpy as np
from scipy.optimize import curve_fit, minimize
import time
from functools import partial


#dictionary to handle function calls -> number of expected params and the function pointer
function_handlers = {
    "longo2020": {"n_params": 8, "function_ptr": equivalent_circuit.Longo2020},
}

class EquivalentCircuit:
    def __init__(self, topology: str, data_medium:data_types.SpectroscopyData, freqs:np.ndarray):
        '''
        :param topology: The circuit topoly that outputs the modeled impedance of the expected circuit
        :param data_medium: SpectrumData structure for the frequency sweep in the medium to be characterized
        :param freqs: array with the swept frequencies
        '''

        #validate topology
        topology = topology.lower() #convert to lower case
        valid_topologies = list(function_handlers.keys()) #expected parameters for the available models
        if topology not in valid_topologies:
            raise ValueError(f'[EquivalentCircuit] {topology} not implemented! Try: {valid_topologies}')
        self.topology = topology
        self.circuit_impedance = None
        self.fit_method = None

        #validate data_medium
        expected_types = [data_types.SpectroscopyData, list]
        if type(data_medium) not in expected_types:
            raise TypeError(f'[EquivalentCircuit] "data_medium" must be {expected_types}! Curr. type = {type(data_medium)}')
        self.data_medium = data_medium

        #validate freqs
        if not isinstance(freqs, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "freqs" must be a Numpy Array! Curr. type = {type(freqs)}')
        self.freqs = freqs

        #compute the measured impedance from the SpectroscopyData objects
        if type(data_medium)==data_types.SpectroscopyData:
            z_meas_real, z_meas_imag = characterization_utils.complex_impedance(data_medium, freqs)
            self.z_meas_real = z_meas_real
            self.z_meas_imag = z_meas_imag
            self.z_meas = z_meas_real-1j*z_meas_imag #complex impedance


    def CUMSE(self, theta, args):
        '''
        :param z_hat: the complex impedance computed from the fitted circuit
        :param z_meas: the complex impedance measured from the real system
        :return: the mean squared error between the measured and fitted impedance values
        '''

        z_hat = self.circuit_impedance(theta, [args[1], args[2]]) #compute the model for the arguments
        z_hat = z_hat.astype('complex')
        args[0] = args[0].astype('complex')
        if z_hat.ndim >= 2:
            z_hat = z_hat.T
            SSE = np.sum(((args[0].real-z_hat.real)**2)+((args[0].imag-z_hat.imag)**2), axis=1)
        else:
            SSE = np.sum(((args[0].real-z_hat.real)**2) + ((args[0].imag-z_hat.imag)**2))

        return SSE / len(z_hat)

    def fit_circuit(self, initial_guess:np.ndarray, scaling_array:np.ndarray, method='BFGS', tol=1e-6, verbose=False):
        '''
        :param initial_guess: the initial guess for the fit to run the iterative algorithms
        :param scaling_array: scale all the search parameters to avoid exploding gradients
        :param method: which optimization algorithm will be used to fit the circuit data
        :param tol: tolerance for the algorithms
        :param verbose: flag to print the statistics in the terminal
        :return: the parameters the best fit the expected equivalent circuit
        '''

        #validate "method"
        valid_methods = ['BFGS', 'NLLS', 'DLS', 'Nelder-Mead', 'PSO']
        if method not in valid_methods:
            raise ValueError(f'[EquivalentCircuit] {method} not implemented! Try: {valid_methods}')
        self.fit_method = method

        #validate "initial_guess"
        if len(initial_guess) != function_handlers[self.topology]["n_params"]:
            raise ValueError(f'[EquivalentCircuit] The number of initial guess parameters do not match with the given model! Should be {function_handlers[self.topology]["n_params"]}.')

        if not isinstance(initial_guess, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "initial_guess" must be a Numpy Array! Curr. type = {type(initial_guess)}')

        #validate "scaling_array"
        if len(scaling_array) != len(initial_guess):
            raise ValueError(f'[EquivalentCircuit] Length of the scaling array do not match the length of the initial guess! {len(scaling_array)} != {len(initial_guess)}')

        if not isinstance(scaling_array, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "scaling_array" must be a Numpy Array! Curr. type = {type(scaling_array)}')

        #run the parameter search
        omega = 2*np.pi*self.freqs #Hz to rad/s
        bounds = function_handlers[self.topology]["bounds"] #optimization boundaries
        if self.fit_method == "BFGS":
            self.circuit_impedance = function_handlers[self.topology]["function_ptr"]
            t_init = time.time()
            fit_obj = minimize(self.CUMSE, initial_guess, args=([self.z_meas, omega, scaling_array]), bounds=bounds, method='L-BFGS-B')
            t_elapsed = time.time() - t_init
            opt_fit = self.circuit_impedance(fit_obj.x, [omega, scaling_array]) #compute the circuit for the optimal values
            opt_params_scaled = fit_obj.x*scaling_array #rescale the minimized parameters
            nmse = self.NMSE(self.z_meas.astype("complex"), opt_fit.astype("complex")) #NMSE score for both complex parts
            nrmse = self.NRMSE(self.z_meas.astype("complex"), opt_fit.astype("complex")) #nrmse score for both complex parts
            chisqr = self.chi_square(self.z_meas.astype("complex"), opt_fit.astype("complex")) #chi-square score for both complex parts
            mae = self.MAE(self.z_meas.astype("complex"), opt_fit.astype("complex")) #mae score for both complex parts

            if verbose:
                print(f'[EquivalentCircuit] Gradient-based impedance fitting:')
                print(f't = {t_elapsed} s')
                print(f'NMSE = {nmse}')
                print(f'NRMSE = {nrmse}')
                print(f'chi-square = {chisqr}')
                print(f'MAE = {mae}')
                fit_params = function_handlers[self.topology]["fit_params"]
                print(f'fitted params = ')
                for i in range(len(fit_params)):
                    print(f'{fit_params[i]} = {opt_params_scaled[i]}')
                print()

            return optimization_utils.OptimizerResults(opt_params=fit_obj.x, opt_params_scaled=opt_params_scaled, opt_cost=fit_obj.fun,
                                    opt_fit=opt_fit, nmse_score=nmse, nrmse_score=nrmse, chi_square=chisqr, mae_score=mae,
                                    n_iter=fit_obj.nit, t_elapsed=t_elapsed) #return the optimized parameters

        elif self.fit_method == "NLLS":
            self.circuit_impedance = function_handlers[self.topology]["partial_function_ptr"]
            bounds = np.array(bounds) #convert the boundaries to numpy array
            bounds = ((bounds[:, 0]), (bounds[:, 1])) #curve_fit receives bounds as tuple

            #handle the real and imaginary parts separately
            circuit_impedance_real = partial(self.circuit_impedance, scaling=scaling_array, return_type="real") #set scaling and return_type to static inputs
            t_init = time.time()
            fit_params_real, fit_cov_real = curve_fit(circuit_impedance_real, omega, self.z_meas_real, p0=initial_guess, bounds=bounds, maxfev=30000) #hacky-fix to attempt never running out of allowed iterations
            t_elapsed = time.time() - t_init
            opt_fit = function_handlers[self.topology]["function_ptr"](fit_params_real, [omega, scaling_array]) #compute the circuit real output for the optimal values
            opt_params_scaled = fit_params_real*scaling_array #rescale the minimized parameters
            nmse = self.NMSE(self.z_meas.astype("complex"), opt_fit.astype("complex")) #NMSE score for both complex parts
            nrmse = self.NRMSE(self.z_meas.astype("complex"), opt_fit.astype("complex")) #nrmse score for both complex parts
            chisqr = self.chi_square(self.z_meas.astype("complex"), opt_fit.astype("complex")) #chi-square score for both complex parts
            mae = self.MAE(self.z_meas.astype("complex"), opt_fit.astype("complex"))  # mae score for both complex parts

            if verbose:
                print(f'[EquivalentCircuit] Non-linear least squares impedance fitting:')
                print(f'NLLS fit elapsed time = {t_elapsed} s')
                print(f'NMSE = {nmse}')
                print(f'NRMSE = {nrmse}')
                print(f'chi-square = {chisqr}')
                print(f'MAE = {mae}')
                fit_params = function_handlers[self.topology]["fit_params"]
                print(f'fitted params = ')
                for i in range(len(fit_params)):
                    print(f'{fit_params[i]} = {opt_params_scaled[i]}')
                print()

            return optimization_utils.OptimizerResults(opt_params=fit_params_real, opt_params_scaled=opt_params_scaled,
                                    opt_fit=opt_fit, nmse_score=nmse, nrmse_score=nrmse, chi_square=chisqr, mae_score=mae, t_elapsed=t_elapsed) #return the optimized parameters

        elif self.fit_method == 'DLS':
            self.circuit_impedance = function_handlers[self.topology]["function_ptr"]
            t_init = time.time()
            fit_obj = optimization_utils.LevenbergMarquardt(self.circuit_impedance, initial_guess, args=([self.z_meas, omega, scaling_array]), tol=tol, bounds=bounds)
            t_elapsed = time.time() - t_init
            opt_fit = self.circuit_impedance(fit_obj, [omega, scaling_array]) #compute the circuit for the optimal values
            opt_params_scaled = fit_obj*scaling_array #rescale the minimized parameters
            nmse = self.NMSE(self.z_meas.astype("complex"), opt_fit.astype("complex")) #NMSE score for both complex parts
            nrmse = self.NRMSE(self.z_meas.astype("complex"), opt_fit.astype("complex")) #nrmse score for both complex parts
            chisqr = self.chi_square(self.z_meas.astype("complex"), opt_fit.astype("complex")) #chi-square score for both complex parts
            mae = self.MAE(self.z_meas.astype("complex"), opt_fit.astype("complex"))  # mae score for both complex parts

            if verbose:
                print(f'[EquivalentCircuit] Damped Least-Squares impedance fitting:')
                print(f't = {t_elapsed} s')
                print(f'NMSE = {nmse}')
                print(f'NRMSE = {nrmse}')
                print(f'chi-square = {chisqr}')
                print(f'MAE = {mae}')
                fit_params = function_handlers[self.topology]["fit_params"]
                print(f'fitted params = ')
                for i in range(len(fit_params)):
                    print(f'{fit_params[i]} = {opt_params_scaled[i]}')
                print()

            return optimization_utils.OptimizerResults(opt_params=fit_obj, opt_params_scaled=opt_params_scaled, opt_fit=opt_fit,
                                    nmse_score=nmse, chi_square=chisqr, nrmse_score=nrmse, mae_score=mae, t_elapsed=t_elapsed) #return the optimized parameters

        elif self.fit_method == "Nelder-Mead":
            self.circuit_impedance = function_handlers[self.topology]["function_ptr"]
            t_init = time.time()
            fit_obj = optimization_utils.NelderMeadSimplex(self.CUMSE, initial_guess, args=([self.z_meas, omega, scaling_array]), tol=tol, bounds=bounds)
            t_elapsed = time.time() - t_init
            opt_fit = self.circuit_impedance(fit_obj, [omega, scaling_array]) #compute the circuit for the optimal values
            opt_params_scaled = fit_obj*scaling_array #rescale the minimized parameters
            nmse = self.NMSE(self.z_meas.astype("complex"), opt_fit.astype("complex")) #NMSE score for both complex parts
            nrmse = self.NRMSE(self.z_meas.astype("complex"), opt_fit.astype("complex")) #nrmse score for both complex parts
            chisqr = self.chi_square(self.z_meas.astype("complex"), opt_fit.astype("complex")) #chi-square score for both complex parts
            mae = self.MAE(self.z_meas.astype("complex"), opt_fit.astype("complex"))  # mae score for both complex parts

            if verbose:
                print(f'[EquivalentCircuit] Nelder-Mead Simplex impedance fitting:')
                print(f't = {t_elapsed} s')
                print(f'NMSE = {nmse}')
                print(f'NRMSE = {nrmse}')
                print(f'chi-square = {chisqr}')
                print(f'MAE = {mae}')
                fit_params = function_handlers[self.topology]["fit_params"]
                print(f'fitted params = ')
                for i in range(len(fit_params)):
                    print(f'{fit_params[i]} = {opt_params_scaled[i]}')
                print()

            return optimization_utils.OptimizerResults(opt_params=fit_obj, opt_params_scaled=opt_params_scaled, opt_fit=opt_fit,
                                    nmse_score=nmse, chi_square=chisqr, nrmse_score=nrmse, mae_score=mae, t_elapsed=t_elapsed) #return the optimized parameters

        elif self.fit_method == "PSO":
            self.circuit_impedance = function_handlers[self.topology]["function_ptr"]
            t_init = time.time()

            #compute 20 independent runs to yield the best fit through particle swarm
            #Note: this is a strategy to bypass PSO's randomness
            # Wang S-C, Liu Y-H. Research on Two-Stage Parameter Identification for Various Lithium-Ion Battery Models Using Bio-Inspired Optimization Algorithms.
            # Applied Sciences. 2026; 16(1):202. https://doi.org/10.3390/app16010202

            n_runs = 40
            best_obj = None #variable to store the 'fit_obj' with the best accuracy
            best_hat = None #varaible to store the fitted impendace with the best accuracy
            best_nmse = 10 #variable to store the best nmse
            best_chisqr = 10 #variable to store the best chi-square
            best_mae = 10 #variable to store the best mae
            for i in range(0,n_runs):
                fit_obj = optimization_utils.ParticleSwarm(self.CUMSE, function_handlers[self.topology]["n_params"], swarm_size=50, method='lbest', args=([self.z_meas, omega, scaling_array]), tol=tol, bounds=bounds)
                opt_fit = self.circuit_impedance(fit_obj, [omega, scaling_array]) #compute the circuit for the optimal values
                nmse = self.NMSE(self.z_meas.astype("complex"), opt_fit.astype("complex")) #NMSE score for both complex parts
                chisqr = self.chi_square(self.z_meas.astype("complex"), opt_fit.astype("complex")) #chi-square score for both complex parts
                mae = self.MAE(self.z_meas.astype("complex"), opt_fit.astype("complex")) #mae score for both complex parts

                #update the variables if a new best has been achieved
                if (nmse<best_nmse) & (chisqr<best_chisqr):
                    best_obj = fit_obj
                    best_hat = opt_fit
                    best_nmse = nmse
                    best_chisqr = chisqr
                    best_mae = mae

            t_elapsed = time.time()-t_init
            best_nrmse = self.NRMSE(self.z_meas.astype("complex"), best_hat.astype("complex")) #nrmse score for both complex parts
            opt_params_scaled = best_obj*scaling_array  # rescale the minimized parameters

            if verbose:
                print(f'[EquivalentCircuit] Particle Swarm Optimization {method} impedance fitting:')
                print(f't = {t_elapsed} s')
                print(f'NMSE = {best_nmse}')
                print(f'NRMSE = {best_nrmse}')
                print(f'chi-square = {best_chisqr}')
                print(f'MAE = {best_mae}')
                fit_params = function_handlers[self.topology]["fit_params"]
                print(f'fitted params = ')
                for i in range(len(fit_params)):
                    print(f'{fit_params[i]} = {opt_params_scaled[i]}')
                print()

            return optimization_utils.OptimizerResults(opt_params=best_obj, opt_params_scaled=opt_params_scaled, opt_fit=best_hat,
                                    nmse_score=best_nmse, chi_square=best_chisqr, nrmse_score=best_nrmse, mae_score=best_mae, t_elapsed=t_elapsed) #return the optimized parameters

        else:
            raise ValueError(f'[EquivalentCircuit] method = {method} not implemented! Try: {valid_methods}')

    def NRMSE(self, z: np.ndarray, z_hat: np.ndarray):
        '''
        :param z: the observed values (real measurements)
        :param z_hat: the predicted values from the fitted circuit
        :return: NRMSE of the fit for both real and imaginary part
        '''

        #validate 'z'
        if not isinstance(z, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z" must be a Numpy Array! Curr. type = {type(z)}')

        #validate 'z_hat'
        if not isinstance(z_hat, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z_hat" must be a Numpy Array! Curr. type = {type(z_hat)}')

        #validate shape
        if len(z) != len(z_hat):
            raise ValueError(f'[EquivalentCircuit] "z" and "z_hat" must match in length!')

        #mean absolute percentage error
        SSE = np.sum(((z_hat.real-z.real)**2)+((z_hat.imag-z.imag)**2)) #sum of squared errors
        RMSE = np.sqrt(SSE/len(z)) #root mean squared error
        CMEAN = np.mean(z.real+z.imag) #complex mean

        return np.abs(RMSE/CMEAN)

    def NMSE(self, z:np.ndarray, z_hat:np.ndarray):
        '''
        :param z: the observed values (real measurements)
        :param z_hat: the predicted values from the fitted circuit
        :return: NMSE of the fit
        '''

        #validate 'z'
        if not isinstance(z, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z" must be a Numpy Array! Curr. type = {type(z)}')

        #validate 'z_hat'
        if not isinstance(z_hat, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z_hat" must be a Numpy Array! Curr. type = {type(z_hat)}')

        #validate shape
        if len(z) != len(z_hat):
            raise ValueError(f'[EquivalentCircuit] "z" and "z_hat" must match in length!')

        #normalized mean squared error
        SSE = np.sum(((z_hat.real-z.real)**2) + ((z_hat.imag-z.imag)**2)) #sum of squared errors
        SSO = np.sum((z.real**2) + (z.imag**2)) #sum of squared measurements

        return SSE/SSO

    def chi_square(self, z:np.ndarray, z_hat:np.ndarray):
        '''
        :param z: the observed values (real measurements)
        :param z_hat: the predicted values from the fitted circuit
        :return: chi square of the fit
        '''

        #validate 'z'
        if not isinstance(z, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z" must be a Numpy Array! Curr. type = {type(z)}')

        #validate 'z_hat'
        if not isinstance(z_hat, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z_hat" must be a Numpy Array! Curr. type = {type(z_hat)}')

        #validate shape
        if len(z) != len(z_hat):
            raise ValueError(f'[EquivalentCircuit] "z" and "z_hat" must match in length!')

        W = np.abs(1/(z.T @ z))
        W = np.abs(W)*np.ones(len(z)) #weighting matrix
        res_real = z.real-z_hat.real #real residue
        chi_square_real = res_real.T @ (W*res_real)
        res_imag = z.imag-z_hat.imag #imaginary residue
        chi_square_imag = res_imag.T @ (W*res_imag)

        return chi_square_real + chi_square_imag

    def MAE(self, z:np.ndarray, z_hat:np.ndarray):
        '''
        :param z: the observed values (real measurements)
        :param z_hat: the predicted values from the fitted circuit
        :return: chi square of the fit
        '''

        #validate 'z'
        if not isinstance(z, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z" must be a Numpy Array! Curr. type = {type(z)}')

        #validate 'z_hat'
        if not isinstance(z_hat, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z_hat" must be a Numpy Array! Curr. type = {type(z_hat)}')

        #validate shape
        if len(z) != len(z_hat):
            raise ValueError(f'[EquivalentCircuit] "z" and "z_hat" must match in length!')

        z_abs = np.abs(z)
        z_hat_abs = np.abs(z_hat)
        abs_err = np.abs(z_abs-z_hat_abs)/z_abs #absolute error

        return 100*np.mean(abs_err)