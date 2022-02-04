import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction
from scipy import optimize
from emittance_calc import get_normemit, adapt_range
from data_handler import check_symmetry, add_measurements, find_inflection_pnt

show_plots_here = False

class LclsOpt():
    def __init__(self, init_scan=[-5, -4, -2.5, -1]): #[-5, -3, -1]
        self.energy = 0.135
        self.varscan = init_scan
        self.num_points_adapt = 5
        self.check_symmetry = check_symmetry
        self.add_measurements = add_measurements
        self.find_inflection_pnt = find_inflection_pnt
        self.uncertainty_lim = 0.15

    def get_adapted_quad_list(self, quad_list, bs_list, axis, num_points=5, show_plots=False):
        """
        Return adapted list of quad values (scalars, unnormalized), given a list of
        initial quad values (unnormalized), a list of beamsize scalars (unnormalized),
        an axis label (either 'x' or 'y'), and the number of points to use in the
        adapted scan.
        """
        #try:
        new_quad_arr = adapt_range(quad_list, bs_list, axis=axis, num_points=num_points, show_plots=show_plots)
        new_quad_list = new_quad_arr.tolist()
        # except:
        #     print("lcls_functions: adapt_range failed. Returning original quad_list.")
        #     print(".", end="")
        #     new_quad_list = quad_list
        return new_quad_list
    
    def get_beamsizes_from_machine(self, varx, vary, varz, varscan):
        """ FILL HERE """
        return x, y, xerr, yerr


    def evaluate(self, varx, vary, varz):
        varscan = self.varscan
        # get init scan
        x_rms, y_rms, x_err, y_err = self.get_beamsizes_from_machine(varx, vary, varz, varscan)

        #### ADAPT RANGES ######
        # TODO: what happens if no new adapted range is returned here
        new_quad_list_x = self.get_adapted_quad_list(self.varscan, x_rms, 'x', self.num_points_adapt, show_plots=show_plots_here)
        new_quad_list_y = self.get_adapted_quad_list(self.varscan, y_rms, 'y', self.num_points_adapt, show_plots=show_plots_here)

        if new_quad_list_x != self.varscan:
            # set quad 525 to values for new scan
            new_scan = self.get_beamsizes_from_machine(varx, vary, varz, new_quad_list_x)
            x_rms = new_scan[0]
            x_err = new_scan[2]
            
        if new_quad_list_y != self.varscan:
            new_scan = self.get_beamsizes_from_machine(varx, vary, varz, new_quad_list_y)
            y_rms = new_scan[0]
            y_err = new_scan[2]

        ##### CHECK SYMMETRY #####
        add_points_x = self.check_symmetry(new_quad_list_x, x_rms)
        add_points_y = self.check_symmetry(new_quad_list_y, y_rms)
        
        if add_points_x is not None:
            side = add_points_x[0]
            x_add = add_points_x[1]
            
            # get new data points
            new_scan = self.get_beamsizes_from_machine(varx, vary, varz, x_add)
            xrms_add = new_scan[0]
            xerr_add = new_scan[2]

            # then append to existing dataset
            if side == "left":
                new_quad_list_x = list(x_add) + list(new_quad_list_x)
                x_rms = list(xrms_add) + list(x_rms)
                x_err = list(xerr_add) + list(x_err)
            else:
                new_quad_list_x = list(new_quad_list_x) + list(x_add)
                x_rms = list(x_rms) + list(xrms_add)
                x_err = list(x_err) + list(xerr_add)
                
        if add_points_y is not None:
            side = add_points_y[0]
            y_add = add_points_y[1]
            
            # get new data points
            new_scan = self.get_beamsizes_from_machine(varx, vary, varz, y_add)
            yrms_add = new_scan[0]
            yerr_add = new_scan[2]

            # then append to existing dataset
            if side == "left":
                new_quad_list_y = list(y_add) + list(new_quad_list_y)
                y_rms = list(yrms_add) + list(y_rms)
                y_err = list(yerr_add) + list(y_err)
            else:
                new_quad_list_y = list(new_quad_list_y) + list(y_add)
                y_rms = list(y_rms) + list(yrms_add)
                y_err = list(y_err) + list(yerr_add)
        
        ##### REMOVE BAD POINTS #####    
        left_x, right_x =  self.find_inflection_pnt(new_quad_list_x, x_rms, show_plots=show_plots_here)
        left_y, right_y =  self.find_inflection_pnt(new_quad_list_y, y_rms, show_plots=show_plots_here)
        
        # print(left_x, right_x)
        # print(left_y, right_y)
        new_quad_list_x = new_quad_list_x[left_x:right_x]
        x_rms = x_rms[left_x:right_x]
        
        new_quad_list_y = new_quad_list_y[left_y:right_y]
        y_rms = y_rms[left_y:right_y]
        
        ##### GET EMITTANCE ####
        norm_emit = get_normemit(self.energy, new_quad_list_x, new_quad_list_y, x_rms, y_rms, xrms_err=x_err, yrms_err=y_err,
                                 adapt_ranges=False, show_plots=show_plots_here)

        if np.isnan(norm_emit[0]) or np.isnan(norm_emit[1]):
            #print("NaN emit")
            return -np.random.uniform(1000,2000), -np.random.uniform(1000,2000)

        norm_emit_sqrt = np.sqrt(norm_emit[0] * norm_emit[1])
        norm_emit_err = norm_emit_sqrt * ((norm_emit[2] / norm_emit[0]) ** 2 + (norm_emit[3] / norm_emit[1]) ** 2) ** 0.5
        return -norm_emit_sqrt/1e-6, norm_emit_err/1e-6   # in um

    def eval_simplex(self, x):
        return -1 * self.evaluate(x[0], x[1], x[2])[0]

    def run_simplex_opt(self, max_iter, initial_guess):
#         initial_guess = np.array(
#             [np.random.uniform(self.pbounds[0][0], self.pbounds[0][1]),
#              np.random.uniform(self.pbounds[1][0], self.pbounds[1][1]),
#              np.random.uniform(self.pbounds[2][0], self.pbounds[2][1])
#              ])

#         initial_guess1 = self.pbounds[0][0]+ np.random.rand(1) * (self.pbounds[0][1] - self.pbounds[0][0])
#         initial_guess2 = self.pbounds[1][0]+ np.random.rand(1) * (self.pbounds[1][1] - self.pbounds[1][0])
#         initial_guess3 = self.pbounds[2][0]+ np.random.rand(1) * (self.pbounds[2][1] - self.pbounds[2][0])

        #initial_guess = np.array([initial_guess1, initial_guess2, initial_guess3])

        min = optimize.minimize(self.eval_simplex, initial_guess,
                                method='Nelder-Mead', options={'maxiter': max_iter,
                                                               'return_all':True,
                                                               'adaptive': True
                                                               },
                                )
        return min
    
    def run_bo_opt(self, rnd_state=11, init_pnts=3, n_iter=120):
        # Set domain
        bounds = {'varx': self.pbounds[0], 'vary': self.pbounds[1], 'varz': self.pbounds[2]}

        # Run BO
        optimizer = BayesianOptimization(
            f=self.evaluate,
            pbounds=bounds,
            random_state=rnd_state,
        )

        optimizer.maximize(init_points=init_pnts,
                           n_iter=n_iter,
                           kappa= 3,
                           # kappa_decay = 0.8,
                           # kappa_decay_delay = 25
                           )

        return optimizer
    
    def run_bo_opt_w_reject(self, rnd_state=11, init_pnts=3, n_iter=120):
        # Set domain
        bounds = {'varx': self.pbounds[0], 'vary': self.pbounds[1], 'varz': self.pbounds[2]}

        # Run BO
        optimizer = BayesianOptimization(
            f=None,
            pbounds=bounds,
            random_state=rnd_state,
            verbose=2
        )
        
        utility = UtilityFunction(kind="ucb", kappa= 3, xi=0.0)
        target_list = []
        
        # get init points
        for i in range(init_pnts):
            target, error = np.nan, np.nan
            while np.isnan(target) or np.isnan(error) or error/target > self.uncertainty_lim:
                next_point =  {'varx': np.random.uniform(self.pbounds[0][0], self.pbounds[0][1]),
                               'vary': np.random.uniform(self.pbounds[1][0], self.pbounds[1][1]), 
                               'varz': np.random.uniform(self.pbounds[2][0], self.pbounds[2][1])
                                       }
                # evaluate next point
                target, error = self.evaluate(**next_point)
          
            optimizer.register(params=next_point, target=target)     
            if target_list and target>np.max(target_list):
                color = '\033[95m', '\033[0m'
            else:
                color = '\u001b[30m', '\033[0m'
            
            print(f"{color[0]}iter {i} | target {-1*target:.3f} | config {next_point['varx']:.6f} {next_point['vary']:.6f} {next_point['varz']:.6f}{color[1]}")
            target_list.append(target)
            
        # BO iters
        for i in range(n_iter):
            target, error = np.nan, np.nan
            while np.isnan(target) or np.isnan(error) or error/target > self.uncertainty_lim: 
                next_point = optimizer.suggest(utility)
                target, error = self.evaluate(**next_point)

            optimizer.register(params=next_point, target=target)
            if target_list and target>np.max(target_list):
                color = '\033[95m', '\033[0m'
            else:
                color = '\u001b[30m', '\033[0m'
            
            print(f"{color[0]}iter {i} | target {-1*target:.3f} | config {next_point['varx']:.6f} {next_point['vary']:.6f} {next_point['varz']:.6f}{color[1]}")
            target_list.append(target)

        return optimizer