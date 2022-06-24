import GPy
import numpy as np
import os
from scipy.spatial.distance import euclidean

from .acq import explore
from .hmc import HMCwrapper
from .kernel_factory import KernelFactory
from .model import Model
from ..utils import InitManager
from ..utils import Minimization
from ..utils import Timer


class BoMain:
    """
    Class for handling Bayesian Optimization
    """

    def __init__(self, STS, mainOutput, rstManager):
        """
        Initializes the BoMain class/object
        """
        self.dim = STS.dim
        self.initpts = STS.initpts
        self.iterpts = STS.iterpts
        self.updatefreq = STS.updatefreq
        self.initupdate = STS.initupdate
        self.updateoffset = STS.updateoffset
        self.updaterestarts = STS.updaterestarts
        self.hmciters = STS.hmciters
        self.cores = STS.cores
        self.f = STS.f
        self.periods = STS.periods
        self.bounds = STS.bounds
        self.acqfn = STS.acqfn
        self.acqfnpars = STS.acqfnpars
        self.acqtol = STS.acqtol
        self.min_dist_acqs = STS.min_dist_acqs
        self.minzacc = STS.minzacc
        self.kerntype = STS.kerntype
        self.noise = STS.noise
        self.timer = STS.timer
        self.dir = STS.dir
        self.dxhat_tol = STS.dxhat_tol

        self.kernel = KernelFactory.construct_kernel(STS)
        self.model = None
        self.initX = np.array([]).reshape(0, self.dim)
        self.initY = np.array([]).reshape(0, 1)
        self.convergence = np.array([]).reshape(0, 5)
        self.normmean = 0
        self.normsd = 1
        self.est_yrange = 0
        self.hmcsample = []
        self.mainOutput = mainOutput
        self.rstManager = rstManager
        self.initManager = InitManager(STS.inittype, STS.dim, STS.bounds,
                                       STS.initpts)

    def get_x(self):
        """
        Returns the points where the objective function has been evaluated,
        in order of acquisition.
        """
        if self.model == None:
            return self.initX
        else:
            return self.model.X

    def get_y(self):
        """
        Returns the evaluated values of the objective function, in order of
        acquistiion. This method should always be called instead of
        BoMain.model.Y for any use outside of the BoMain class.
        """
        if self.model == None:
            return self.initY
        else:
            return self.model.Y * self.normsd + self.normmean

    def get_mu(self, x):
        """
        Returns the model prediction at point x. This method should always be
        called instead of BoMain.model.mu for any use outside of the BoMain
        class.
        """
        if self.model == None:
            return None
        else:
            return self.model.mu(x) * self.normsd + self.normmean

    def get_nu(self, x):
        """
        Returns the variance of the model prediction at point x, with added
        noise (model variance).
        """
        if self.model == None:
            return None
        else:
            return self.model.nu(x) * self.normsd

    def get_grad(self, x):
        """
        Returns the predictive gradients. If the implemented mean shift is
        a constant, should just wrap self.model.predictive_gradients.
        """
        g = self.model.predictive_gradients(x)
        return(np.array([g[0] * self.normsd, None]))

    def _add_xy(self, xnew, ynew):
        """
        Internal functionality for saving a new acquisition (x, y), accounting
        for model mean shift. Initializes the GP model when the number of
        acquisitions meets the pre-specified number of initialization points.
        """
        if self.model == None:
            self.initX = np.vstack([self.initX, xnew])
            self.initY = np.vstack([self.initY, ynew])
            if self.get_x().shape[0] == self.initpts:
                self._init_model()
        else:
            X = np.vstack([self.get_x(), np.atleast_2d(xnew)])
            Y = np.vstack([self.get_y(), ynew])
            self.normmean = np.mean(Y)
#            self.normsd = np.std(Y)    # NORMALIZATION
            self.est_yrange = np.max(Y)-np.min(Y)
            self.model.redefine_data(X, (Y - self.normmean) / self.normsd)

    def add_xy_list(self, xnew, ynew):
        """
        Saves multiple acquisitions, accounting for model mean shift.
        Initializes the GP model when the number of acquisitions meets the
        pre-specified number of initialization points.
        """
        for i in range(xnew.shape[0]):
            self._add_xy(xnew[i,:], ynew[i])

    def _init_model(self):
        """
        Initializes the GP model. This should be called when all initialization
        points have been evaluated. Further acquisition locations can then be
        queried by optimizing an acquisition function.
        """
        self.normmean = np.mean(self.initY)
#        self.normsd = np.std(self.initY)   # NORMALIZATION
        self.est_yrange = np.max(self.initY)-np.min(self.initY)
        self.model = Model(
            self.initX,
            (self.initY - self.normmean) / self.normsd,
            self.bounds,
            self.min_dist_acqs,
            self.minzacc,
            self.kerntype,
            self.kernel,
            self.dim,
            self.noise
        )

    def _should_optimize(self, i):
        """
        Returns True if the model should be optimized at iteration i.
        """
        bo_i = np.max([0, i - self.initpts + 1]) # == 0 means initial iters

        # No model -> no optimization
        if self.model == None: return False

        # Check if initialization has just completed and we want to optimize
        elif bo_i == 0:
            if self.initupdate: return True
            else: return False

        # Check if optimization threshold and interval have passed
        elif bo_i >= self.updateoffset and bo_i % self.updatefreq == 0:
            return True

        # Otherwise there is no need to optimize
        else: return False

    def run_optimization(self):
        """
        The Bayesian optimization main loop. Evaluates first the initialization
        points, then creates a GP model and uses it and an acquisition function
        to locate the next points where to evaluate. Stops when a pre-specified
        number of initialization points and BO points have been acquired or a
        convergence criterion is met.
        """
        xnext = self._get_xnext(0)

        for i_iter in range(self.initpts + self.iterpts):

            ### BO ITERATION
            # iteration start
            self.mainOutput.iteration_start(i_iter+1, self.initpts)
            self.timer.startLap()

            # evaluation
            xnew, ynew = self._evaluate(i_iter, xnext)

            # store new data and refit model
            # create the model when all initial data has been acquired
            for i in range(len(ynew)):
                self._add_xy(xnew[i], ynew[i])
                self.rstManager.new_data(xnew[i], ynew[i])

            # optimize model if needed
            if self._should_optimize(i_iter):
                self._optimize_model(i_iter)

            # finding next acquisition location
            xnext = self._get_xnext(i_iter + 1)

            ### CONVERGENCE DIAGNOSTICS
            # calculate iteration specific info
            self._add_convergence()

            ### OUTPUT
            # add model parameters to rst-file
            if self.model != None:
                mod_unfixed_par = self.model.get_unfixed_params()
                self.rstManager.new_model_params(mod_unfixed_par)
            else:
                mod_unfixed_par = None

            # iteration output routine
            self.mainOutput.iteration_summary(
                self.get_y().size, xnew, ynew, self.convergence,
                xnext, self.est_yrange, mod_unfixed_par, self.timer
            )

            ### ADDITIONAL STOPPING CRITERION: CONVERGENCE
            if self._has_converged(i_iter):
                self.mainOutput.convergence_stop()
                break

    def _has_converged(self, i_iter):
        """
        Checks whether dxhat has been within tolerance for long enough
        TODO: should use dxmuhat instead?
        """
        if self.dxhat_tol is not None:
            conv_tol, conv_it = self.dxhat_tol
            if i_iter > self.initpts + conv_it:
                curr_xhat = self.convergence[-1,-3]
                within_tol = True
                for test_i in range(2,conv_it-1):
                    if euclidean(self.convergence[-test_i,-3], \
                                 curr_xhat) > conv_tol:
                        within_tol = False
                return within_tol
        return False

    def _get_xnext(self, i_iter):
        """
        Get a new point to evaluate by either reading it from the rst-file or,
        in case it doesn't contain the next point to evaluate, by obtaining
        a new initial point (when run is in initialization stage) or
        minimizing the acquisition function (when the run is in BO stage). 
        """
        xnext = self.rstManager.get_x(i_iter)
        if np.any(xnext == None):
            if i_iter < self.initpts: xnext = self.initManager.get_x(i_iter)
            else: xnext, acqfn = self._acqnext(i_iter)
        return xnext

    def _acqnext(self, i_iter):
        """
        Selects the acquisition function to use and returns its xnext location
        as well as the used acquisition function.
        """
        if self.hmciters == 0:
            acqfn = self.acqfn
            expfn = explore
        else:
            hmc = HMCwrapper(self.hmcsample)
            acqfn = hmc.averageacq(self.acqfn, self.cores)
            expfn = hmc.averageacq(explore, self.cores)

        xnext = self._minimize_acqfn(acqfn)

        # check if xnext indicates we should trigger pure exploration
        if self._location_overconfident(xnext):
            xnext = self._minimize_acqfn(expfn)
            return xnext, expfn
        else:
            return xnext, acqfn

    def _minimize_acqfn(self, acqfn):
        """
        Minimizes the acquisition function to find the next
        sampling location 'xnext'.
        """

        nof_stdp = len(np.where(np.array(self.kerntype) == 'stdp')[0])

        # Calculate the number of local minimizers to start.
        # For the ith dimension, the number of local minima along a slice
        # is approximately n(i) = boundlength(i)/(2*lengthscale(i)). To get
        # the total number of minima for all of the search space, multiply
        # together n(i) over all i. The length scale of stdp kernel must be
        # multiplied by ~100 to be comparable with rbf kernel length scale.
        lengthscale_numpts = 1
        for i in range(self.dim):
            lengthscale = 2 * self.model.get_unfixed_params()[i+1]
            if self.kerntype[i] == 'stdp':
                lengthscale *= 100 # TODO: fix prefactor
            lengthscale_numpts *= max(1, self.periods[i] / lengthscale)

        num_pts = min(len(self.model.X), int(lengthscale_numpts))

        # minimize acqfn to obtain sampling location
        gmin = Minimization.minimize_from_random(
                    acqfn, self.bounds, num_pts=num_pts,
                    shift=0.1*np.array(self.periods),
                    args=[self.model, self.acqfnpars]
               )
        return np.atleast_1d(np.array(gmin[0]))

    def _location_overconfident(self, xnext):
        """
        Checks is model variance is lower than tolerance at suggested xnext.
        """
        if self.acqtol is None:
            return False
        else:
            if self.model.nu(xnext) >= self.acqtol:
                return False
            else:
                self.mainOutput.progress_msg("Acquisition location " +
                "too confident, doing pure exploration", 1)
                return True

    def _evaluate(self, i, xnext):
        """
        Get a new evaluation either from the rst-file or, in case it doesn't
        contain the corresponding evaluation, by evaluating the user function
        """
        ynew = self.rstManager.get_y(i)
        if np.any(ynew == None): return self._evaluate_xnew(xnext)
        else: return np.atleast_2d(xnext), np.atleast_1d(ynew)

    def _evaluate_xnew(self, xnew):
        """
        Evaluates user function at given location 'xnew'
        to get the observation scalar 'ynew'.
        Later also gradient 'ydnew' should be made possible.
        """
        # run user script to evaluate sample
        self.mainOutput.progress_msg(
            "Evaluating objective function at x =" +
                self.mainOutput.utils.oneDarray_line(
                    xnew.flatten(), self.dim, float
                ),
            1
        )

        local_timer = Timer()
        xnew = np.atleast_2d(xnew)
        ynew = np.atleast_1d(self.f(xnew)) # call the user function
        os.chdir(self.dir) # in case the user function changed the dir
        self.mainOutput.progress_msg(
            "Objective function evaluated," + 
            " time [s] %s" % (local_timer.str_lapTime()), 1)

        # return new data
        for i in range(len(xnew)-1):
            ynew = np.insert(ynew, -1, ynew[0])
        return xnew, ynew

    def _optimize_model(self, i):
        """
        Optimize the GP model or, if the next hyperparameters are contained in
        the rst-file, just use them.
        """
        if self.hmciters > 0:
            hmc = GPy.inference.mcmc.HMC(self.model)
            burnin = hmc.sample(int(self.hmciters*0.33))
            self.hmcsample = hmc.sample(self.hmciters)

        n = self.model.get_unfixed_params().size
        theta = self.rstManager.get_theta(i, n)
        if np.any(theta == None):
            self.model.optimize_controlrstr(self.updaterestarts)
        else:
            self.model.set_unfixed_params(theta)
            self.model.redefine_data(self.model.X, self.model.Y)
            # the latter is required for the model to properly update

    def _add_convergence(self):
        """
        Updates self.convergence with a new row containing
        bestx, besty, xhat, muhat, nuhat (in this order).
        """
        if self.model == None:
            conv = np.atleast_2d(np.repeat(np.nan, 5))
        else:
            bestx, besty = self.model.get_bestxy()
            xhat, muhat, nuhat = self.model.min_prediction()
            besty = besty * self.normsd + self.normmean
            muhat = muhat * self.normsd + self.normmean
            conv = np.atleast_2d([bestx, besty, xhat, muhat, nuhat])

        self.convergence = np.append(self.convergence, conv, axis=0)

    # add by guoxm
    
    def run_optimization_single(self):
        """
        The Bayesian optimization main loop. Evaluates first the initialization
        points, then creates a GP model and uses it and an acquisition function
        to locate the next points where to evaluate. Stops when a pre-specified
        number of initialization points and BO points have been acquired or a
        convergence criterion is met.
        """
        xnext = self._get_xnext(0)        

        for i_iter in range(self.initpts + self.iterpts):

            ### BO ITERATION
            # iteration start
            self.mainOutput.iteration_start(i_iter+1, self.initpts)
            self.timer.startLap()

            # evaluation
            xnew, ynew = self._evaluate(i_iter, xnext)

            # store new data and refit model
            # create the model when all initial data has been acquired
            for i in range(len(ynew)):
                self._add_xy(xnew[i], ynew[i])
                self.rstManager.new_data(xnew[i], ynew[i])

            # optimize model if needed
            if self._should_optimize(i_iter):
                self._optimize_model(i_iter)
            else:
                # finding next acquisition location
                xnext = self._get_xnext(i_iter + 1)

                ### CONVERGENCE DIAGNOSTICS
                # calculate iteration specific info
                self._add_convergence()

            ### OUTPUT
            # add model parameters to rst-file
            if self.model != None:
                mod_unfixed_par = self.model.get_unfixed_params()
                self.rstManager.new_model_params(mod_unfixed_par)
            else:
                mod_unfixed_par = None

            # iteration output routine
            self.mainOutput.iteration_summary(
                self.get_y().size, xnew, ynew, self.convergence,
                xnext, self.est_yrange, mod_unfixed_par, self.timer
            )

            ### ADDITIONAL STOPPING CRITERION: CONVERGENCE
            if self._has_converged(i_iter):
                self.mainOutput.convergence_stop()
                break
    # -- 20210413 guoxm