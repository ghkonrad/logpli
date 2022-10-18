#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#############################################################################
#   Copyright 2018 Konrad Sakowski, Stanislaw Krukowski, Pawel Strak
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#############################################################################
#

import argparse
import collections
import csv
from functools import cached_property
from IPython.display import clear_output, display, HTML, Markdown, Latex
import matplotlib.pyplot as plt

import numpy
import numpy.linalg

import pint

import scipy.sparse
import scipy.sparse.linalg
import scipy.interpolate
import scipy.optimize 
import scipy.ndimage

import sys


un_default = pint.UnitRegistry(system='mks');

class EstimateABC(object):
	"""
		Estimate ABC constants.
		
		Prameters:
		* *un* --- a :class:`pint.UnitRegistry` instance
	"""
	def __init__(self, un = un_default):
		self.un = un;
		

	def limit_args(self, x, y, x_min, x_max):
		"""
			Limit argument vector *x* and value vector *y* so that only arguments in [*x_min*, *x_max*] remain.

			Return: (x,y) with arguments in [*x_min*, *x_max*].
		"""
		indices = numpy.where(numpy.logical_and(x >= x_min, x <= x_max));

		return (x[indices], y[indices]);

	@cached_property
	def plot(self):
		"""
			Return a compatible :class:`Plotter` instance to perform plots.
			Multiple uses results in the same instance being returned.
		"""
		return Plotter(un=self.un);

	@cached_property
	def print(self):
		"""
			Return a compatible :class:`PrintInfo` instance to display information.
			Multiple uses results in the same instance being returned.
		"""
		return PrintInfo(un=self.un);

	def abc(self, alpha, beta, gamma, mu, n0):
		"""
			Compute ABC recombination parameters based on approximation of a negative logarithmic derivative of luminescence intensity by 2nd-order polynomial :math:`r_L(y) \\approx \\alpha + \\beta y + \\gamma y^2`. 

			Parameters:

			* *alpha*, *beta*, *gamma* --- polynomial coefficients
			* *mu* --- argument in definition :math:`y = J^{1/\mu}`
			* *n0* --- initial minority carrier concentration

			Return: a named tuple:

			* *A*, *B*, *C* --- ABC recombination parameters
		"""
		Return = collections.namedtuple("Return", ["A", "B", "C"])
		un = self.un;

		return Return(
			A = (alpha/(mu)).to(1/un.s),
			B = (beta/(mu*n0)).to(un.cm**3/un.s),
			C = (gamma/(mu * n0**2)).to(un.cm**6/un.s),
			);

	def fit_J(self, te, Je, mu, t0, t1, init_alpha = None, init_beta = None, init_gamma = None, alpha = None, beta = None, gamma = None, minimize_method = "CG", log = False, **minimize_options_kwargs):
		r"""
			This function improves optical output *Jfit* based on assumptions that the (negative)logarithmic derivative of *J* is given by polynomial
			:math:`r_L(y) = \alpha + \beta y + \gamma y^2`
			
			The procedure chooses *alpha*, *beta* and *gamma* by minimalizing least-squares error of *Jfit* versus *Je* on a given interval.
			
			Parameters:
			
			* *t0*, *t1* --- lower and upper boundary of interval; fitting interval beginning time must be within this interval
			* *init_alpha*,*init_beta*,*init_gamma* --- initial values of *alpha*, *beta* and *gamma*; if not set, already computed fit will be used; it shall be just float numbers (no units), as these parameters are dimensionless anyway
			* *alpha*, *beta*, *gamma* --- parameter value may be fixed (*None* --- do not fix; this is default),
			* *minimize_method* --- minimalization method accepted by :func:`scipy.optimize.minimize`
			* *minimize_options_kwards* --- anything else will be passed as named arguments to :func:`scipy.optimize.fmin_cg`
			* *log* --- operate on logarithms of parameters instead of parameters themselves (default: false); the parameters shall not be converted to logarithms manually, this procedure takes care of that transparently, they just have to be positive
		"""
		
		Return = collections.namedtuple("Return", ["alpha", "beta", "gamma", "mu"])

		un = self.un;
		time_unit = te.units;

		params = ["alpha", "beta", "gamma"];

		fixvals = {
			"alpha": alpha,
			"beta" : beta,
			"gamma": gamma,
		};

		if(log):
			for key, val in [(param, fixvals[param]) for param in params]+[("init_alpha", init_alpha), ("init_beta", init_beta), ("init_gamma", init_gamma)]:
				if val is not None:
					assert val > 0, f"Invalid value of {key} ({val}). For logarithmic search scale, parameters must be positive";

		rngf = (t0 <= te) * (te <= t1) * (Je.magnitude != 0); # wywalam też zera w prądzie, bo przez nie tylko trudniej liczyć (czasem się zdarzają przez błędy pomiaru)

		tstart = te[0];
		Jstart = Je[0];


		tf = te[rngf]; # for these times we do the fitting
		Jf = Je[rngf];

		def params2x(**kwargs):
			x = [];
			for param in params:
				if(fixvals[param] is None):
					if(not log):
						x.append(kwargs[param].to(1/time_unit).magnitude);
					else:
						x.append(numpy.log(kwargs[param].to(1/time_unit).magnitude));

			return x;

		def x2params(x):
			ret = {};
			d = collections.deque(x);
			for param in params:
				if(fixvals[param] is None):
					if(not log):
						ret[param] = d.popleft() / time_unit;
					else:
						ret[param] = numpy.exp(d.popleft()) / time_unit;
				else:
					ret[param] = fixvals[param].to(time_unit);

			assert len(d) == 0, "Too many elements on the input list";
			return ret;
		
		#print(tf)

		def func(x):
			Jfit = self.J_from_poly(
				tf=tf, 
				tstart=tstart,
				Jstart=Jstart,
				mu=mu,
				**x2params(x),
				);
			return numpy.linalg.norm(((Jf - Jfit) / Jf).to(un.dimensionless).magnitude);

		if(init_alpha is None):
			init_alpha = alpha;

		if(init_beta is None):
			init_beta = beta;

		if(init_gamma is None):
			init_gamma = gamma;

		x0 = params2x(
			alpha = init_alpha,
			beta = init_beta,
			gamma = init_gamma,
			);

		if(log):
			for key, val in x2params(x0).items():
				if val is not None:
					assert val.magnitude > 0, f"Invalid initial value {key} ({val}). For logarithmic search scale, parameters must be positive (check implicit values --- from initial fits)";

		#f_Jfit0 = scipy.interpolate.interp1d(
		#	*self._reverse_fit_helper(t0.magnitude, t1.magnitude, tstart, *x0),
		#	kind="linear"
		#		 );

		#print(func(x0))

		#print(self.te[idx0].magnitude, self.Je[idx0].magnitude)
		#print(self.fit_approx_idcs(), idx0)

		print("Initial value:", func(x0));
		#result = scipy.optimize.fmin_cg(func, x0, epsilon = 1e-11)
		res = scipy.optimize.minimize(func, x0, method = minimize_method, options=minimize_options_kwargs)

		result = x2params(res.x);

		print(result);

		return Return(
			alpha = result["alpha"],
			beta  = result["beta"],
			gamma = result["gamma"],
			mu = mu,
			);


	def fit_poly(self, ts, Js, mdlnJs, mu, fit_interval=None):
		"""
			Polynomial approximation of negative logarithmic derivative of luminescence intensity.

			Parameters:

			* *ts*  --- a vector of time steps for smoothed luminescence intensity
			* *Js* --- a vector of smoothed luminescence intensity
			* *mdlnJs*  --- a vector of negative logarithmic derivative of luminescence intensity
			* *mu* --- degree of the fitting polynomial

			Return:

			* *ys* --- total range of :math:`J_L^{1/\\mu}` based on smoothed values
			* *yf* --- range of :math:`J_L^{1/\\mu}` used in fitting procedure
			* *fit_interval* --- used fitting interval; either as given or inferred by this procedure, with units
			* *p_mdlnJf* --- polynomial coefficients of approximation of negative logarithmic derivative of luminescence intensity :math:`r_L(y) \\approx \\alpha + \\beta y + \\gamma y^2`; lowest-degree coeficients are at the end of this vector
			* *mu* --- *mu* used in fitting procedure
		"""

		Return = collections.namedtuple("Return", ["ys", "yf", "fit_interval", "p_mdlnJf", "mu"])

		un=self.un;

		ys = Js**(1/mu);

		f_mdlnJs = scipy.interpolate.interp1d(ys.magnitude, mdlnJs.magnitude, kind="linear");

		if(fit_interval is None):
			fit_interval = [min(ys).magnitude, max(ys).magnitude];

		if(hasattr(fit_interval, "units")):
			fit_interval = fit_interval.to(ys.units).magnitude;
		
		yf = numpy.linspace(*fit_interval, 100);
		
		polycoefs=numpy.polyfit(yf, f_mdlnJs(yf), deg=2);

		p_mdlnJf=[p*mdlnJs.units / (ys.units**(len(polycoefs)-i-1)) for i,p in enumerate(polycoefs)]

		ret = Return(
			ys = ys,
			yf = yf*ys.units,
			fit_interval = numpy.array(fit_interval)*ys.units,
			p_mdlnJf=p_mdlnJf,
			mu = mu,
			);

		return ret;

	def J_from_poly(self, tf, tstart, Jstart, alpha, beta, gamma, mu):
		r"""
			This function computes optical output *Jf* based on assumptions that the (negative)logarithmic derivative of *J* is given by polynomial
			:math:`r_L(y) = \alpha + \beta y + \gamma y^2`
			where :math:`y = J^{1/\mu}`.
			
			This helper function accepts all involved parameters (it does not use *self* parameters).
			
			The parameters must be numbers (not *pint* units), scaled as in description below.
			
			Parameters:
			
			* *tf* --- a monotone vector of time argument to compute optical output for
			* *tstart* --- initial condition time
			* *Jstart* --- initial condition normalized output
			* *alpha*, *beta*, *gamma* --- polynomial coefficients
			* *mu* --- power used in polynomial argument definition: :math:`y = J^{1/\mu}`

			Return: *Jf* --- estimated optical output corresponding to given :math:`r_L`
		"""
		

		un = self.un;
		tfit = tf.magnitude;
		tstart = tstart.to(tf.units).magnitude;

		alpha = alpha.to(1/tf.units).magnitude;
		beta  = beta.to(1/tf.units).magnitude;
		gamma = gamma.to(1/tf.units).magnitude;
		
		Jstart = Jstart.to(un.dimensionless).magnitude;



		tfit0 = tfit[0]; # initial condition would be here for the tf range

		rhs = lambda J, t: -(alpha * J + beta * (J**(1 + 1/mu)) + gamma * (J**(1 + 2/mu)));

		# First, we go from whatever the starting point is to the beginning of the tf
		if(tstart == tfit0):
			# alright, so we are at the start already
			Jf0 = Jstart;
		else:
			Jfitpre = scipy.integrate.odeint(
				rhs,
				Jstart,
				[tstart, tfit0],
			);

			Jf0 = Jfitpre[-1];

		# Initial condition prepared, we compute Jf values for tf=tfit

		Jf = scipy.integrate.odeint(
				rhs,
				Jf0,
				tfit,
			);
		Jf = Jf.T[0]; # odeint zwraca wektor kolumnowy, przynajmniej tutaj
		return Jf * un.dimensionless;


	def poly_coefs(self, p_mdlnJs):
		"""
			This function returns coefficients of a polynomial up to 2nd order as
			:math:`\\gamma x^2 + \\beta x + \\alpha`.

			Parameters:

			* *p_mdlnJs* --- polynomial coefficients, 1, 2 or 3 numbers: [*gamma*, *beta*, *alpha*]; *alpha* is always last
		"""
		assert 0 < len(p_mdlnJs) <= 3, "Polynomial degree is to high for this procedure";

		Return = collections.namedtuple("Return", ["alpha", "beta", "gamma"]);

		alpha = p_mdlnJs[-1];

		if(len(p_mdlnJs) >= 2):
			beta = p_mdlnJs[-2];
		else:
			beta = 0.0;

		if(len(p_mdlnJs) >= 3):
			gamma = p_mdlnJs[-3];
		else:
			gamma = 0.0;

		return Return(
			alpha=alpha,
			beta=beta,
			gamma=gamma,
			);

	def remove_nonzero(self, x, y):
		"""
			Remove arguments and values for indices where the value is 0.

			Parameters:

			* *x* --- argument vector
			* *y* --- value vector

			Return: (*x*, *y*) with zero values removed
		"""
		assert x.shape == y.shape

		nonzero_idcs = y.magnitude.nonzero();
		return (x[nonzero_idcs], y[nonzero_idcs]);

	def smooth(self, te, Je, Jodj=None, **smoothing_kwargs):
		"""
			Smooth experimental data.

			Parameters:

			* *te* --- time (with units)
			* *Je* --- (noisy) data to be de-noised (with units)
			* *Jodj* --- value to be subtracted from Je (with units)
			* All optional parameters of :func:`smoothing` may be passed.

			Return: a named tuple (*ts*, *Jts*, *mdlnJts*) 

			* *ts*  --- a vector of time steps for smoothed luminescence intensity
			* *Js* --- a vector of smoothed luminescence intensity
			* *mdlnJs*  --- a vector of negative logarithmic derivative of luminescence intensity
		"""

		Return = collections.namedtuple("Return", ["ts", "Js", "mdlnJs"])
		#plt.plot(te , Je, **self.style_data);
		#plt.yscale("log");
		#plt.xlabel(f"Time [${self.time_unit:~L}$]");
		#plt.ylabel(f"Optical output [arb. u.]");
		#self._save_fig("Je");
		#plt.show();

		if(Jodj != None):
			Jodj = (0*Je.units + Jodj); # this is to ensure Jodj unit is compatible with Je unit

		ret = smoothing(
			te = te.magnitude,
			Je = Je.magnitude,
			Jodj = Jodj.to(Je.units).magnitude,
			**smoothing_kwargs
			);

		return Return(
			ts = ret.ts * te.units,
			Js = ret.Jts * Je.units, 
			mdlnJs = ret.mdlnJts / te.units); 

class Plotter(object):
	"""
		A class for some standard auxiliary plots.
	"""
	def __init__(self, un=un_default):
		self.style_data = {'color':'#1F20E0', 'label':'orig.'};
		self.style_smoothed = {'color':'#E07E1F', 'label':'smooth.'};
		self.style_fit = {'color':'#1FE07E', 'label':'fit'};
		self.style_fit_outside_interval = {'color':'#1FE07E', "ls":"--"}; # approximation outside of fitting interval

		self.un=un;

	def Je(self, te, Je):
		"""
			Plot of experimentally measured luminescence output *Je* versus time *te*.
		"""
		plt.plot(te.magnitude, Je.magnitude, **self.style_data);
		plt.yscale("log");
		plt.xlabel(f"Time [${te.units:~L}$]");
		plt.ylabel(f"Optical output [arb. u.]");
		plt.legend();
		#plt.show();

	def JeJs(self, te, Je, ts, Js):
		"""
			Plot of experimentally measured luminescence output *Je* versus time *te* and smoothed luminescence output *Js* versus time *ts*
		"""
		assert te.units == ts.units
		assert Je.units == Js.units

		plt.plot(te.magnitude, Je.magnitude, **self.style_data);
		plt.plot(ts.magnitude, Js.magnitude, **self.style_smoothed);
		plt.yscale("log");
		plt.xlabel(f"Time [${te.units:~L}$]");
		plt.ylabel(f"Optical output [arb. u.]");
		plt.legend();
		#plt.show();

	def Jf(self, tf, Jf):
		"""
			Plot of luminescence output *Jf* versus time *tf* calculated from polynomial approximation of :math:`r_L`.
		"""
		plt.plot(tf.magnitude, Jf.magnitude, **self.style_fit);
		plt.yscale("log");
		plt.xlabel(f"Time [${tf.units:~L}$]");
		plt.ylabel(f"Optical output [arb. u.]");
		plt.legend();

	def Js(self, ts, Js):
		"""
			Plot of smoothed luminescence output *Js* versus time *ts*
		"""
		plt.plot(ts.magnitude, Js.magnitude, **self.style_smoothed);
		plt.yscale("log");
		plt.xlabel(f"Time [${ts.units:~L}$]");
		plt.ylabel(f"Optical output [arb. u.]");
		plt.legend();
		#plt.show();

	def _mdlnJ_extras(self, y, mdlnJ, mu):
		"""
			Labels, units and other details for negative logarithmic derivative of the normalized optical output plots.
		"""
		if(y.units != self.un.dimensionless):
			y_units = " $\\left[{y.units:~L}\\right]$";
		else:
			y_units = "";

		if(mu is not None):
			plt.xlabel(f"$J^{{{1/mu}}}$ {y_units}");
		else:
			plt.xlabel(f"{y_units}");
		plt.ylabel(f"-d log(J) / dt $\\left[{mdlnJ.units:~L}\\right]$");
		plt.legend(loc="best");
		#plt.show();

	def mdlnJf(self, yf, mdlnJf, mu=None, fit_interval=None):
		"""
			Plot fitted negative logarithmic derivative of the normalized optical output *mdlnJs* versus chosen argument *ys*.
			Generally *ys* should be :math:`J_L^{1/\\mu}`, so if a correct *mu* is provided, a proper x-axis label will be added. Parameter *mu* is not used otherwise.

		"""
		plt.plot(yf.magnitude , mdlnJf.magnitude, **self.style_fit);
		self._mdlnJ_extras(yf, mdlnJf, mu);

		if(fit_interval is not None):
			fit_interval = fit_interval.to(yf.units);
			plt.axvline(x = fit_interval[0].magnitude, color=self.style_fit["color"], ls="--", lw=2) # granica fitowania
			plt.axvline(x = fit_interval[1].magnitude, color=self.style_fit["color"], ls="--", lw=2) # granica fitowania



	def mdlnJs(self, ys, mdlnJs, mu=None):
		"""
			Plot (smoothed) negative logarithmic derivative of the normalized optical output *mdlnJs* versus chosen argument *ys*.
			Generally *ys* should be :math:`J_L^{1/\\mu}`, so if a correct *mu* is provided, 
		"""
		plt.plot(ys.magnitude , mdlnJs.magnitude, **self.style_smoothed);
		self._mdlnJ_extras(ys, mdlnJs, mu);

	def rel_Je_vs_Js(self, te, Je, ts=None, Js=None, tf = None, Jf = None):
		"""
			Plot of relative difference of experimentally measured luminescence output *Je* (time *te*) and smoothed luminescence output *Js* (time *ts*; if *None*, then it is omitted).
			Also *Jf* (time *tf*) may be passed to plot ODE solution result base on ploynomial fit.
		"""
		if(ts is not None):
			assert te.units == ts.units
			assert Je.units == Js.units
		if(tf is not None):
			assert te.units == tf.units
			assert Je.units == Jf.units

		time_unit = te.units;

		te = te.magnitude;
		Je = Je.magnitude;
		if(ts is not None):
			ts = ts.magnitude;
			Js = Js.magnitude;

		if(tf is not None):
			tf = tf.magnitude;
			Jf = Jf.magnitude;

		if(ts is not None):
			f_Js = scipy.interpolate.interp1d(ts, Js, kind="linear");
			plt.plot(te, (Je - f_Js(te)) / Je, label="(orig. - smooth.) / orig.", color=self.style_smoothed["color"]);

		if(tf is not None):
			f_Jf = scipy.interpolate.interp1d(tf, Jf, kind="linear");
			plt.plot(te, (Je - f_Jf(te)) / Je, label="(orig. - fit) / orig.", color=self.style_fit["color"]);
		
		plt.plot(te, 0*te, **self.style_data);
		plt.yscale("linear");
		plt.xlabel(f"Time [${time_unit:~L}$]");
		plt.ylabel(f"Rel. difference");
		plt.legend();
		#plt.show();

class PrintInfo(object):
	"""
		A class for displaying information..
	"""
	def __init__(self, un=un_default):
		self.un=un;

	def abc(self, A, B, C):
		"""
			Print ABC recombination constants
		"""
		display(Latex(f"$A={A:~Le}$"));
		display(Latex(f"$B={B:~Le}$"));
		display(Latex(f"$C={C:~Le}$"));

	def mu(self, mu):
		"""
			Print dependence of polynomial argument on *mu*.
		"""
		display(Latex(f"$y = J^{{{1/mu}}}$"));

	def poly_coefs(self, alpha, beta, gamma):
		"""
			Print polynomial coefficients for :math:`r_L(y) = \alpha + \beta y + \gamma y^2`
		"""
		display(Latex(f"$\\alpha={alpha:~Le}$"));
		display(Latex(f"$\\beta={beta:~Le}$"));
		display(Latex(f"$\\gamma={gamma:~Le}$"));


def exp_plus_c_fit(x, fx, c0):
	def fun(c, x, y):
		return (numpy.exp(c[0] * x + c[1]) + c[2]) - y;

	res = scipy.optimize.least_squares(fun, c0, args=(x,fx))
	c = res.x;
	#print(c0, c);
	fit = fun(c,x,0);
	return (fit,c);

def load_exp_data(plik, kol_t=0, kol_J = 1, delimiter=' ', **kwords):
	t_J = numpy.genfromtxt(plik, delimiter=delimiter, **kwords)

	t = t_J[:,kol_t];# * un.ns;
	J = t_J[:,kol_J];

	return (t,J);

def lznk_fit(x, fx, st):
	A = numpy.vstack([x**i for i in range(st+1)]).T;
	c = numpy.linalg.lstsq(A,fx,rcond=None)[0];
	#print(c);
	fit = sum(c[i] * x**i for i in range(st+1));
	return (fit,c);

def lznk_exp_fit(x, fx, c0, st):
	def fun(c, x, y):
		return numpy.exp(sum([c[i] * (x**i) for i in range(st+1)])) - y;

	res = scipy.optimize.least_squares(fun, c0, args=(x,fx))
	c = res.x;
	#print(c0, c);
	fit = numpy.exp(sum([c[i] * (x**i) for i in range(st+1)]));
	return (fit,c);

def new_approx(te,Je,initr, endr, st, nielinznk=False, gaussian=False):

	assert(initr>=1);
	assert(initr<=endr);

	if(endr >= len(Je)):
		raise RuntimeError("Final averaging interval is larger than total data length");

	if(gaussian):
		n_gaussian = 5
		Jg = scipy.ndimage.filters.gaussian_filter1d(Je, n_gaussian, mode='nearest');
		Jg[:n_gaussian] = Je[:n_gaussian];
		Jg[-n_gaussian:] = Je[-n_gaussian:];
	else:
		Jg = Je

	dlnJts = [];
	dJts = [];
	Jts = [];
	ts = [];
	for i in range(len(Jg)):
		#prom =  initr + endr * (i - initr) / (len(Jg) - initr - endr);
		fac = min(1, float(i) / (len(Jg) - endr))
		prom =  int(initr*(1-fac) + endr*fac);

		zakr, poz = zakres(i, prom, len(Jg)); 
		#if(i<=initr):
		#	#print(1);
		#	zakr = range(2*initr+1);
		#	poz = i;
		#elif(i > len(Jg)-initr-endr):
		#	#print(2);
		#	zakr = range(len(Jg)-2*(initr+endr) + 1,len(Jg));
		#	poz = i + 2*initr + 2*endr - len(Jg) -1;
		#else:
		#	#print(3);
		#	zakr = range(i-prom, i+prom+1);
		#	poz = len(zakr)/2;

		#print(zakr);
		##print(i,poz, zakr[poz]);

		assert(i == zakr[poz]);

		tt = te[zakr];
		#print(te[i], tt[poz]);

		if(nielinznk):
			_,c0 = lznk_fit(tt, numpy.log(Je[zakr]), st=st);
			Jt,_ = lznk_exp_fit(tt, Jg[zakr], c0=c0, st=st);
			dJt = numpy.gradient(Jt,tt);
			dlnJt = dJt/Jt;
		else:
			lnJt,_ = lznk_fit(tt, numpy.log(Jg[zakr]), st=st);
			dlnJt = numpy.gradient(lnJt,tt);
			Jt = numpy.exp(lnJt);
			dJt = numpy.gradient(Jt,tt);
		
		ts.append(te[i]);
		Jts.append(Jt[poz]);
		dlnJts.append(dlnJt[poz]);
		if(i<len(Jg)-1):
			dJts.append((Jt[poz+1]-Jt[poz])/(tt[poz+1]-tt[poz]));
		else:
			dJts.append(dJts[-1]);

	dlnJts = numpy.array(dlnJts);
	dJts = numpy.array(dJts);
	Jts = numpy.array(Jts);
	ts = numpy.array(ts);

	return (ts, Jts, dlnJts, Jt);

def polyval(ps, x):
	"""
		Compute polynomial from values acounting for coefficients with units.

		Parameters:

		* *ps* --- array of polynomial coefficients, starting with highest-order coefficients; individual elements of this array can have units
		* *x* --- vector of arguments
	"""
	e = [p*(x**i) for i, p in enumerate(reversed(ps))];
	return sum(e);

def read_data_headers(plik, delimiter=' ', **kwords):
	with open(plik, 'r') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=delimiter, quotechar='"')
		linia=0;
		for row in csvreader:
			#print(row)
			linia += 1;
			if(linia == 1):
				names = row;
			elif(linia == 2):
				labels = row;
			elif(linia == 3):
				break;

		assert(len(names) == len(labels));


	Column = collections.namedtuple('Column', ['number', 'name', 'label']);

	columns = dict();

	for number, (name, label) in enumerate(zip(names, labels)):
		columns[number] = Column(number = number, name = name, label = label);

	return columns;

# Zwraca zakres indeksów zawarty w [0,N), który jest długości 2*n+1 i zwykle środkiem jest liczba "i", poza sytuacją, gdy "i" jest zbyt blisko brzegów i po prostu zwracany jest zakres przy brzegu o długości średnicy
def zakres(i,n, N):
	srednica = 2*n + 1;
	if(N <= srednica):
		zakr = range(N);
		poz = i;
	else:
		if(i<n):
			zakr = range(srednica);
			poz = i;
		elif(i>=N-n):
			zakr = range(N-srednica, N);
			poz = 2*n + 1 -(N-i);
		else:
			zakr = range(i-n, i+n+1);
			poz = n;

	return (zakr, poz);


def smoothing(
	te, Je,
	st=1,
	n0=1,
	n1=100,
	iters=1,
	Jodj=None,
	nielinznk=False,
	gaussian=False,
	averageoutfirst=False,
	verbose=False,
	plots=False,
	title = "",
	tpoint=None,
	):
	"""
		Smoothing of noisy numerical data.

		Parameters:

		* *te* --- a vector of (increasing) experimental time steps
		* *Je* --- a vector of corresponding experimental luminescence intensity
		* *st* --- polynomial degree for least-squares fitting
		* *n0* --- initial averaging interval
		* *n1* --- final averaging interval
		* *iters* --- multiple iterations of the averaging
		* *Jodj* --- constant subtracted from the experimental J data (default: *None*, find automatically)
		* *nielinznk* --- use nonlinear least squares (default is to use linear)
		* *gaussian* --- use Gaussian filtering (default: no)
		* *averageoutfirst* --- first average out J, then subtract Jodj (default is to subtract Jodj from J and then to average out)
		* *verbose* --- show text
		* *plots* --- show plots
		* *title* --- title of produced figures
		* *tpoint* --- auxiliary indicatory point to be put on the figures (default: *None*, no point)

		Return: a named tuple of:

		* *ts*  --- a vector of time steps for smoothed luminescence intensity
		* *Jts* --- a vector of smoothed luminescence intensity
		* *mdlnJts*  --- a vector of negative logarithmic derivative of luminescence intensity
		* *te* --- a vector of experimental time steps
		* *Je* --- a vector of corresponding experimental luminescence intensity with *Jodj* subtracted
		* *Je* --- a vector of corresponding original experimental luminescence intensity
		* *mdlnJe*  --- a vector of negative logarithmic derivative of luminescence intensity, computed directly from the experimental data


	"""

	SmoothingReturn = collections.namedtuple("SmoothingReturn", ["ts", "Jts", "mdlnJts", "te", "Je", "mdlnJe", "Jo"])

	# we will work on copies
	te = numpy.copy(te);
	Je = numpy.copy(Je);

	# original data, stored separately
	Jo = numpy.copy(Je);

	if(Jodj is None):
		nfit = n1;
		(Jodjfit,c) = exp_plus_c_fit(te[-nfit:], Je[-nfit:], [0,0,0]);
		Jodj = c[2];
		if(verbose):
			print("Jodj found: %f"%Jodj);

		if(plots):
			plt.plot(te, Je, color = 'green', label="$J$ experimental");
			plt.plot(te[-nfit:], Jodjfit, color='blue', label="$J$ fit");
			plt.yscale('log');
			plt.legend(loc='best');
			plt.xlabel("$t$");
			plt.title(title);
			plt.show();

	Je = numpy.abs(Je - Jodj);
	dJe = numpy.gradient(Je, te);
	dlnJe = dJe/Je;


	for iter in range(iters):
		if(iter==0):
			tin = te;
			if(averageoutfirst):
				Jin = Jo;
			else:
				Jin = Je;
		else:
			tin = ts;
			Jin = Jts;


		(ts, Jts, dlnJts, _) = new_approx(tin, Jin, n0, n1, st=st, nielinznk=nielinznk, gaussian=gaussian);

	if(averageoutfirst):
		Jts -= Jodj;

	if(tpoint is not None):
		point = (numpy.abs(ts-tpoint)).argmin();

	if(plots):
		plt.plot(te, Je, color = 'green', label="$J$ experimental");
		plt.plot(ts, Jts, color='red', label="$J$ denoised");
		if(tpoint is not None):
			plt.plot(ts[point], Jts[point], 'ro');
		plt.yscale('log');
		plt.legend(loc='best');
		plt.xlabel("$t$");
		plt.title(title);
		plt.show();

		#dJe = numpy.gradient(Je, te);
		#plt.plot(te, dJe, color='green', label=r"$\frac{d J}{d t}$ experimental");
		#plt.plot(ts, dlnJts*Jts, color='red', label=r"$\frac{d J}{d t}$ denoised");
		#if(tpoint is not None):
		#	plt.plot(ts[point], dlnJts[point]*Jts[point], 'ro');
		#plt.yscale('linear');
		#plt.legend(loc='best');
		#plt.xlabel("$t$");
		#plt.title(title);
		#plt.show();

		plt.plot(Je , -dlnJe , color='green', label=r"$-\frac{d \log(J)}{d t}$ experimental");
		plt.plot(Jts, -dlnJts, color='red',   label=r"$-\frac{d \log(J)}{d t}$ denoised");
		if(tpoint is not None):
			plt.plot(Jts[point], -dlnJts[point], 'ro');
		plt.yscale('linear');
		plt.ylim(0.5*min(-dlnJts), 1.15*max(-dlnJts));
		plt.legend(loc='best');
		plt.xlabel("$J$");
		plt.title(title);
		plt.show();

		plt.plot(numpy.sqrt(Je),  -dlnJe , color='green', label=r"$-\frac{d \log(J)}{d t}$ experimental");
		plt.plot(numpy.sqrt(Jts), -dlnJts, color='red',   label=r"$-\frac{d \log(J)}{d t}$ denoised");
		if(tpoint is not None):
			plt.plot(numpy.sqrt(Jts[point]), -dlnJts[point], 'ro');
		plt.yscale('linear');
		plt.ylim(0.5*min(-dlnJts), 1.15*max(-dlnJts));
		plt.legend(loc='best');
		plt.xlabel(r"$\sqrt{J}$");
		plt.title(title);
		plt.show();

	return SmoothingReturn(
		ts = ts,
		Jts = Jts,
		mdlnJts = -dlnJts,
		te = te,
		Je = Je,
		mdlnJe = -dlnJe,
		Jo = Jo,
		);


def main():

	parser = argparse.ArgumentParser(description='(Linear-)least-squares-based fitting algorithm.');
	parser.add_argument('sourcefile', type=str, help='CSV data file')
	parser.add_argument('N', type=int, nargs='?', default=0, help='Column number, indexed from 0')
	parser.add_argument('--ct', type=int, default=0, help='Column number for time (default: 0)')
	parser.add_argument('--delimiter', type=str, default='\t', help='CSV column delimiter')
	parser.add_argument('--title', type=str, default="", help='Title of produced figures')
	parser.add_argument('--savecsv', type=str, default=None, help='Save results to this file')
	parser.add_argument('--n0', type=int, default=1, help='Initial averaging interval')
	parser.add_argument('--n1', type=int, default=100, help='Final averaging interval')
	parser.add_argument('--iters', type=int, default=1, help='Multiple iterations of the averaging')
	parser.add_argument('--Jodj', type=float, default=None, help='Constant subtracted from the experimental J data')
	parser.add_argument('--tmin', type=float, default=-numpy.inf, help='Minimal t')
	parser.add_argument('--tmax', type=float, default=numpy.inf, help='Maximal t')
	parser.add_argument('--tpoint', type=float, default=None, help='Auxiliary indicatory point to be put on the figures')
	parser.add_argument('--st', type=int, default=1, help='Polynomial degree for least-squares fitting')
	parser.add_argument('--nielinznk', action='store_true', default=False, help='Use nonlinear least squares (default is to use linear)')
	parser.add_argument('--gaussian', action='store_true', default=False, help='Use Gaussian filtering (default: no)')
	parser.add_argument('--averageoutfirst', action='store_true', default=False, help='First average out J, then subtract Jodj (default is to subtract Jodj from J and then to average out)')
	parser.add_argument('--noplots', action='store_true', default=False, help='Suppress plots')
	parser.add_argument('--ignorezeros', action='store_true', default=False, help='Ignore measurements which were equal to 0')
	args = parser.parse_args()

	plots = not args.noplots;

	columns = read_data_headers(args.sourcefile, delimiter=args.delimiter)

	if(args.N == 0):

		print("Specify the column to average out as a last parameter.\nList of columns:");
		for col in sorted(columns.values(), key= lambda col: col.number):
			print("%3d: %3s (%s)"%((col.number, col.name, col.label)));

		if(len(columns)==1):
			print("Only one column in this file? Perhaps --delimiter is not set correctly?".format(args=args));

	else:
		if(args.N not in columns):
			print("No column {args.N}. Perhaps --delimiter is not set correctly?".format(args=args));
			sys.exit(2);

		colx = columns[args.ct];
		col = columns[args.N];
		print("Using columns:");
		print("X: %d, %s (%s)"%((colx.number, colx.name, colx.label)));
		print("Y: %d, %s (%s)"%((col.number, col.name, col.label)));

		title = "%s --- %s (%s)"%((args.title, col.name, col.label));

		[te, Je] = load_exp_data(plik=args.sourcefile, kol_t=args.ct, kol_J=args.N, delimiter=args.delimiter, skip_header=1);

		if(args.ignorezeros):
			# remove zeros
			nonzero_idcs = Je.nonzero();
			te = te[nonzero_idcs];
			Je = Je[nonzero_idcs];


		indices = numpy.where(numpy.logical_and(te >= args.tmin, te <= args.tmax));
		te = te[indices];
		Je = Je[indices];



		ret = smoothing(
			te, Je,
			st = args.st,
			n0 = args.n0,
			n1 = args.n1,
			iters = args.iters,
			Jodj = args.Jodj,
			nielinznk = args.nielinznk,
			gaussian = args.gaussian,
			averageoutfirst = args.averageoutfirst,
			verbose = True,
			plots = plots,
			title = title,
			tpoint = args.tpoint,
			);

		if(args.savecsv is not None):
			numpy.savetxt(args.savecsv, numpy.column_stack((ret.te, ret.Jo, ret.Je, ret.mdlnJe, ret.ts, ret.Jts, ret.mdlnJts)),
				fmt='%15.7e',
				delimiter=args.delimiter,
				header=args.delimiter.join(('%15s'%s for s in ('te', 'Jo', 'Je', '-dlnJe', 'ts', 'Jts', '-dlnJts'))),
				comments='',
				);

if __name__ == "__main__":
	main();
